# path: framework/policy_manager.py

import threading
import time
import ray
from typing import Any, Dict, Optional

from compression.controller import CompressionController
from compression.pipeline import CompressionPipeline
from compression.policy import CompressionPolicy
from compression.base import BaseCompressor
from enum import Enum
from ray.rllib.utils.framework import try_import_torch
torch, _ = try_import_torch()


# ============================================================
# Compile Mode
# ============================================================
class CompileMode(Enum):
    NONE = "none"
    SYNC = "sync"
    ASYNC = "async"


# ============================================================
# PolicyManager - Glue between RLlib & Compression System
# ============================================================
class PolicyManager:
    """
    Manages compression pipeline and model synchronization between training and inference.
    
    Responsibilities:
        - Manages compression pipeline & controller (sync/async)
        - Extracts backbone from RLlib training model
        - Triggers compression based on policy
        - Performs async model swapping
        - Broadcasts compressed backbone to all rollout workers
        - Handles Teacher-Student architecture for pruning
    
    
    Usage:
        manager = PolicyManager(algo, compressors, CompileMode.ASYNC, trigger_every=5)
        manager.maybe_swap(epoch)
        meta = manager.maybe_trigger(epoch)
    """

    def __init__(self,
                 algo,
                 compressors: [BaseCompressor],
                 mode: CompileMode = CompileMode.NONE,
                 trigger_every: int = 5,
                 enable_diff_check: bool = True,
                 infer_output_index: int = 0,
                 compile_training_backbone: bool = False,
                 device: str = "cpu",
                 async_warmup: bool = True,
                 min_epoch_before_compress: int = 0,
                 prune_training_model: bool = False):

        self.algo = algo
        self.mode = mode

        if not compressors:
            raise ValueError("PolicyManager requires at least one compressor.")
        if infer_output_index < 0 or infer_output_index >= len(compressors):
            raise ValueError("infer_output_index 超出了 compressors 范围。")

        self.device = self._resolve_device(device)

        self.compressors = compressors
        self.async_warmup = async_warmup

        # compression policy
        self.policy = CompressionPolicy(
            trigger_every=trigger_every,
            enable_diff_check=enable_diff_check,
            min_epoch_before_compress=min_epoch_before_compress
        )

        # pipeline + controller
        self.pipeline = CompressionPipeline(compressors, self.policy)
        self.model_lock = threading.Lock()
        self.controller = CompressionController(self.pipeline, mode, self.model_lock)

        # RLlib training model
        self.train_model = self.algo.get_policy().model

        self.infer_output_index = infer_output_index
        self.infer_compressor_name = compressors[infer_output_index].__class__.__name__
        self._supports_weight_sync = bool(
            getattr(compressors[infer_output_index], "supports_weight_sync", False)
        )

        # Current inference backbone used by samplers
        self.current_infer_model: Optional[Any] = None

        # Metadata from most recent compression (latency, sparsity, etc.)
        self.last_meta: Optional[Dict[str, Any]] = None

        self._compile_training_backbone_flag = compile_training_backbone
        self._training_backbone_compiled = False
        
        self._current_masks = None
        self._prune_training_model = prune_training_model
        
        if self._compile_training_backbone_flag:
            self._compile_training_backbone_once()

    # ------------------------------------------------------------------
    # Broadcast inference model to rollout workers
    # ------------------------------------------------------------------
    def _broadcast_inference_model(self, model, warmup=False, update_only=False, also_update_training=False):
        """
        Broadcast the given inference backbone to all rollout workers.
        
        For pruning:
            - Masks are broadcast separately and reapplied on each worker
            - This ensures register_hook persists on worker models
        
        Args:
            model: Compressed/compiled inference model to broadcast
            warmup: Whether to warmup compiled model on workers
            update_only: Only update weights, don't replace entire model
            also_update_training: Also update local worker's training backbone
                                 (used for structure-changing compression like structure pruning)
        
        Note: Your CustomPolicyNet must implement set_compiled_backbone()
        """
        workers = self.algo.workers.remote_workers()
        state_dict = None
        serializable_model = model
        
        # ✅ Solution 2: Pass masks separately for reapplication on workers
        # This is critical for pruning: hooks cannot be serialized, must be recreated
        masks_to_apply = None
        if hasattr(self, '_current_masks') and self._current_masks is not None:
            # Convert masks to CPU tensors for serialization
            masks_to_apply = {k: v.cpu() for k, v in self._current_masks.items()}
            print(f"[PolicyManager] 📦 Preparing to broadcast {len(masks_to_apply)} pruning masks")
        
        # ✅ Fix: If model is torch.compile'd, only pass state_dict (avoid serialization failure)
        if model is not None and hasattr(model, '_orig_mod'):
            # This is a compiled model, extract state_dict
            update_only = True
            try:
                state = model.state_dict()
                state_dict = {k: (v.detach().cpu() if torch.is_tensor(v) else v)
                              for k, v in state.items()}
                serializable_model = None  # Don't pass compiled model object
                if masks_to_apply:
                    print(f"[PolicyManager] 🔧 Detected compiled+pruned model, using state_dict + masks")
                else:
                    print(f"[PolicyManager] 🔧 Detected compiled model, using state_dict for serialization")
            except Exception as exc:
                print(f"[PolicyManager] ⚠️ Failed to extract state_dict from compiled model: {exc}")
                serializable_model = model._orig_mod  # Try using original model
        elif update_only and model is not None:
            try:
                state = model.state_dict()
                state_dict = {k: (v.detach().cpu() if torch.is_tensor(v) else v)
                              for k, v in state.items()}
            except Exception as exc:
                print(f"[PolicyManager] ⚠️ Failed to capture compiled state for update-only swap: {exc}")
                update_only = False
                state_dict = None

        def _set(worker):
            def inner(policy, pid):
                did_update = False
                if update_only and hasattr(policy.model, "update_compiled_backbone_weights") and state_dict is not None:
                    try:
                        policy.model.update_compiled_backbone_weights(state_dict)
                        did_update = True
                    except Exception as exc:
                        print(f"[PolicyManager] ⚠️ update_only failed on worker, fallback to full swap: {exc}")
                if not did_update and serializable_model is not None and hasattr(policy.model, "set_compiled_backbone"):
                    policy.model.set_compiled_backbone(serializable_model)
                    if warmup and hasattr(policy.model, "warmup_compiled_backbone"):
                        policy.model.warmup_compiled_backbone()
                
                # ✅ Solution 2: Reapply masks on worker side
                # Critical for pruning: This recreates masked weights on worker's model
                # Use inference_only=True to avoid gradient hook overhead (workers never train!)
                if masks_to_apply is not None:
                    try:
                        from compression.mask_prune_compressor import apply_masks_to_backbone
                        # Get actual backbone (may be wrapped in compiled wrapper)
                        backbone = policy.model.backbone
                        if hasattr(backbone, '_orig_mod'):
                            backbone = backbone._orig_mod
                        # Apply masks - this will:
                        # 1. Zero out masked weights
                        # 2. Skip gradient hooks (inference_only=True) to avoid overhead
                        # 
                        # Key: inference_only=True gives ~2-3% speedup by skipping unnecessary hooks!
                        # Rollout workers NEVER do backward passes, so hooks are pure overhead
                        apply_masks_to_backbone(backbone, masks_to_apply, inference_only=True)
                        # Note: Don't print per-worker to avoid log spam
                    except Exception as exc:
                        print(f"[PolicyManager] ⚠️ Failed to apply masks on worker: {exc}")
                
                return 1
            worker.foreach_policy(inner)
            return 1

        if workers:
            ray.get([w.apply.remote(_set) for w in workers])

        # 剪枝等结构变化的压缩：同时更新训练模型
        if also_update_training and model is not None:
            self._update_training_backbone(model)

        if not update_only:
            msg = "[Broadcast] 📤 Inference backbone updated on all sampler workers."
            if also_update_training:
                msg += " (Training backbone also updated)"
            print(msg)

    def _update_training_backbone(self, new_backbone):
        """
        Update training backbone on all workers (for structure-changing compression).
        
        Note: This is for STRUCTURE pruning, NOT mask pruning!
        
        For mask pruning:
            - This method is NOT called
            - Training model stays unchanged (unless prune_training_model=True)
            - Only inference model gets masked
        
        For structure pruning:
            - This method IS called
            - Both training and inference models change structure
            - Needed to keep them synchronized for weight updates
        """
        if new_backbone is None:
            return
        
        try:
            # 获取新 backbone 的实际模型（如果被 compile 包装了）
            actual_backbone = getattr(new_backbone, "_orig_mod", new_backbone)
            
            # 获取 state_dict（用于同步到 remote workers）
            backbone_state = actual_backbone.state_dict()
            cpu_state = {k: (v.detach().cpu() if torch.is_tensor(v) else v)
                        for k, v in backbone_state.items()}
            
            # 获取新模型的结构信息
            new_structure = {
                "in_dim": actual_backbone.hidden_layers[0].in_features if len(actual_backbone.hidden_layers) > 0 else None,
                "num_outputs": actual_backbone.policy_head.out_features,
                "hidden_dims": [layer.out_features for layer in actual_backbone.hidden_layers],
                "use_residual": actual_backbone.use_residual,
            }
            
            # 1. 更新 local worker
            local_worker = self.algo.workers.local_worker()
            local_policy = local_worker.get_policy()
            
            # 保存是否之前被编译过
            was_compiled = hasattr(local_policy.model.backbone, "_orig_mod")
            
            # 创建新 backbone 的副本用于训练
            import copy
            training_backbone = copy.deepcopy(actual_backbone)
            training_backbone.to(self.device)
            
            # 如果之前训练 backbone 被编译过，新的也编译
            if was_compiled and self._compile_training_backbone_flag:
                training_backbone = torch.compile(training_backbone, backend="inductor")
            
            # 替换 local backbone
            local_policy.model.backbone = training_backbone
            
            # 更新元信息
            local_policy.model.hidden_dims = new_structure["hidden_dims"]
            
            # 2. 更新所有 remote workers 的训练 backbone
            workers = self.algo.workers.remote_workers()
            
            def _update_remote_training(worker):
                def inner(policy, pid):
                    # 重建 backbone
                    from models.policy import PolicyBackbone
                    new_bb = PolicyBackbone(
                        new_structure["in_dim"],
                        new_structure["num_outputs"],
                        new_structure["hidden_dims"],
                        new_structure["use_residual"]
                    )
                    new_bb.load_state_dict(cpu_state)
                    
                    # 替换
                    policy.model.backbone = new_bb
                    policy.model.hidden_dims = new_structure["hidden_dims"]
                    return 1
                
                worker.foreach_policy(inner)
                return 1
            
            if workers:
                ray.get([w.apply.remote(_update_remote_training) for w in workers])
            
            # 3. 关键：重置优化器状态（避免维度不匹配）
            self._reset_optimizer_after_prune()
            
            print(f"[PolicyManager] 🔄 Training backbone updated: {new_structure['hidden_dims']}")
            
        except Exception as exc:
            import traceback
            print(f"[PolicyManager] ⚠️ Failed to update training backbone: {exc}")
            traceback.print_exc()
    
    def _reset_optimizer_after_prune(self):
        """
        剪枝后重置优化器状态
        
        关键：剪枝改变了模型维度，优化器的 momentum/variance 等状态会维度不匹配
        必须重置！
        """
        try:
            local_worker = self.algo.workers.local_worker()
            local_policy = local_worker.get_policy()
            
            # 调试：检查优化器结构
            print(f"[PolicyManager] 🔍 Checking optimizer...")
            print(f"  hasattr _optimizer: {hasattr(local_policy, '_optimizer')}")
            print(f"  hasattr _optimizers: {hasattr(local_policy, '_optimizers')}")
            
            # RLlib 的优化器可能在不同的属性中
            optimizer_to_reset = None
            lr = 5e-5  # 默认学习率
            
            # 尝试多种可能的优化器位置
            if hasattr(local_policy, "_optimizer") and local_policy._optimizer is not None:
                if isinstance(local_policy._optimizer, tuple):
                    optimizer_to_reset = local_policy._optimizer[0]
                else:
                    optimizer_to_reset = local_policy._optimizer
            elif hasattr(local_policy, "_optimizers") and local_policy._optimizers:
                # 有些 RLlib 版本用 _optimizers (list)
                optimizer_to_reset = local_policy._optimizers[0]
            
            if optimizer_to_reset is not None:
                # 获取学习率
                if hasattr(optimizer_to_reset, 'param_groups'):
                    lr = optimizer_to_reset.param_groups[0]['lr']
                
                # 重新初始化优化器
                import torch.optim as optim
                new_optimizer = optim.Adam(local_policy.model.parameters(), lr=lr)
                
                # 替换
                if hasattr(local_policy, "_optimizer"):
                    if isinstance(local_policy._optimizer, tuple):
                        local_policy._optimizer = (new_optimizer, local_policy._optimizer[1])
                    else:
                        local_policy._optimizer = new_optimizer
                elif hasattr(local_policy, "_optimizers"):
                    local_policy._optimizers[0] = new_optimizer
                
                print(f"[PolicyManager] ✅ Optimizer reset with lr={lr}")
            else:
                print(f"[PolicyManager] ⚠️ No optimizer found to reset")
            
        except Exception as exc:
            import traceback
            print(f"[PolicyManager] ⚠️ Failed to reset optimizer: {exc}")
            traceback.print_exc()

    # ------------------------------------------------------------------
    # 异步模式：在每个 epoch 开头尝试 swap（若异步线程已完成）
    # ------------------------------------------------------------------
    def maybe_swap(self) -> Optional[Dict[str, Any]]:
        if self.mode != CompileMode.ASYNC:
            return None

        outputs, meta = self.controller.try_swap()
        if outputs is None:
            return None

        infer_model = self._select_infer_model(outputs)
        if infer_model is None:
            return None

        self.current_infer_model = infer_model
        self.last_meta = meta

        warmup = (self.mode == CompileMode.ASYNC and self.async_warmup)
        update_only = self._should_update_only(meta)
        
        # 调试已禁用（避免刷屏）
        # print(f"[DEBUG] maybe_swap: meta keys = {list(meta.keys()) if meta else None}")
        
        # 检测是否是结构变化的压缩（如剪枝）
        also_update_training = self._is_structure_changing_compression(meta)
        
        t0 = time.time()
        self._broadcast_inference_model(
            infer_model, 
            warmup=warmup and not update_only, 
            update_only=update_only,
            also_update_training=also_update_training
        )
        swap_latency = time.time() - t0
        if meta is None:
            meta = {}
        meta.setdefault("SwapLatency", swap_latency)
        if not update_only:
            print("[AsyncCompile] 🔁 Swapped inference model.")
        return meta

    def push_weight_update(self):
        """
        将训练模型最新的 backbone 权重同步到已存在的推理 backbone。
        仅对支持纯权重更新的压缩器（例如 compile）启用。
        
        注意：对于 Mask Pruning，需要在同步后重新应用mask！
        """
        if not self._supports_weight_sync:
            return
        if self.current_infer_model is None:
            return

        snapshot = self._snapshot_train_backbone()
        if snapshot is None:
            return

        # 检查结构是否一致（剪枝会改变结构）
        if not self._check_structure_match(snapshot):
            # 结构不匹配，跳过同步（等待异步压缩完成）
            return

        # 先更新本地推理模型，避免下一次广播仍旧是旧权重
        self._load_state_into_infer(snapshot)
        
        # Mask pruning 特殊处理：重新应用mask
        self._reapply_masks_after_weight_sync()

        workers = self.algo.workers.remote_workers()
        if not workers:
            return

        def _update(worker):
            def inner(policy, pid):
                if hasattr(policy.model, "update_compiled_backbone_weights"):
                    try:
                        policy.model.update_compiled_backbone_weights(snapshot)
                    except Exception as exc:
                        print(f"[PolicyManager] ⚠️ Weight push failed on worker, skipping: {exc}")
                return 1

            worker.foreach_policy(inner)
            return 1

        ray.get([w.apply.remote(_update) for w in workers])

    # ------------------------------------------------------------------
    # 同步/异步触发压缩
    # ------------------------------------------------------------------
    def maybe_trigger(self, epoch: int) -> Optional[Dict[str, Any]]:
        if self.mode == CompileMode.NONE:
            return None
        # 同步模式 —— 立即执行
        if self.mode == CompileMode.SYNC:
            outputs, meta = self.controller.run_sync(self.train_model, epoch)
            if outputs is None:
                return None

            infer_model = self._select_infer_model(outputs)
            if infer_model is None:
                return None

            self.current_infer_model = infer_model
            self.last_meta = meta

            update_only = self._should_update_only(meta)
            also_update_training = self._is_structure_changing_compression(meta)
            self._broadcast_inference_model(
                infer_model, 
                warmup=False, 
                update_only=update_only,
                also_update_training=also_update_training
            )
            print("[SyncCompile] ✅ Compiled & swapped immediately.")
            return meta

        # 异步模式 —— 触发后台线程
        elif self.mode == CompileMode.ASYNC:
            self.controller.trigger_async(self.train_model, epoch)
            return None

        return None

    # ------------------------------------------------------------------
    # 获取最近压缩信息
    # ------------------------------------------------------------------
    def get_last_meta(self):
        return self.last_meta

    # ------------------------------------------------------------------
    # 供 Trainer 访问的辅助
    # ------------------------------------------------------------------
    def _select_infer_model(self, outputs):
        if not outputs:
            return None
        if self.infer_output_index >= len(outputs):
            return None
        return outputs[self.infer_output_index]

    def get_infer_compressor_name(self) -> str:
        return self.infer_compressor_name

    def _should_update_only(self, meta: Optional[Dict[str, Any]]) -> bool:
        if not meta:
            return False
        name = self.get_infer_compressor_name()
        info = meta.get(name)
        if not info:
            return False
        return bool(info.get("reused"))
    
    def _is_structure_changing_compression(self, meta: Optional[Dict[str, Any]]) -> bool:
        """
        Detect if compression changes model structure and handle pruning modes.
        
        Pruning Mode (controlled by prune_training_model flag):
        - prune_training_model=True: Both training and inference models are pruned
        - prune_training_model=False: Only inference model is pruned (training model stays full)
        
        Returns:
            False for mask pruning (doesn't change structure)
            True for structure pruning (changes model dimensions)
        """
        if not meta:
            return False
        
        # ✅ Iterate through all compressor info in meta
        # This captures outputs from all compressors in pipeline
        for comp_name, info in meta.items():
            if not isinstance(info, dict):
                continue
            
            compression_type = info.get("type", "")
            
            # ✅ Mask Pruning: Check prune_training_model flag
            if "mask_prune" in compression_type.lower():
                masks = info.get("masks")
                if masks is not None:
                    self._current_masks = masks
                    sparsity = info.get('actual_sparsity', 0) * 100
                    
                    # Check prune_training_model flag
                    if self._prune_training_model:
                        print(f"[PolicyManager] 🔥 Both training and inference models pruned (sparsity: {sparsity:.1f}%)")
                        # Apply masks to training model
                        self._apply_masks_to_training(masks)
                    else:
                        print(f"[PolicyManager] 🎯 Only inference model pruned (sparsity: {sparsity:.1f}%), training model stays full")
                
                return False  # Mask pruning doesn't change structure
            
            # Structure Pruning: Actually changes model dimensions
            if "structure_prune" in compression_type.lower():
                return True
        
        return False
    
    def _apply_masks_to_training(self, masks: Dict[str, torch.Tensor]):
        """
        Apply masks to training model.
        
        This is ONLY called when prune_training_model=True.
        When True: Both training and inference models use pruned weights (strictly on-policy).
        When False: Only inference model is pruned, training model stays full.
        """
        if masks is None:
            return
        
        try:
            from compression.mask_prune_compressor import apply_masks_to_backbone
            
            # Get local training model
            local_worker = self.algo.workers.local_worker()
            local_policy = local_worker.get_policy()
            training_backbone = local_policy.model.backbone
            
            # Unwrap if compiled
            if hasattr(training_backbone, '_orig_mod'):
                training_backbone = training_backbone._orig_mod
            
            # Apply masks with hooks (training model needs gradient masking)
            apply_masks_to_backbone(training_backbone, masks, inference_only=False)
            
            # Also apply to all remote workers
            # Note: Remote workers primarily do rollout (inference), not training
            # So we use inference_only=True to avoid forward hook overhead
            workers = self.algo.workers.remote_workers()
            if workers:
                def _apply_to_worker(worker):
                    def inner(policy, pid):
                        try:
                            backbone = policy.model.backbone
                            if hasattr(backbone, '_orig_mod'):
                                backbone = backbone._orig_mod
                            from compression.mask_prune_compressor import apply_masks_to_backbone
                            # inference_only=True: Remote workers primarily do rollout (no optimizer updates)
                            # This avoids forward pre-hook overhead during inference!
                            apply_masks_to_backbone(backbone, masks, inference_only=True)
                        except Exception as exc:
                            print(f"[PolicyManager] ⚠️ Failed to apply training masks on worker: {exc}")
                        return 1
                    worker.foreach_policy(inner)
                    return 1
                
                ray.get([w.apply.remote(_apply_to_worker) for w in workers])
            
            # Verify masks were applied by checking actual sparsity
            actual_sparsity = self._check_training_model_sparsity()
            print(f"[PolicyManager] ✅ Masks applied to training model (Both Pruned mode)")
            print(f"[PolicyManager] 📊 Training model sparsity verified: {actual_sparsity*100:.1f}%")
            
        except Exception as exc:
            import traceback
            print(f"[PolicyManager] ⚠️ Failed to apply masks to training model: {exc}")
            traceback.print_exc()
    
    def _check_training_model_sparsity(self) -> float:
        """Check actual sparsity of training model weights (for diagnostics)"""
        try:
            local_worker = self.algo.workers.local_worker()
            local_policy = local_worker.get_policy()
            backbone = local_policy.model.backbone
            
            if hasattr(backbone, '_orig_mod'):
                backbone = backbone._orig_mod
            
            total_params = 0
            zero_params = 0
            for layer in backbone.hidden_layers:
                weight = layer.weight.data
                total_params += weight.numel()
                zero_params += (weight.abs() < 1e-8).sum().item()
            
            return zero_params / total_params if total_params > 0 else 0.0
        except Exception:
            return 0.0
    
    def _reapply_masks_after_weight_sync(self):
        """
        Reapply masks after weight synchronization.
        
        Why this is needed:
            - When we sync weights from training model to inference model
            - The sync operation (push_weight_update) overwrites ALL weights
            - This includes masked weights, which get restored to non-zero values
            - We must reapply masks to keep inference model pruned
        """
        if not hasattr(self, '_current_masks') or self._current_masks is None:
            return  # No masks to apply, skip
        
        try:
            from compression.mask_prune_compressor import apply_masks_to_backbone
            
            # Get inference model
            infer_model = self.current_infer_model
            if hasattr(infer_model, "_orig_mod"):
                infer_model = infer_model._orig_mod
            
            # Apply masks to inference model
            # Use inference_only=True to skip hooks (inference model doesn't train)
            apply_masks_to_backbone(infer_model, self._current_masks, inference_only=True)
            
        except Exception as exc:
            # Fail silently - don't crash training
            pass

    def _snapshot_train_backbone(self):
        bb = getattr(self.train_model, "backbone", None)
        if bb is None:
            return None
        return {
            k: v.detach().cpu().clone()
            for k, v in bb.state_dict().items()
        }

    def _check_structure_match(self, train_state_dict):
        """
        检查训练模型和推理模型的结构是否一致
        
        如果不一致（例如剪枝后），返回 False
        """
        if self.current_infer_model is None:
            return True
        
        try:
            # 获取推理模型的 state_dict
            infer_model = self.current_infer_model
            if hasattr(infer_model, "_orig_mod"):
                infer_model = infer_model._orig_mod
            
            infer_state_dict = infer_model.state_dict()
            
            # 检查关键层的形状
            for key in train_state_dict.keys():
                if key not in infer_state_dict:
                    return False
                
                train_shape = train_state_dict[key].shape
                infer_shape = infer_state_dict[key].shape
                
                if train_shape != infer_shape:
                    # 结构不匹配（可能是剪枝导致）
                    return False
            
            return True
            
        except Exception:
            # 出错时保守处理，认为不匹配
            return False

    def _load_state_into_infer(self, snapshot: Dict[str, Any]):
        if self.current_infer_model is None:
            return
        if torch is None:
            return
        target = getattr(self.current_infer_model, "_orig_mod", self.current_infer_model)
        try:
            device = next(target.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        converted = {}
        for k, v in snapshot.items():
            if torch.is_tensor(v):
                converted[k] = v.to(device)
            else:
                converted[k] = v
        try:
            target.load_state_dict(converted, strict=False)
        except Exception as exc:
            print(f"[PolicyManager] ⚠️ Failed to update local inference model: {exc}")

    # ------------------------------------------------------------------
    # 可选：编译本地训练 backbone 加速前向
    # ------------------------------------------------------------------
    def _compile_training_backbone_once(self):
        if self._training_backbone_compiled:
            return
        if not hasattr(self.train_model, "backbone"):
            return
        if torch is None:
            return

        backend = "inductor"
        primary = self.compressors[0]
        if hasattr(primary, "backend"):
            backend = getattr(primary, "backend") or backend

        if hasattr(self.train_model, "to"):
            self.train_model.to(self.device)

        self.train_model.backbone = torch.compile(self.train_model.backbone, backend=backend)
        self._training_backbone_compiled = True
        print(f"[PolicyManager] 🧠 Local training backbone compiled via torch.compile backend={backend}.")

    def _resolve_device(self, device: str):
        if torch is None:
            return "cpu"
        try:
            resolved = torch.device(device)
            if resolved.type.startswith("cuda") and not torch.cuda.is_available():
                print(f"[PolicyManager] ⚠️ Device {device} unavailable, fallback to CPU.")
                return torch.device("cpu")
            return resolved
        except (RuntimeError, TypeError):
            print(f"[PolicyManager] ⚠️ Invalid device {device}, fallback to CPU.")
            return torch.device("cpu")

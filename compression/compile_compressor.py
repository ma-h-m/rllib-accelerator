# path: compression/compile_compressor.py

import time
import numpy as np
from typing import Any, Dict, Tuple, Optional, List
import torch
from ray.rllib.utils.framework import try_import_torch
torch, nn = try_import_torch()

from compression.base import BaseCompressor
from models.policy import PolicyBackbone  # ⚠️你需要把这个改成你的真实路径


class CompileCompressor(BaseCompressor):
    """
    用于 torch.compile 的压缩器。

    主要功能：
    - 从 train_model.backbone 拍 snapshot（state_dict clone）
    - 基于权重 diff 判断是否需要重新 compile
    - 调用 torch.compile 生成 compiled_backbone
    """

    def __init__(self,
                 backend: str = "inductor",
                 diff_threshold: float = 1e-4,
                 device: str = "cpu",
                 recompile_every: int = 2,
                 sparsity_change_threshold: float = 0.05):
        """
        参数:
            backend: torch.compile backend（一般用 inductor）
            diff_threshold: 若新旧 snapshot 平均差异大于此阈值则重新编译
            recompile_every: 每 N 次压缩后强制重新编译（解决稀疏性变化问题）
            sparsity_change_threshold: 稀疏性变化超过此阈值时重新编译
        """
        self.backend = backend
        self.diff_threshold = diff_threshold
        self.device_str = device
        self.recompile_every = recompile_every
        self.sparsity_change_threshold = sparsity_change_threshold
        
        # ✅ 支持 weight sync
        # 对于 prune+compile pipeline，会先同步权重，然后重新应用 mask
        self.supports_weight_sync = True
        if torch is not None:
            try:
                resolved = torch.device(device)
                if resolved.type.startswith("cuda") and not torch.cuda.is_available():
                    print(f"[CompileCompressor] ⚠️ Device {device} unavailable, fallback to CPU.")
                    resolved = torch.device("cpu")
                self.device = resolved
            except (RuntimeError, TypeError):
                self.device = torch.device("cpu")
        else:
            self.device = None
        
        self._raw_model: Optional[PolicyBackbone] = None
        self._compiled_model: Optional[Any] = None
        self._meta: Optional[Dict[str, Any]] = None
        
        # 📊 Track compression count and sparsity for recompilation logic
        self._compression_count = 0
        self._last_sparsity = 0.0

    # ------------------------------------------------------------
    # 1. snapshot
    # ------------------------------------------------------------
    def snapshot(self, train_model: Any) -> Dict[str, torch.Tensor]:
        """复制 backbone 的 state_dict（无梯度，cpu clone）。"""
        bb = train_model.backbone
        if hasattr(bb, "_orig_mod"):
            bb_to_copy = bb._orig_mod
        else:
            bb_to_copy = bb
        state = {
            k: v.detach().cpu().clone()
            for k, v in bb_to_copy.state_dict().items()
        }
        hidden_dims = getattr(train_model, "hidden_dims", None)
        if hidden_dims is None:
            hidden_dims = [64, 64]
        self._meta = {
            "in_dim": getattr(train_model, "in_dim", None),
            "num_outputs": getattr(train_model, "num_outputs", None),
            "hidden_dims": list(hidden_dims),
            "use_residual": getattr(train_model, "use_residual", False),
        }
        return state

    # ------------------------------------------------------------
    # 2. diff 检测
    # ------------------------------------------------------------
    def should_recompress(self,
                          new_snapshot: Dict[str, torch.Tensor],
                          last_snapshot: Dict[str, torch.Tensor]) -> bool:
        """基于参数差分判断是否需要重新编译。"""

        if last_snapshot is None:
            return True  # 第一次必须压缩

        diffs = []
        for k in new_snapshot:
            diff_value = (new_snapshot[k] - last_snapshot[k]).abs().mean().item()
            diffs.append(diff_value)

        mean_diff = float(np.mean(diffs))

        return mean_diff > self.diff_threshold

    # ------------------------------------------------------------
    # 3. compress（torch.compile）
    # ------------------------------------------------------------
    def compress(self, snapshot) -> Tuple[Any, Dict[str, Any]]:
        """
        执行 torch.compile，返回新的 compiled_backbone。
        
        支持两种输入：
        1. Dict[str, torch.Tensor] - state_dict (来自 snapshot)
        2. PolicyBackbone - 模型对象 (来自上游 compressor，如剪枝)
        """
        
        # 检测输入类型
        if isinstance(snapshot, dict):
            # 情况 1: state_dict
            if self._meta is None:
                raise RuntimeError("CompileCompressor snapshot meta is missing.")
            in_dim = self._meta["in_dim"]
            num_outputs = self._meta["num_outputs"]
            hidden_dims: List[int] = self._meta["hidden_dims"]
            use_residual: bool = self._meta.get("use_residual", False)
            state_dict = snapshot
        elif isinstance(snapshot, (PolicyBackbone, torch.nn.Module)):
            # 情况 2: 来自上游的模型对象（例如 MaskPruneCompressor 的输出）
            bb_input = snapshot
            
            # 解包 compile wrapper
            if hasattr(bb_input, "_orig_mod"):
                bb_input = bb_input._orig_mod
            
            # 如果是 MaskedPolicyBackbone，提取内部的 backbone
            if hasattr(bb_input, "backbone"):
                actual_bb = bb_input.backbone
                if hasattr(actual_bb, "_orig_mod"):
                    actual_bb = actual_bb._orig_mod
            else:
                actual_bb = bb_input
            
            if self.device is not None and hasattr(actual_bb, 'to'):
                actual_bb = actual_bb.to(self.device)
            
            # 从模型推断结构（用于 meta）
            in_dim = actual_bb.hidden_layers[0].in_features if len(actual_bb.hidden_layers) > 0 else 4
            num_outputs = actual_bb.policy_head.out_features
            hidden_dims = [layer.out_features for layer in actual_bb.hidden_layers]
            use_residual = actual_bb.use_residual
            
            # 更新 meta
            self._meta = {
                "in_dim": in_dim,
                "num_outputs": num_outputs,
                "hidden_dims": hidden_dims,
                "use_residual": use_residual,
            }
            
            # 📊 Calculate current sparsity (for recompilation decision)
            current_sparsity = self._calculate_sparsity(actual_bb)
            sparsity_delta = abs(current_sparsity - self._last_sparsity)
            
            # 🔧 Periodic recompilation logic
            self._compression_count += 1
            force_recompile = False
            recompile_reason = None
            
            # Reason 1: Periodic recompilation (every N compressions)
            if self.recompile_every > 0 and self._compression_count % self.recompile_every == 0:
                force_recompile = True
                recompile_reason = f"periodic (every {self.recompile_every} compressions)"
            
            # Reason 2: Significant sparsity change (e.g., pruning increased zeros)
            if sparsity_delta > self.sparsity_change_threshold:
                force_recompile = True
                recompile_reason = f"sparsity change ({self._last_sparsity*100:.1f}% → {current_sparsity*100:.1f}%)"
            
            if force_recompile:
                print(f"[CompileCompressor] 🔄 Forcing recompilation: {recompile_reason}")
                self._compiled_model = None
                self._raw_model = None
                self._last_sparsity = current_sparsity
            
            # ✅ Try to reuse compiled model if structure matches and no force recompile
            reused = False
            if self._compiled_model is not None and self._raw_model is not None:
                # Check if we can reuse by comparing structure
                try:
                    # Extract state dicts to compare structure
                    old_state = self._raw_model.state_dict()
                    new_state = actual_bb.state_dict()
                    
                    # Check if all keys and shapes match (structure identical)
                    structure_match = True
                    if set(old_state.keys()) != set(new_state.keys()):
                        structure_match = False
                    else:
                        for key in old_state.keys():
                            if old_state[key].shape != new_state[key].shape:
                                structure_match = False
                                break
                    
                    if structure_match:
                        # Structure matches! Just update weights in existing model
                        # This preserves hooks from pruning!
                        t0 = time.time()
                        self._raw_model.load_state_dict(new_state, strict=True)
                        latency = time.time() - t0
                        
                        # Reuse existing compiled version
                        compiled_bb = self._compiled_model
                        actual_bb = self._raw_model  # Use existing model
                        reused = True
                        
                        print(f"[CompileCompressor] ♻️  Reused compiled model (structure unchanged, {latency:.4f}s to update weights)")
                    else:
                        # Structure changed, need fresh compile
                        reused = False
                        self._compiled_model = None
                        self._raw_model = None
                        
                except Exception as exc:
                    # Any error, fallback to fresh compile
                    print(f"[CompileCompressor] ⚠️ Failed to check structure match: {exc}")
                    reused = False
                    self._compiled_model = None
                    self._raw_model = None
            
            if not reused:
                # Fresh compilation needed
                t0 = time.time()
                compiled_bb = torch.compile(actual_bb, backend=self.backend)
                latency = time.time() - t0
                
                # Save references
                self._raw_model = actual_bb
                self._compiled_model = compiled_bb
                print(f"[CompileCompressor] 🔧 Compiled new model ({latency:.4f}s)")
            
            state_dict = None  # Not needed
            
        else:
            raise TypeError(f"CompileCompressor.compress() expects Dict or nn.Module, got {type(snapshot)}")

        # 只有在 snapshot 是 dict 时才尝试复用
        if isinstance(snapshot, dict):
            reused = False
            if self._compiled_model is not None and self._raw_model is not None:
                # 检查结构是否一致
                try:
                    load_start = time.time()
                    self._raw_model.load_state_dict(state_dict)
                    latency = time.time() - load_start
                    compiled_bb = self._compiled_model
                    reused = True
                except RuntimeError:
                    # 结构不匹配，需要重新编译
                    reused = False
                    self._compiled_model = None
                    self._raw_model = None
            
            if not reused:
                bb = PolicyBackbone(in_dim, num_outputs, hidden_dims, use_residual)
                if self.device is not None:
                    bb = bb.to(self.device)
                bb.load_state_dict(state_dict)

                t0 = time.time()
                compiled_bb = torch.compile(bb, backend=self.backend)
                latency = time.time() - t0

                self._raw_model = bb
                self._compiled_model = compiled_bb

        return compiled_bb, {
            "type": "torch.compile",
            "backend": self.backend,
            "latency": latency,
            "in_dim": in_dim,
            "num_outputs": num_outputs,
            "hidden_dims": hidden_dims,
            "use_residual": use_residual,
            "reused": reused,
        }
    
    def _calculate_sparsity(self, backbone: PolicyBackbone) -> float:
        """
        Calculate sparsity (percentage of near-zero weights) in backbone.
        
        Used to detect when pruning has increased sparsity, triggering recompilation
        to let torch.compile optimize for the new sparse pattern.
        
        Returns:
            float: Sparsity ratio (0.0 = dense, 1.0 = all zeros)
        """
        try:
            total_params = 0
            zero_params = 0
            for layer in backbone.hidden_layers:
                weight = layer.weight.data
                total_params += weight.numel()
                zero_params += (weight.abs() < 1e-8).sum().item()
            
            return zero_params / total_params if total_params > 0 else 0.0
        except Exception:
            return 0.0

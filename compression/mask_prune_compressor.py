# path: compression/mask_prune_compressor.py

import time
from typing import Any, Dict, Tuple, Optional, List
import numpy as np
import torch
from torch import nn

from compression.base import BaseCompressor
from models.policy import PolicyBackbone


def apply_masks_to_backbone(backbone: PolicyBackbone, masks: Dict[str, torch.Tensor], inference_only: bool = False):
    """
    Apply masks to backbone weights
    
    Args:
        backbone: Model to apply masks to
        masks: Dictionary of mask tensors
        inference_only: If True, skip gradient hooks (for inference workers that never train)
                       This avoids unnecessary overhead on inference-only models
    
    Key: Use register_hook to ensure masks persist during training!
    After each backward pass, gradients are masked so the optimizer won't update masked weights
    
    Performance Note:
        - inference_only=True: ~0% overhead (just zeros out weights)
        - inference_only=False: ~2-3% overhead (adds gradient hooks)
        - Only use inference_only=False for models that actually do backward passes!
    """
    for i, layer in enumerate(backbone.hidden_layers):
        weight_mask_name = f"hidden_layers_{i}_weight_mask"
        if weight_mask_name in masks:
            mask = masks[weight_mask_name].to(layer.weight.device)
            
            # 1. Zero out masked weights
            with torch.no_grad():
                layer.weight.data *= mask
            
            # 2. Register backward hook ONLY if this model will be trained
            # Skip hooks for inference-only workers to avoid overhead!
            if not inference_only:
                # Key: Register backward hook to ensure gradients are also masked
                # This prevents the optimizer from updating masked weights!
                def make_mask_hook(mask):
                    def hook(grad):
                        # Set gradients of masked weights to 0
                        return grad * mask
                    return hook
                
                # Remove previous hook (if exists)
                if hasattr(layer.weight, '_mask_hook_handle'):
                    layer.weight._mask_hook_handle.remove()
                
                # Register new hook
                handle = layer.weight.register_hook(make_mask_hook(mask))
                layer.weight._mask_hook_handle = handle
    
    return backbone


class MaskPruneCompressor(BaseCompressor):
    """
    Mask-Based (Unstructured) Pruning Compressor
    
    Uses masks to "zero out" weights without changing model structure
    """
    
    def __init__(
        self,
        prune_ratio: float = 0.2,
        diff_threshold: float = 1e-3,
        technique: str = "magnitude",  # "magnitude", "random", "gradient"
        schedule: str = "iterative",   # "iterative", "oneshot"
        prune_steps: int = 10,         # Total steps for iterative pruning
    ):
        """
        Args:
            prune_ratio: Target pruning ratio (0.2 = remove 20% of weights)
            diff_threshold: Weight change threshold, re-prune only if exceeded
            technique: Pruning technique - "magnitude" (weight size), "random", "gradient"
            schedule: Pruning schedule - "iterative" (gradual), "oneshot" (one-time)
            prune_steps: Total steps for iterative pruning (default 10 steps)
        """
        self.prune_ratio = prune_ratio
        self.diff_threshold = diff_threshold
        self.technique = technique
        self.schedule = schedule
        self.prune_steps = max(1, prune_steps) 
        self.supports_weight_sync = True
        
        self._meta: Optional[Dict[str, Any]] = None
        self._current_sparsity = 0.0 
        self._pruning_step = 0
        self._masks: Optional[Dict[str, torch.Tensor]] = None
    
    def snapshot(self, train_model: Any) -> Dict[str, torch.Tensor]:
        """Extract backbone state_dict from training model"""
        bb = train_model.backbone
        if hasattr(bb, "_orig_mod"):
            bb_to_copy = bb._orig_mod
        else:
            bb_to_copy = bb
        
        state = {k: v.detach().cpu().clone() for k, v in bb_to_copy.state_dict().items()}
        
        # Diagnostic: Check sparsity of training model at snapshot time
        total_params = 0
        zero_params = 0
        for layer in bb_to_copy.hidden_layers:
            weight = layer.weight.data
            total_params += weight.numel()
            zero_params += (weight.abs() < 1e-8).sum().item()
        snapshot_sparsity = zero_params / total_params if total_params > 0 else 0.0
        
        if snapshot_sparsity > 0.01:  # More than 1% sparse
            print(f"[MaskPrune] 📸 Snapshot taken - Training model sparsity: {snapshot_sparsity*100:.1f}%")
        else:
            print(f"[MaskPrune] 📸 Snapshot taken - Training model is FULL (not pruned)")
        
        hidden_dims = getattr(train_model, "hidden_dims", None) or [64, 64]
        self._meta = {
            "in_dim": getattr(train_model, "in_dim", None),
            "num_outputs": getattr(train_model, "num_outputs", None),
            "hidden_dims": list(hidden_dims),
            "use_residual": getattr(train_model, "use_residual", False),
        }
        return state
    
    def should_recompress(
        self,
        new_snapshot: Dict[str, torch.Tensor],
        last_snapshot: Optional[Dict[str, torch.Tensor]]
    ) -> bool:
        """Determine if re-pruning is needed based on weight changes"""
        if last_snapshot is None:
            return True
        
        diffs = []
        for k in new_snapshot:
            if k in last_snapshot:
                diff_value = (new_snapshot[k] - last_snapshot[k]).abs().mean().item()
                diffs.append(diff_value)
        
        mean_diff = float(np.mean(diffs)) if diffs else 0.0
        return mean_diff > self.diff_threshold
    
    def compress(self, snapshot: Dict[str, torch.Tensor]) -> Tuple[Any, Dict[str, Any]]:
        """Execute Mask-Based Pruning"""
        if self._meta is None:
            raise RuntimeError("MaskPruneCompressor snapshot meta is missing.")
        
        t0 = time.time()
        
        in_dim = self._meta["in_dim"]
        num_outputs = self._meta["num_outputs"]
        hidden_dims = self._meta["hidden_dims"]
        use_residual = self._meta.get("use_residual", False)
        
        # 1. Create backbone with original size
        backbone = PolicyBackbone(in_dim, num_outputs, hidden_dims, use_residual)
        backbone.load_state_dict(snapshot)
        
        # 2. Calculate target sparsity (iterative vs oneshot)
        if self.schedule == "iterative":
            # Gradually increase sparsity (complete in prune_steps steps)
            self._pruning_step += 1
            step_size = self.prune_ratio / self.prune_steps
            target_sparsity = min(
                self._pruning_step * step_size,
                self.prune_ratio
            )
        else:
            # Prune to target in one shot
            target_sparsity = self.prune_ratio
            self._pruning_step = self.prune_steps  # Mark as complete
        
        # 3. Compute masks
        masks = self._compute_masks(backbone, target_sparsity)
        
        # 4. Apply masks directly to backbone (modify weights)
        # Note: This model will be used for inference, so we skip hooks (inference_only=True)
        # The hooks will be added on worker side if needed (for training workers)
        apply_masks_to_backbone(backbone, masks, inference_only=True)
        
        # 5. Calculate actual sparsity
        self._current_sparsity = self._calculate_actual_sparsity(backbone, masks)
        self._masks = masks
        
        latency = time.time() - t0
        
        info = {
            "type": "mask_prune",
            "technique": self.technique,
            "schedule": self.schedule,
            "target_sparsity": target_sparsity,
            "actual_sparsity": self._current_sparsity,
            "pruning_step": self._pruning_step,
            "latency": latency,
            "masks": masks,  # Return masks for PolicyManager to use
        }
        
        print(f"[MaskPrune]  Step {self._pruning_step}: "
              f"Sparsity {self._current_sparsity*100:.1f}% "
              f"(target {target_sparsity*100:.1f}%) | "
              f"Time: {latency:.3f}s")
        
        return backbone, info
    
    def _calculate_actual_sparsity(
        self,
        backbone: PolicyBackbone,
        masks: Dict[str, torch.Tensor]
    ) -> float:
        """Calculate actual sparsity"""
        total_params = 0
        zero_params = 0
        
        for i, layer in enumerate(backbone.hidden_layers):
            weight_mask_name = f"hidden_layers_{i}_weight_mask"
            if weight_mask_name in masks:
                mask = masks[weight_mask_name]
                total_params += mask.numel()
                zero_params += (mask == 0).sum().item()
        
        return zero_params / total_params if total_params > 0 else 0.0
    
    def _compute_masks(
        self,
        backbone: PolicyBackbone,
        target_sparsity: float
    ) -> Dict[str, torch.Tensor]:
        """
        Compute pruning masks
        
        Returns: {layer_name: mask_tensor}
        """
        masks = {}
        
        for i, layer in enumerate(backbone.hidden_layers):
            weight = layer.weight.data.cpu()
            
            if self.technique == "magnitude":
                # Magnitude-based: Keep weights with larger magnitude
                importance = weight.abs()
            elif self.technique == "random":
                # Random: Random pruning
                importance = torch.rand_like(weight)
            elif self.technique == "gradient":
                # Gradient-based: Needs gradient info (not implemented, fallback to magnitude)
                importance = weight.abs()
            else:
                raise ValueError(f"Unknown technique: {self.technique}")
            
            # Calculate threshold
            flat_importance = importance.flatten()
            threshold = torch.quantile(flat_importance, target_sparsity)
            
            # Create mask (1 = keep, 0 = prune)
            mask = (importance > threshold).float()
            
            # Ensure each neuron has at least some connections (optional)
            # For simplicity, we directly use global threshold here
            
            masks[f"hidden_layers_{i}_weight_mask"] = mask
        
        return masks




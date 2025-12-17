"""
Pruning Experiment Configuration File

Experiment Design:
1. Baseline (no optimization)
2. Async Compile (torch.compile acceleration)
3. Async Prune (Mask-based pruning, Teacher-Student architecture)
4. Async Prune+Compile (Pruning + Compile dual acceleration, Teacher-Student architecture)
"""

from framework.policy_manager import CompileMode


# ============================================================
# Base Hyperparameters (Unified config, control all parameters here)
# ============================================================
DEFAULT_HPARAMS = {
    # ========== Environment & Training ==========
    "env_id": "CartPole-v1",
    "num_epochs": 150,              
    "train_batch_size": 4000,
    "lr": 1e-4,                     
    "seed": 42,
    "num_rollout_workers": 4,
    "rollout_fragment_length": 500,
    
    # ========== Model Architecture ==========
    "hidden_dim": 256,              
    "hidden_depth": 4,               
    "use_residual": True,
    
    # ========== Compression Timing ==========
    "trigger_every": 10,             
    "min_epoch_before_compress": 20, 
    # Note: With LINEAR iterative pruning (5 steps), full sparsity reached at:
    # trigger_epochs = [20, 30, 40, 50, 60] → 15% by epoch 60
    # Each step increases sparsity by: prune_ratio / prune_steps 
    
    # ========== Compile Settings ==========
    "compile_backend": "inductor",
    "compile_diff_threshold": 1e-4,
    "compile_recompile_every": 2,          
    "compile_sparsity_change_threshold": 0.05,  
    
    # ========== Pruning Settings ==========
    "prune_ratio": 0.15,             
    "prune_diff_threshold": 1e-3,    
    "prune_technique": "magnitude", 
    "prune_schedule": "iterative",   
    "prune_steps": 5,                # Reduced from 10 to 5 for faster convergence                
    
    # ========== Pruning Mode ==========
    "prune_training_model": True,   # Both training and inference models are pruned
    
    # ========== Quantization Settings ==========
    "quant_diff_threshold": 5e-4,
    
    # ========== Device ==========
    "device": "cpu",
    
    # ========== Learning Rate Decay ==========
    "lr_decay": {
        "enabled": False,           
        "gamma": 0.5,
        "step_epochs": 100,
        "min_lr": 1e-6,
    },
    
    # ========== Logging ==========
    "use_wandb": True,  # Enable W&B logging for experiments
    "wandb_project": "rllib-accelerator",
    "wandb_group": "pruning_experiments",
    "wandb_tags": ["pruning", "compression", "rllib"],  # Tags for experiment organization
}


# ============================================================
# Experiment 1: Basic Comparison (Baseline vs Compile vs Prune)
# ============================================================
EXPERIMENTS_BASIC = [
    {
        "name": "baseline",
        "mode": CompileMode.NONE,
        "compile_training_backbone": False,
        "trigger_every": 0,  
        "enable_diff_check": False,
        "compressors": ["compile"],
        "async_warmup": False,
        "infer_output_index": -1,
    },
    {
        "name": "async_compile",
        "mode": CompileMode.ASYNC,
        "compile_training_backbone": False,
        "trigger_every": 15,  # ✅ 每15个epoch触发一次编译
        "enable_diff_check": False,
        "compressors": ["compile"],
        "async_warmup": True,
        "infer_output_index": -1,
    },
    {
        "name": "async_prune",
        "mode": CompileMode.ASYNC,
        "compile_training_backbone": False,  
        "trigger_every": 15,  # ✅ 每15个epoch触发一次剪枝
        "enable_diff_check": False,     
        "compressors": ["prune"],
        "async_warmup": True,
        "infer_output_index": -1,
    },
    {
        "name": "async_prune_compile",
        "mode": CompileMode.ASYNC,
        "compile_training_backbone": False,  
        "trigger_every": 15,  
        "enable_diff_check": False,     
        "compressors": ["prune+compile"],
        "async_warmup": True,
        "infer_output_index": -1,  
    },
]


# ============================================================
# Experiment 2: Different Pruning Ratios Comparison
# ============================================================
EXPERIMENTS_PRUNE_RATIOS = [
    {
        "name": "baseline",
        "mode": CompileMode.NONE,
        "compile_training_backbone": False,
        "trigger_every": 0,
        "enable_diff_check": False,
        "compressors": ["compile"],
        "async_warmup": False,
        "infer_output_index": -1,
    },
]

for ratio, steps in [(0.1, 5), (0.2, 5), (0.3, 3), (0.5, 2)]:
    # Final sparsity = ratio (with LINEAR iterative pruning)
    # Steps = how many compressions to reach it
    # Formula: sparsity at step N = ratio * (N / steps)
    
    EXPERIMENTS_PRUNE_RATIOS.append({
        "name": f"async_prune_compile_ratio={ratio:.1f}",
        "mode": CompileMode.ASYNC,
        "compile_training_backbone": False,
        "trigger_every": 15,
        "enable_diff_check": False,
        "compressors": ["prune+compile"],
        "async_warmup": True,
        "infer_output_index": -1,  
        "_prune_ratio_override": ratio,      # Final target sparsity
        "_prune_steps_override": steps,      # Number of steps to reach it
    })
    print(f"[Config] Target Sparsity={ratio*100:.0f}%, Steps={steps}, "
          f"Increment per step={(ratio/steps)*100:.1f}%")


# ============================================================
# Experiment 3: Different Pruning Strategies Comparison
# ============================================================
EXPERIMENTS_STRATEGIES = [
    {
        "name": "baseline",
        "mode": CompileMode.NONE,
        "compile_training_backbone": False,
        "trigger_every": 0,
        "enable_diff_check": False,
        "compressors": ["compile"],
        "async_warmup": False,
        "infer_output_index": -1,
    },
]

for technique in ["magnitude", "random"]:
    EXPERIMENTS_STRATEGIES.append({
        "name": f"async_prune_compile_{technique}",
        "mode": CompileMode.ASYNC,
        "compile_training_backbone": False,
        "trigger_every": 15,
        "enable_diff_check": False,
        "compressors": ["prune+compile"],
        "async_warmup": True,
        "infer_output_index": -1,
        "_prune_technique_override": technique,
    })


# ============================================================
# Experiment 4: Different Trigger Frequencies Comparison
# ============================================================
EXPERIMENTS_TRIGGER_FREQ = [
    {
        "name": "baseline",
        "mode": CompileMode.NONE,
        "compile_training_backbone": False,
        "trigger_every": 0,
        "enable_diff_check": False,
        "compressors": ["compile"],
        "async_warmup": False,
        "infer_output_index": -1,
    },
]

for freq in [5, 10, 15, 20]:
    EXPERIMENTS_TRIGGER_FREQ.append({
        "name": f"async_prune_compile_freq={freq}",
        "mode": CompileMode.ASYNC,
        "compile_training_backbone": False,
        "trigger_every": freq,
        "enable_diff_check": False,
        "compressors": ["prune+compile"],
        "async_warmup": True,
        "infer_output_index": -1,
    })




# ============================================================
# Default Experiment Set (for quick testing)
# ============================================================
EXPERIMENTS = EXPERIMENTS_BASIC

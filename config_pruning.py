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
    "num_epochs": 100,              
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
    "trigger_every": 15,             
    "min_epoch_before_compress": 30, 
    
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
    "prune_steps": 15,                
    
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
        "trigger_every": 15,  # ✅ 每15个epoch触发一次剪枝+编译
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

for ratio in [0.1, 0.2, 0.3, 0.4]:
    EXPERIMENTS_PRUNE_RATIOS.append({
        "name": f"async_prune_compile_ratio={ratio:.1f}",
        "mode": CompileMode.ASYNC,
        "compile_training_backbone": False,
        "trigger_every": 15,
        "enable_diff_check": False,
        "compressors": ["prune+compile"],
        "async_warmup": True,
        "infer_output_index": -1,  
        "_prune_ratio_override": ratio,
    })


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
# Experiment 5: Different Model Sizes Comparison
# ============================================================
EXPERIMENTS_MODEL_SIZES = [
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

for dim in [256, 512, 1024, 2048]:
    EXPERIMENTS_MODEL_SIZES.append({
        "name": f"async_prune_compile_dim={dim}",
        "mode": CompileMode.ASYNC,
        "compile_training_backbone": False,
        "trigger_every": 15,
        "enable_diff_check": False,
        "compressors": ["prune+compile"],
        "async_warmup": True,
        "infer_output_index": -1,
        "_hidden_dim_override": dim,
    })


# ============================================================
# Experiment 6: Compare Teacher-Student vs Both Pruned
# ============================================================
EXPERIMENTS_PRUNING_MODES = [
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
        "name": "prune_teacher_student",
        "mode": CompileMode.ASYNC,
        "compile_training_backbone": False,
        "trigger_every": 15,
        "enable_diff_check": False,
        "compressors": ["prune+compile"],
        "async_warmup": True,
        "infer_output_index": -1,
        "_prune_training_model_override": False,  # Teacher-Student mode
    },
    {
        "name": "prune_both_pruned",
        "mode": CompileMode.ASYNC,
        "compile_training_backbone": False,
        "trigger_every": 15,
        "enable_diff_check": False,
        "compressors": ["prune+compile"],
        "async_warmup": True,
        "infer_output_index": -1,
        "_prune_training_model_override": True,  # Both Pruned mode
    },
]


# ============================================================
# Default Experiment Set (for quick testing)
# ============================================================
EXPERIMENTS = EXPERIMENTS_BASIC

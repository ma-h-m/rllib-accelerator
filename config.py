"""
Centralized hyper-parameter definitions for RLlib accelerator experiments.
"""

from framework.policy_manager import CompileMode


DEFAULT_HPARAMS = {
    # Environment & training loop
    "env_id": "CartPole-v1",
    "num_epochs": 10,
    "train_batch_size": 800000,
    "lr": 1e-4,
    "num_rollout_workers": 4,
    "rollout_fragment_length": 2000,
    # Model architecture
    "hidden_dim": 1024,
    "hidden_depth": 8,
    "use_residual": True,
    # Compile settings
    "compile_backend": "inductor",
    "compile_diff_threshold": 1e-4,
    # Device selection（默认 CPU，可改为 "cuda:0"）
    "device": "cuda:0",
    # Logging
    "use_wandb": False,
    "wandb_project": "rllib-accelerator",
}


EXPERIMENTS = [
    {
        "name": "baseline",
        "mode": CompileMode.NONE,
        "compile_training_backbone": False,
        "trigger_every": 0,
        "enable_diff_check": False,
    },
    {
        "name": "torch_compile_sync",
        "mode": CompileMode.SYNC,
        "compile_training_backbone": True,
        "trigger_every": 0,     # 仅首次（last_snapshot is None）触发 compile
        "enable_diff_check": False,
        "async_warmup": False,
    },
    {
        "name": "torch_compile_async",
        "mode": CompileMode.ASYNC,
        "compile_training_backbone": True,
        "trigger_every": 0,
        "enable_diff_check": False,
        "async_warmup": False,
    },
        {
        "name": "torch_compile_async",
        "mode": CompileMode.ASYNC,
        "compile_training_backbone": True,
        "trigger_every": 0,
        "enable_diff_check": False,
        "async_warmup": True,
    },
]

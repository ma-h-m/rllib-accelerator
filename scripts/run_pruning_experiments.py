"""
运行剪枝实验脚本

用法：
    python scripts/run_pruning_experiments.py --experiment basic

    python scripts/run_pruning_experiments.py --experiment ratios
    
    python scripts/run_pruning_experiments.py --experiment strategies
    
    python scripts/run_pruning_experiments.py --experiment freq
    
    python scripts/run_pruning_experiments.py --experiment basic --epochs 300 --hidden-dim 512
"""

# ✅ 修复 OpenMP 冲突（必须在所有导入之前设置，使用 os.environ 而不是 import os）
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import argparse
import random
import copy

# 添加项目根目录到 path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import ray
import torch
from datetime import datetime
from ray.rllib.algorithms.ppo import PPOConfig

from framework.trainer import Trainer
from compression.compile_compressor import CompileCompressor
from compression.mask_prune_compressor import MaskPruneCompressor
from framework.policy_manager import CompileMode

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 注册全局模型
from models.policy import CustomPolicyNet  # noqa


def get_experiments(experiment_type: str):
    """根据实验类型返回对应的实验配置"""
    if experiment_type == "basic":
        from config_pruning import EXPERIMENTS_BASIC as EXPERIMENTS
    elif experiment_type == "ratios":
        from config_pruning import EXPERIMENTS_PRUNE_RATIOS as EXPERIMENTS
    elif experiment_type == "strategies":
        from config_pruning import EXPERIMENTS_STRATEGIES as EXPERIMENTS
    elif experiment_type == "freq":
        from config_pruning import EXPERIMENTS_TRIGGER_FREQ as EXPERIMENTS
    elif experiment_type == "sizes":
        from config_pruning import EXPERIMENTS_MODEL_SIZES as EXPERIMENTS
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    return EXPERIMENTS


def resolve_device(config_device: str):
    requested = os.environ.get("ACCEL_DEVICE", config_device)
    normalized = requested.lower()
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        print(f"[main] ⚠️ Requested device '{requested}' unavailable, fallback to CPU.")
        return "cpu"
    return normalized if normalized.startswith("cuda") or normalized == "cpu" else requested


def apply_global_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_lr_schedule(hparams):
    decay_conf = hparams.get("lr_decay") or {}
    if not decay_conf.get("enabled"):
        return None
    base_lr = hparams["lr"]
    gamma = float(decay_conf.get("gamma", 0.5))
    step_epochs = max(1, int(decay_conf.get("step_epochs", 1)))
    min_lr = float(decay_conf.get("min_lr", 0.0))
    total_epochs = hparams["num_epochs"]
    steps_per_epoch = max(1, hparams["train_batch_size"])

    schedule = [[0, base_lr]]
    current_lr = base_lr
    epoch = step_epochs
    while epoch <= total_epochs:
        current_lr = max(min_lr, current_lr * gamma)
        schedule.append([epoch * steps_per_epoch, current_lr])
        epoch += step_epochs

    return schedule if len(schedule) > 1 else None


def build_config(hidden_layers, device: str, hparams):
    use_gpu = device.startswith("cuda") and torch.cuda.is_available()
    lr_schedule = build_lr_schedule(hparams)
    training_kwargs = {
        "model": {
            "custom_model": "custom_policy",
            "fcnet_hiddens": hidden_layers,
            "custom_model_config": {
                "use_residual": hparams["use_residual"],
                "device": device,
            },
        },
        "train_batch_size": hparams["train_batch_size"],
        "lr": hparams["lr"],
    }
    if lr_schedule is not None:
        training_kwargs["lr_schedule"] = lr_schedule
    config = (
        PPOConfig()
        .environment(hparams["env_id"])
        .framework("torch")
        .resources(num_gpus=1 if use_gpu else 0)
        .training(**training_kwargs)
    )
    
    # 禁用新的 API stack（兼容旧的 custom_model）
    try:
        config = config.api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
    except AttributeError:
        pass  # 旧版本没有这个方法
    
    # 兼容不同版本的 Ray API
    try:
        # Ray >= 2.10 使用 env_runners
        config = config.env_runners(
            num_env_runners=hparams["num_rollout_workers"],
            rollout_fragment_length=hparams["rollout_fragment_length"],
        )
    except AttributeError:
        # Ray < 2.10 使用 rollouts
        config = config.rollouts(
            num_rollout_workers=hparams["num_rollout_workers"],
            rollout_fragment_length=hparams["rollout_fragment_length"],
        )
    seed = hparams.get("seed")
    if seed is not None:
        config.seed = seed
    return config


def build_compressors(exp_conf, device, hparams):
    names = exp_conf.get("compressors", ["compile"])
    comps = []
    for name in names:
        if name == "compile":
            comps.append(
                CompileCompressor(
                    backend=hparams["compile_backend"],
                    diff_threshold=hparams["compile_diff_threshold"],
                    device=device,
                    recompile_every=hparams.get("compile_recompile_every", 2),
                    sparsity_change_threshold=hparams.get("compile_sparsity_change_threshold", 0.05),
                )
            )
        elif name == "prune":
            # Mask-Based (Unstructured) Pruning
            from compression.mask_prune_compressor import MaskPruneCompressor
            comps.append(
                MaskPruneCompressor(
                    prune_ratio=hparams.get("prune_ratio", 0.25),
                    diff_threshold=hparams.get("prune_diff_threshold", 1e-3),
                    technique=hparams.get("prune_technique", "magnitude"),
                    schedule=hparams.get("prune_schedule", "iterative"),
                )
            )
        elif name == "prune+compile":
            # Mask-Based Pruning + Compile
            from compression.mask_prune_compressor import MaskPruneCompressor
            comps.append(
                MaskPruneCompressor(
                    prune_ratio=hparams.get("prune_ratio", 0.25),
                    diff_threshold=hparams.get("prune_diff_threshold", 1e-3),
                    technique=hparams.get("prune_technique", "magnitude"),
                    schedule=hparams.get("prune_schedule", "iterative"),
                )
            )
            comps.append(
                CompileCompressor(
                    backend=hparams["compile_backend"],
                    diff_threshold=hparams["compile_diff_threshold"],
                    device=device,
                    recompile_every=hparams.get("compile_recompile_every", 2),
                    sparsity_change_threshold=hparams.get("compile_sparsity_change_threshold", 0.05),
                )
            )
        else:
            raise ValueError(f"Unknown compressor name: {name}")
    return comps


def main():
    parser = argparse.ArgumentParser(description="Run pruning experiments")
    parser.add_argument(
        "--experiment",
        type=str,
        default="basic",
        choices=["basic", "ratios", "strategies", "freq", "sizes"],
        help="Experiment type to run"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs (default: from config_pruning.py)")
    parser.add_argument("--hidden-dim", type=int, default=None, help="Hidden layer dimension (default: from config_pruning.py)")
    parser.add_argument("--hidden-depth", type=int, default=None, help="Number of hidden layers (default: from config_pruning.py)")
    parser.add_argument("--prune-ratio", type=float, default=None, help="Default pruning ratio (default: from config_pruning.py)")
    parser.add_argument("--seed", type=int, default=43, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda:0)")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging (overrides config)")
    parser.add_argument("--wandb-project", type=str, default="rllib-accelerator-pruning", help="W&B project name")
    
    args = parser.parse_args()
    
    # 初始化 Ray
    ray.init(include_dashboard=False)
    
    from config_pruning import DEFAULT_HPARAMS
    hparams = copy.deepcopy(DEFAULT_HPARAMS)
    
    if args.epochs is not None:
        hparams["num_epochs"] = args.epochs
    if args.hidden_dim is not None:
        hparams["hidden_dim"] = args.hidden_dim
    if args.hidden_depth is not None:
        hparams["hidden_depth"] = args.hidden_depth
    if args.prune_ratio is not None:
        hparams["prune_ratio"] = args.prune_ratio
    
    hparams["seed"] = args.seed
    hparams["device"] = args.device
    
    if args.wandb:
        hparams["use_wandb"] = True
    
    hparams["wandb_project"] = args.wandb_project
    hparams["wandb_group"] = f"pruning_{args.experiment}_layer={hparams['hidden_depth']}_dim={hparams['hidden_dim']}"
    
    if "wandb_tags" not in hparams:
        hparams["wandb_tags"] = []
    hparams["wandb_tags"].append(f"exp:{args.experiment}")
    
    device = resolve_device(hparams["device"])
    apply_global_seed(hparams.get("seed"))
    
    if hparams.get("seed") is not None:
        print(f"[main] Using seed: {hparams['seed']}")
    print(f"[main] Using device: {device}")
    print(f"[main] Running experiment type: {args.experiment}")
    
    hidden_layers = [hparams["hidden_dim"]] * hparams["hidden_depth"]
    
    experiments = get_experiments(args.experiment)
    
    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"Running: {exp['name']} ({exp['mode'].value})")
        print(f"{'='*60}")
        
        exp_hparams = copy.deepcopy(hparams)
        if "_prune_ratio_override" in exp:
            exp_hparams["prune_ratio"] = exp["_prune_ratio_override"]
            print(f"  → Prune ratio: {exp_hparams['prune_ratio']}")
        if "_prune_strategy_override" in exp:
            exp_hparams["prune_strategy"] = exp["_prune_strategy_override"]
            print(f"  → Prune strategy: {exp_hparams['prune_strategy']}")
        if "_prune_training_model_override" in exp:
            exp_hparams["prune_training_model"] = exp["_prune_training_model_override"]
            mode_name = "Both Pruned" if exp_hparams["prune_training_model"] else "Teacher-Student"
            print(f"  → Pruning mode: {mode_name}")
        if "_hidden_dim_override" in exp:
            exp_hparams["hidden_dim"] = exp["_hidden_dim_override"]
            hidden_layers = [exp_hparams["hidden_dim"]] * exp_hparams["hidden_depth"]
            print(f"  → Hidden dim: {exp_hparams['hidden_dim']}")
        
        config = build_config(hidden_layers, device, exp_hparams)
        compressors = build_compressors(exp, device, exp_hparams)
        infer_index = exp.get("infer_output_index")
        if compressors:
            if infer_index is None or infer_index < 0:
                infer_index = len(compressors) - 1
        else:
            infer_index = -1

        trigger_every = exp.get("trigger_every")
        if trigger_every is None:
            trigger_every = exp_hparams.get("trigger_every", 15)
        
        trainer = Trainer(
            config=config,
            compressors=compressors,
            compile_mode=exp["mode"],
            trigger_every=trigger_every,  # ← 使用统一的配置
            enable_diff_check=exp.get("enable_diff_check", True),
            compile_training_backbone=exp["compile_training_backbone"],
            log_dir=os.path.join("logs", f"pruning_{args.experiment}", exp["name"]),
            device=device,
            infer_output_index=infer_index,
            wandb_enabled=exp_hparams.get("use_wandb", False),
            wandb_project=exp_hparams.get("wandb_project"),
            wandb_run_name=f"{exp['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            min_epoch_before_compress=exp_hparams.get("min_epoch_before_compress", 0),
            prune_training_model=exp_hparams.get("prune_training_model", False),
            wandb_config={
                "experiment": exp["name"],
                "experiment_type": args.experiment,
                "env_id": exp_hparams["env_id"],
                "group": exp_hparams.get("wandb_group"),
                "tags": exp_hparams.get("wandb_tags", []),
                "prune_ratio": exp_hparams.get("prune_ratio"),
                "prune_technique": exp_hparams.get("prune_technique"),
                "prune_training_model": exp_hparams.get("prune_training_model"),
                "hidden_dim": exp_hparams["hidden_dim"],
                "hidden_depth": exp_hparams["hidden_depth"],
                "trigger_every": trigger_every,
                "seed": exp_hparams["seed"],
                "compile_backend": exp_hparams.get("compile_backend"),
            },
            async_warmup=exp.get("async_warmup", False),
        )
        trainer.run(num_epochs=exp_hparams["num_epochs"])
        trainer.summary()
        
        print(f"\n✅ Completed: {exp['name']}")
    
    print(f"\n{'='*60}")
    print(f"All experiments completed!")
    print(f"Logs saved to: logs/pruning_{args.experiment}/")
    print(f"{'='*60}\n")
    
    ray.shutdown()


if __name__ == "__main__":
    main()


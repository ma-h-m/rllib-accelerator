import os
import random

import numpy as np
import ray
import torch
from datetime import datetime
from ray.rllib.algorithms.ppo import PPOConfig

from framework.trainer import Trainer
from compression.compile_compressor import CompileCompressor
from compression.quant_compressor import QuantCompressor
from config import DEFAULT_HPARAMS, EXPERIMENTS

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Custom ModelV2.*")
warnings.filterwarnings("ignore", message=".*Install gputil.*")
warnings.filterwarnings("ignore", message=".*remote_workers.*")

# 注册全局模型
from models.policy import CustomPolicyNet    # noqa


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
        elif name == "quant":
            comps.append(
                QuantCompressor(
                    diff_threshold=hparams["quant_diff_threshold"],
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
                    prune_steps=hparams.get("prune_steps", 10),
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
                    prune_steps=hparams.get("prune_steps", 10),
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


if __name__ == "__main__":

    ray.init(include_dashboard=False)

    hparams = DEFAULT_HPARAMS
    device = resolve_device(hparams["device"])
    apply_global_seed(hparams.get("seed"))
    if hparams.get("seed") is not None:
        print(f"[main] Using seed: {hparams['seed']}")
    print(f"[main] Using device: {device}")

    hidden_layers = [hparams["hidden_dim"]] * hparams["hidden_depth"]

    for exp in EXPERIMENTS:
        print(f"\n========== Running {exp['name']} ({exp['mode'].value}) ==========")
        config = build_config(hidden_layers, device, hparams)
        compressors = build_compressors(exp, device, hparams)
        infer_index = exp.get("infer_output_index")
        if compressors:
            if infer_index is None or infer_index < 0:
                infer_index = len(compressors) - 1
        else:
            infer_index = -1

        trainer = Trainer(
            config=config,
            compressors=compressors,
            compile_mode=exp["mode"],
            trigger_every=exp.get("trigger_every", 0),
            enable_diff_check=exp.get("enable_diff_check", True),
            compile_training_backbone=exp["compile_training_backbone"],
            log_dir=os.path.join("logs", exp["name"]),
            device=device,
            infer_output_index=infer_index,
            wandb_enabled=hparams.get("use_wandb", False),
            wandb_project=hparams.get("wandb_project"),
            wandb_run_name=f"{exp['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            wandb_config={
                "experiment": exp["name"],
                "env_id": hparams["env_id"],
                "group": hparams.get("wandb_group"),
            },
            async_warmup=exp.get("async_warmup", False),
        )
        trainer.run(num_epochs=hparams["num_epochs"])
        trainer.summary()

    ray.shutdown()

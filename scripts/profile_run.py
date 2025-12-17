#!/usr/bin/env python3
"""
Utility script to capture PyTorch profiler traces for different compile modes.
Runs a short training session with reduced batch sizes to keep traces small.
"""

import argparse
import os
from copy import deepcopy
from datetime import datetime

import ray
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from torch.profiler import (
    profile,
    schedule,
    tensorboard_trace_handler,
    ProfilerActivity,
)

from compression.compile_compressor import CompileCompressor
from config import DEFAULT_HPARAMS
from framework.policy_manager import CompileMode
from framework.trainer import Trainer
# 注册自定义模型
from models.policy import CustomPolicyNet  # noqa


def resolve_device(config_device: str):
    requested = os.environ.get("ACCEL_DEVICE", config_device)
    normalized = requested.lower()
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        print(f"[profile] ⚠️ Requested device '{requested}' unavailable, fallback to CPU.")
        return "cpu"
    return normalized if normalized.startswith("cuda") or normalized == "cpu" else requested


def build_config(hidden_layers, device: str, hparams):
    use_gpu = device.startswith("cuda") and torch.cuda.is_available()
    config = (
        PPOConfig()
        .environment(hparams["env_id"])
        .framework("torch")
        .resources(num_gpus=1 if use_gpu else 0)
        .training(
            model={
                "custom_model": "custom_policy",
                "fcnet_hiddens": hidden_layers,
                "custom_model_config": {
                    "use_residual": hparams["use_residual"],
                    "device": device,
                },
            },
            train_batch_size=hparams["train_batch_size"],
            lr=hparams["lr"],
        )
    )
    
    # 兼容不同版本的 Ray API
    try:
        config = config.env_runners(
            num_env_runners=hparams["num_rollout_workers"],
            rollout_fragment_length=hparams["rollout_fragment_length"],
        )
    except AttributeError:
        config = config.rollouts(
            num_rollout_workers=hparams["num_rollout_workers"],
            rollout_fragment_length=hparams["rollout_fragment_length"],
        )
    
    return config


def main():
    parser = argparse.ArgumentParser(description="Run a short profiling session.")
    parser.add_argument(
        "--mode",
        choices=["none", "sync", "async"],
        default="sync",
        help="Compile mode to profile.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Number of epochs to record inside the profiler.",
    )
    parser.add_argument(
        "--out-dir",
        default="profile_logs",
        help="Directory used to store profiler traces.",
    )
    args = parser.parse_args()

    hparams = deepcopy(DEFAULT_HPARAMS)
    # Reduce workload so profiler runs quickly
    hparams["train_batch_size"] = max(1000, hparams["train_batch_size"] // 4)
    hparams["rollout_fragment_length"] = max(250, hparams["rollout_fragment_length"] // 4)
    hparams["num_rollout_workers"] = min(2, hparams["num_rollout_workers"])

    device = resolve_device(hparams["device"])
    hidden_layers = [hparams["hidden_dim"]] * hparams["hidden_depth"]

    compile_mode = {
        "none": CompileMode.NONE,
        "sync": CompileMode.SYNC,
        "async": CompileMode.ASYNC,
    }[args.mode]

    ray.init(include_dashboard=False)

    config = build_config(hidden_layers, device, hparams)
    compressors = [
        CompileCompressor(
            backend=hparams["compile_backend"],
            diff_threshold=hparams["compile_diff_threshold"],
            device=device,
        ),
    ]
    trainer = Trainer(
        config=config,
        compressors=compressors,
        compile_mode=compile_mode,
        trigger_every=0,
        enable_diff_check=False,
        compile_training_backbone=(compile_mode != CompileMode.NONE),
        log_dir=os.path.join("logs", f"profile_{args.mode}"),
        device=device,
        wandb_enabled=False,
        async_warmup=(compile_mode == CompileMode.ASYNC),
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_dir = os.path.join(args.out_dir, f"{args.mode}_{ts}")
    os.makedirs(trace_dir, exist_ok=True)

    activities = [ProfilerActivity.CPU]
    if device.startswith("cuda") and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    prof_schedule = schedule(wait=1, warmup=1, active=max(1, args.epochs - 2), repeat=1)
    print(f"[profile] Traces will be written to: {trace_dir}")

    with profile(
        activities=activities,
        schedule=prof_schedule,
        on_trace_ready=tensorboard_trace_handler(trace_dir),
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for epoch in range(1, args.epochs + 1):
            trainer.train_epoch(epoch)
            prof.step()

    trainer.summary()
    ray.shutdown()


if __name__ == "__main__":
    main()

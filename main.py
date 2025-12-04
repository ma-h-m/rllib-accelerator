import os

import ray
import torch
from datetime import datetime
from ray.rllib.algorithms.ppo import PPOConfig

from framework.trainer import Trainer
from compression.compile_compressor import CompileCompressor
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


def build_config(hidden_layers, device: str, hparams):
    use_gpu = device.startswith("cuda") and torch.cuda.is_available()
    num_workers = hparams["num_rollout_workers"]
    gpus_per_worker = (1.0 / num_workers) if (use_gpu and num_workers > 0) else 0.0
    return (
        PPOConfig()
        .environment(hparams["env_id"])
        .framework("torch")
        .resources(
            num_gpus=1 if use_gpu else 0,
            num_gpus_per_worker=gpus_per_worker,
        )
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
        .rollouts(
            num_rollout_workers=hparams["num_rollout_workers"],
            rollout_fragment_length=hparams["rollout_fragment_length"],
        )
    )


if __name__ == "__main__":

    ray.init(include_dashboard=False)

    hparams = DEFAULT_HPARAMS
    device = resolve_device(hparams["device"])
    print(f"[main] Using device: {device}")

    hidden_layers = [hparams["hidden_dim"]] * hparams["hidden_depth"]

    for exp in EXPERIMENTS:
        print(f"\n========== Running {exp['name']} ({exp['mode'].value}) ==========")
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
            compile_mode=exp["mode"],
            trigger_every=exp.get("trigger_every", 0),
            enable_diff_check=exp.get("enable_diff_check", True),
            compile_training_backbone=exp["compile_training_backbone"],
            log_dir=os.path.join("logs", exp["name"]),
            device=device,
            wandb_enabled=hparams.get("use_wandb", False),
            wandb_project=hparams.get("wandb_project"),
            wandb_run_name=f"{exp['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            wandb_config={"experiment": exp["name"], "env_id": hparams["env_id"]},
            async_warmup=exp.get("async_warmup", False),
        )
        trainer.run(num_epochs=hparams["num_epochs"])
        trainer.summary()

    ray.shutdown()

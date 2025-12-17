# path: framework/trainer.py

import os
import json
import time
from datetime import datetime
from typing import List

import numpy as np
import ray
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.sample_batch import concat_samples, SampleBatch, MultiAgentBatch

from compression.base import BaseCompressor
from framework.policy_manager import PolicyManager, CompileMode


class Trainer:
    """
    主训练驱动层：
        - RLlib rollout + train
        - 调用 PolicyManager 进行同步/异步压缩
        - 日志统计
        - 支持任意压缩器列表（compile/quant/prune/...）
    """

    def __init__(self,
                 config: PPOConfig,
                 compressors: List[BaseCompressor],
                 compile_mode=CompileMode.NONE,
                 trigger_every=5,
                 enable_diff_check=True,
                 compile_training_backbone=False,
                 log_dir="logs",
                 device: str = "cpu",
                 infer_output_index: int = 0,
                 wandb_enabled: bool = False,
                 wandb_project: str = None,
                 wandb_run_name: str = None,
                 wandb_config: dict = None,
                 async_warmup: bool = True,
                 min_epoch_before_compress: int = 0,
                 prune_training_model: bool = False):

        # 构建 RLlib algorithm
        self.algo = config.build()

        # 压缩管理器
        self.manager = PolicyManager(
            algo=self.algo,
            compressors=compressors,
            mode=compile_mode,
            trigger_every=trigger_every,
            enable_diff_check=enable_diff_check,
            compile_training_backbone=compile_training_backbone,
            device=device,
            infer_output_index=infer_output_index,
            async_warmup=async_warmup,
            min_epoch_before_compress=min_epoch_before_compress,
            prune_training_model=prune_training_model,
        )

        # 日志
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"{compile_mode.value}_{timestamp}.jsonl")
        self.log_file = open(self.log_path, "w")

        self.stats = []
        self.compile_mode = compile_mode
        self.device = device
        self._pending_episode_rewards = {}

        self.wandb_run = None
        if wandb_enabled and wandb_project:
            try:
                import wandb
                run_name = wandb_run_name or f"{compile_mode.value}_{timestamp}"
                cfg = {
                    "compile_mode": compile_mode.value,
                    "device": str(device),
                    "trigger_every": trigger_every,
                    "enable_diff_check": enable_diff_check,
                }
                group = None
                if wandb_config:
                    group = wandb_config.pop("group", None)
                    cfg.update(wandb_config)
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    name=run_name,
                    config=cfg,
                    group=group,
                )
                self._wandb = wandb
            except ImportError:
                print("[Trainer] ⚠️ 未检测到 wandb，跳过云端日志。")
                self.wandb_run = None
        elif wandb_enabled and not wandb_project:
            print("[Trainer] ⚠️ 未提供 wandb 项目名，跳过云端日志。")

    # ------------------------------------------------------------
    # 写日志到 JSONL
    # ------------------------------------------------------------
    def _log(self, rec):
        json.dump(rec, self.log_file)
        self.log_file.write("\n")
        self.log_file.flush()
        if self.wandb_run is not None:
            self.wandb_run.log(rec, step=rec.get("epoch"))

    # ------------------------------------------------------------
    # 单个 epoch
    # ------------------------------------------------------------
    def train_epoch(self, epoch: int):
        """
        一个完整 epoch：
            1) async 模式：尝试 swap（若后台已压缩完成）
            2) rollout
            3) learn_on_batch
            4) trigger（同步或异步压缩）
            5) 统计日志
        """

        # ==================================================================
        # (1) async 模式：尝试 swap（将 pending compiled backbone 下发）
        # ==================================================================
        meta_swap = self.manager.maybe_swap()

        # ==================================================================
        # (2) rollout：采样
        # ==================================================================
        t_rollout_start = time.time()
        workers = self.algo.workers.remote_workers()
        if workers:
            samples = ray.get([w.sample.remote() for w in workers])
        else:
            samples = [self.algo.workers.local_worker().sample()]
        rollout_reward_mean = self._estimate_reward_from_batches(samples)
        train_batch = concat_samples(samples)
        sample_count = train_batch.count
        t_rollout_end = time.time()
        rollout_time = t_rollout_end - t_rollout_start

        # ==================================================================
        # (3) train：local worker
        # ==================================================================
        with self.manager.model_lock:
            t_train_start = time.time()
            result = self.algo.workers.local_worker().learn_on_batch(train_batch)
            # 训练后立即同步权重，保持 rollout workers on-policy
            self.algo.workers.sync_weights()
            # 若推理 backbone 支持仅更新权重，则在同步后立即推送最新参数
            self.manager.push_weight_update()
            t_train_end = time.time()
        train_time = t_train_end - t_train_start

        # ==================================================================
        # (4) trigger（同步或异步触发压缩）
        # ==================================================================
        meta_trigger = self.manager.maybe_trigger(epoch)
        
        step_time = (time.time() - t_rollout_start)
        throughput = sample_count / step_time

        # ==================================================================
        # (5) compile stats
        # ==================================================================
        compile_latency = None
        swap_latency = None
        compressor_name = self.manager.get_infer_compressor_name()

        # SYNC → 当前 epoch 编译
        if self.compile_mode == CompileMode.SYNC and meta_trigger:
            info = meta_trigger.get(compressor_name)
            if info:
                compile_latency = info.get("latency")

        # ASYNC → 在 swap 时统计
        if self.compile_mode == CompileMode.ASYNC and meta_swap:
            info = meta_swap.get(compressor_name)
            if info:
                compile_latency = info.get("latency")
            swap_latency = meta_swap.get("SwapLatency")

        # ==================================================================
        # (6) log
        # ==================================================================
        reward_mean = result.get("episode_reward_mean")
        if reward_mean is None:
            reward_mean = rollout_reward_mean

        # Extract sparsity info from compression meta
        sparsity = None
        if meta_swap and compressor_name:
            info = meta_swap.get(compressor_name)
            if info and "actual_sparsity" in info:
                sparsity = info["actual_sparsity"]
        if sparsity is None and meta_trigger and compressor_name:
            info = meta_trigger.get(compressor_name)
            if info and "actual_sparsity" in info:
                sparsity = info["actual_sparsity"]
        
        rec = {
            "epoch": epoch,
            "reward_mean": reward_mean,
            "total_time": step_time,
            "rollout_time": rollout_time,
            "train_time": train_time,
            "throughput": throughput,
            "compile_latency": compile_latency,
            "swap_latency": swap_latency,
            "inference_time": self._collect_inference_time(),
            "sparsity": sparsity,  # Add sparsity tracking
        }
        avg_infer = rec["inference_time"]
        workers = max(len(self.algo.workers.remote_workers()), 1)
        avg_infer /= workers
        rec["inference_time"] = avg_infer
        rec["env_time"] = max(0.0, rec["rollout_time"] - avg_infer)
        self.stats.append(rec)
        self._log(rec)

        print(
            f"[{self.compile_mode.value.upper()}] Epoch {epoch:3d} | "
            f"Reward={rec['reward_mean']:.2f} | "
            f"Samples={sample_count} | "
            f"Total={step_time:.2f}s "
            f"(Rollout={rollout_time:.2f}s, Train={train_time:.2f}s) | "
            f"Thrpt={throughput:.1f}/s | "
            f"Compile={compile_latency} | "
            f"Swap={swap_latency} | "
            f"Infer={rec['inference_time']:.3f}s | Env={rec['env_time']:.3f}s"
        )

    # ------------------------------------------------------------
    # 训练主循环
    # ------------------------------------------------------------
    def run(self, num_epochs=10):
        for e in range(1, num_epochs + 1):
            self.train_epoch(e)

    # ------------------------------------------------------------
    # 打印总结
    # ------------------------------------------------------------
    def summary(self):
        if not self.stats:
            print(f"\n=== Summary ({self.compile_mode.value}) ===")
            print("No epochs recorded.")
            return

        total_epochs = len(self.stats)
        reward_sum = sum(s["reward_mean"] for s in self.stats)
        time_sum = sum(s["total_time"] for s in self.stats)
        rollout_sum = sum(s["rollout_time"] for s in self.stats)
        train_sum = sum(s["train_time"] for s in self.stats)
        throughput_sum = sum(s["throughput"] for s in self.stats)

        reward_avg = reward_sum / total_epochs
        time_avg = time_sum / total_epochs
        rollout_avg = rollout_sum / total_epochs
        train_avg = train_sum / total_epochs
        throughput_avg = throughput_sum / total_epochs

        compile_latencies = [s["compile_latency"] for s in self.stats if s["compile_latency"] is not None]
        compile_avg = sum(compile_latencies) / len(compile_latencies) if compile_latencies else None

        # Calculate sparsity statistics
        sparsities = [s.get("sparsity") for s in self.stats if s.get("sparsity") is not None]
        final_sparsity = sparsities[-1] if sparsities else None

        print(f"\n=== Summary ({self.compile_mode.value}) ===")
        print(f"Epochs: {total_epochs}")
        print(f"Reward mean (avg over epochs): {reward_avg:.2f}")
        print(f"Total time (avg per epoch): {time_avg:.2f}s")
        print(f"  Rollout time avg: {rollout_avg:.2f}s | Train time avg: {train_avg:.2f}s")
        print(f"Throughput (avg samples/s): {throughput_avg:.1f}")
        if final_sparsity is not None:
            print(f"Final sparsity: {final_sparsity*100:.1f}%")
        if compile_avg is not None:
            print(f"Compile latency (avg when available): {compile_avg:.3f}s")
        else:
            print("Compile latency: N/A")

        self.log_file.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()

    def _estimate_reward_from_batches(self, batches) -> float:
        completed = []

        def accumulate(batch: SampleBatch):
            if not isinstance(batch, SampleBatch):
                return
            if SampleBatch.REWARDS not in batch or SampleBatch.EPS_ID not in batch:
                return

            rewards = np.asarray(batch[SampleBatch.REWARDS], dtype=np.float32)
            eps_ids = np.asarray(batch[SampleBatch.EPS_ID], dtype=np.int64)

            # RLlib 2.x no longer guarantees `dones` to be present, instead splitting
            # the signal into `terminateds`/`truncateds`. Merge them so we always
            # detect the end of an episode.
            if SampleBatch.DONES in batch:
                dones = np.asarray(batch[SampleBatch.DONES], dtype=np.bool_)
            else:
                terminateds = batch.get(SampleBatch.TERMINATEDS)
                truncateds = batch.get(SampleBatch.TRUNCATEDS)
                if terminateds is None and truncateds is None:
                    return
                term = (
                    np.asarray(terminateds, dtype=np.bool_)
                    if terminateds is not None else np.zeros_like(rewards, dtype=np.bool_)
                )
                trunc = (
                    np.asarray(truncateds, dtype=np.bool_)
                    if truncateds is not None else np.zeros_like(rewards, dtype=np.bool_)
                )
                dones = np.logical_or(term, trunc)

            for r, done, eps in zip(rewards, dones, eps_ids):
                acc = self._pending_episode_rewards.get(eps, 0.0)
                acc += float(r)
                if done:
                    completed.append(acc)
                    self._pending_episode_rewards.pop(eps, None)
                else:
                    self._pending_episode_rewards[eps] = acc

        for batch in batches:
            if isinstance(batch, MultiAgentBatch):
                for sub_batch in batch.policy_batches.values():
                    accumulate(sub_batch)
            else:
                accumulate(batch)

        if completed:
            return float(np.mean(completed))
        if self._pending_episode_rewards:
            return float(np.mean(list(self._pending_episode_rewards.values())))
        return 0.0

    def _collect_inference_time(self) -> float:
        """汇总 rollout workers（以及无 remote 时的 local worker）的推理耗时。"""
        total = 0.0

        def _pull(worker):
            def inner(policy, pid):
                model = getattr(policy, "model", None)
                if model is not None and hasattr(model, "consume_inference_time"):
                    return model.consume_inference_time()
                return 0.0
            values = worker.foreach_policy(inner)
            return sum(values)

        workers = self.algo.workers.remote_workers()
        if workers:
            totals = ray.get([w.apply.remote(_pull) for w in workers])
            total += sum(totals)
        else:
            total += _pull(self.algo.workers.local_worker())

        return total

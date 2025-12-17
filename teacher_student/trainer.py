import json
import os
import time
from datetime import datetime
from typing import Dict, Any

import numpy as np
import ray
import torch
import torch.nn.functional as F
import gymnasium as gym
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.sample_batch import (
    SampleBatch,
    MultiAgentBatch,
    concat_samples,
)
from ray.rllib.evaluation.postprocessing import Postprocessing

from models.policy import PolicyBackbone


class TeacherStudentTrainer:
    """
    Minimal trainer where a small “student” policy interacts with the
    environment via RLlib PPO, while a large “teacher” backbone is updated
    on the exact same trajectories.
    """

    def __init__(self, hparams: Dict[str, Any]):
        self.hparams = dict(hparams)
        self.device = torch.device(self.hparams.get("device", "cpu"))
        self._apply_global_seed(self.hparams.get("seed"))
        self._build_student_algo()
        self._build_teacher_model()
        self.stats = []
        self._init_logger()
        self._pending_episode_rewards = {}
        self.teacher_eval_episodes = int(self.hparams.get("teacher_eval_episodes", 0))

    def _student_hidden_layers(self):
        dim = self.hparams["student_hidden_dim"]
        depth = self.hparams["student_hidden_depth"]
        return [dim] * depth

    def _teacher_hidden_layers(self):
        dim = self.hparams["teacher_hidden_dim"]
        depth = self.hparams["teacher_hidden_depth"]
        return [dim] * depth

    def _build_student_algo(self):
        student_layers = self._student_hidden_layers()
        training_kwargs = {
            "model": {
                "custom_model": "custom_policy",
                "fcnet_hiddens": student_layers,
                "custom_model_config": {
                    "use_residual": self.hparams["student_use_residual"],
                    "device": self.hparams["device"],
                },
            },
            "train_batch_size": self.hparams["train_batch_size"],
            "lr": self.hparams["student_lr"],
        }
        lr_schedule = self._build_lr_schedule()
        if lr_schedule is not None:
            training_kwargs["lr_schedule"] = lr_schedule

        cfg = (
            PPOConfig()
            .environment(self.hparams["env_id"])
            .framework("torch")
            .training(**training_kwargs)
        )
        
        # 兼容不同版本的 Ray API
        try:
            cfg = cfg.env_runners(
                num_env_runners=self.hparams["num_rollout_workers"],
                rollout_fragment_length=self.hparams["rollout_fragment_length"],
            )
        except AttributeError:
            cfg = cfg.rollouts(
                num_rollout_workers=self.hparams["num_rollout_workers"],
                rollout_fragment_length=self.hparams["rollout_fragment_length"],
            )
        seed = self.hparams.get("seed")
        if seed is not None:
            cfg.seed = seed
        self.student_algo = cfg.build()
        policy = self.student_algo.get_policy()
        self.obs_shape = policy.observation_space.shape
        self.num_actions = policy.action_space.n

    def _build_teacher_model(self):
        obs_dim = int(np.prod(self.obs_shape))
        hidden_layers = self._teacher_hidden_layers()
        self.teacher_model = PolicyBackbone(
            obs_dim,
            self.num_actions,
            hidden_layers,
            use_residual=self.hparams["teacher_use_residual"],
        ).to(self.device)
        self.teacher_optimizer = torch.optim.Adam(
            self.teacher_model.parameters(), lr=self.hparams["teacher_lr"]
        )
        self._setup_teacher_scheduler()

    def _init_logger(self):
        self.log_dir = self.hparams.get("log_dir") or "logs/teacher_student"
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = self.hparams.get("run_name", "teacher_student")
        run_name = f"{base}_{timestamp}"
        self.log_path = os.path.join(self.log_dir, f"{run_name}.jsonl")
        self.log_file = open(self.log_path, "w")

    def _prepare_batch(self, batch):
        if isinstance(batch, MultiAgentBatch):
            list_batches = []
            for sub_batch in batch.policy_batches.values():
                list_batches.append(sub_batch)
            batch = concat_samples(list_batches)
        return batch

    def _train_teacher(self, batch: SampleBatch, epoch: int):
        if SampleBatch.ACTION_LOGP not in batch:
            return {}
        obs = batch[SampleBatch.OBS]
        obs = np.asarray(obs, dtype=np.float32).reshape(len(obs), -1)
        obs_t = torch.as_tensor(obs, device=self.device)
        actions = torch.as_tensor(batch[SampleBatch.ACTIONS], device=self.device)
        advantages = torch.as_tensor(batch[Postprocessing.ADVANTAGES], device=self.device)
        value_targets = torch.as_tensor(
            batch[Postprocessing.VALUE_TARGETS], device=self.device
        )
        behavior_logp = torch.as_tensor(batch[SampleBatch.ACTION_LOGP], device=self.device)

        logits, values = self.teacher_model(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(actions.long())
        ratio = torch.exp(logp - behavior_logp)
        clip_param = self.hparams["clip_param"]
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
        surrogate = torch.min(ratio * advantages, clipped_ratio * advantages)
        policy_loss = -surrogate.mean()

        value_loss = F.mse_loss(values.view(-1), value_targets)
        entropy = dist.entropy().mean()

        loss = (
            policy_loss
            + self.hparams["value_loss_coeff"] * value_loss
            - self.hparams["entropy_coeff"] * entropy
        )
        self.teacher_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.teacher_model.parameters(), self.hparams["max_grad_norm"]
        )
        self.teacher_optimizer.step()
        self._maybe_decay_teacher_lr(epoch)

        return {
            "teacher_policy_loss": policy_loss.item(),
            "teacher_value_loss": value_loss.item(),
            "teacher_entropy": entropy.item(),
        }

    def train_epoch(self, epoch: int):
        t_epoch_start = time.time()

        workers = self.student_algo.workers.remote_workers()
        if workers:
            samples = ray.get([w.sample.remote() for w in workers])
        else:
            samples = [self.student_algo.workers.local_worker().sample()]
        t_rollout_end = time.time()
        batch = concat_samples(samples)
        batch = self._prepare_batch(batch)
        sample_count = batch.count
        rollout_time = t_rollout_end - t_epoch_start

        t_student_start = time.time()
        result = self.student_algo.workers.local_worker().learn_on_batch(batch)
        self.student_algo.workers.sync_weights()
        t_student_end = time.time()
        teacher_stats = self._train_teacher(batch, epoch)
        t1 = time.time()
        student_train_time = t_student_end - t_student_start
        teacher_train_time = t1 - t_student_end
        total_time = t1 - t_epoch_start
        throughput = sample_count / max(total_time, 1e-8)
        teacher_eval_reward = self._evaluate_teacher()
        reward_mean = result.get("episode_reward_mean")
        if reward_mean is None:
            reward_mean = self._estimate_reward(samples)
        inference_time = self._collect_inference_time()
        env_time = max(0.0, rollout_time - inference_time)

        rec = {
            "epoch": epoch,
            "reward_mean": reward_mean,
            "student_loss": result.get("policy_loss"),
            "teacher_policy_loss": teacher_stats.get("teacher_policy_loss"),
            "teacher_value_loss": teacher_stats.get("teacher_value_loss"),
            "teacher_entropy": teacher_stats.get("teacher_entropy"),
            "total_time": total_time,
            "rollout_time": rollout_time,
            "train_time": teacher_train_time,
            "student_train_time": student_train_time,
            "throughput": throughput,
            "inference_time": inference_time,
            "env_time": env_time,
            "samples": sample_count,
            "teacher_eval_reward": teacher_eval_reward,
        }
        self.stats.append(rec)
        self._log(rec)

        eval_msg = ""
        if teacher_eval_reward is not None:
            eval_msg = f" | TchReward={teacher_eval_reward:.2f}"

        print(
            f"[TeacherStudent] Epoch {epoch:03d} | "
            f"Reward={reward_mean:.2f} | "
            f"StudentLoss={rec['student_loss']} | "
            f"TchPolicy={rec['teacher_policy_loss']} | "
            f"TchValue={rec['teacher_value_loss']} | "
            f"TchEntropy={rec['teacher_entropy']} | "
            f"Total={rec['total_time']:.3f}s "
            f"(Rollout={rec['rollout_time']:.3f}s, "
            f"StuTrain={rec['student_train_time']:.3f}s, "
            f"TchTrain={rec['train_time']:.3f}s) | "
            f"Thrpt={rec['throughput']:.1f}/s | "
            f"Infer={rec['inference_time']:.3f}s | Env={rec['env_time']:.3f}s"
            f"{eval_msg}"
        )

    def run(self, num_epochs: int):
        for epoch in range(1, num_epochs + 1):
            self.train_epoch(epoch)

    def summary(self):
        if not self.stats:
            print("\n[TeacherStudent] No epochs recorded.")
            return
        reward_avg = np.mean([s["reward_mean"] for s in self.stats])
        print(f"\n[TeacherStudent] Completed {len(self.stats)} epochs")
        print(f"Average reward: {reward_avg:.2f}")
        print(f"Logs saved to {self.log_path}")

    def shutdown(self):
        self.student_algo.stop()
        if hasattr(self, "log_file") and not self.log_file.closed:
            self.log_file.close()

    def _log(self, rec):
        json.dump(rec, self.log_file)
        self.log_file.write("\n")
        self.log_file.flush()

    def _estimate_reward(self, batches) -> float:
        completed = []

        def accumulate(batch: SampleBatch):
            if not isinstance(batch, SampleBatch):
                return
            if SampleBatch.REWARDS not in batch or SampleBatch.EPS_ID not in batch:
                return

            rewards = np.asarray(batch[SampleBatch.REWARDS], dtype=np.float32)
            eps_ids = np.asarray(batch[SampleBatch.EPS_ID], dtype=np.int64)

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
                for sub in batch.policy_batches.values():
                    accumulate(sub)
            else:
                accumulate(batch)

        if completed:
            return float(np.mean(completed))
        if self._pending_episode_rewards:
            return float(np.mean(list(self._pending_episode_rewards.values())))
        return 0.0

    def _apply_global_seed(self, seed):
        if seed is None:
            return
        np.random.seed(seed)
        torch.manual_seed(seed)
        try:
            import random
            random.seed(seed)
        except ImportError:
            pass
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_lr_schedule(self):
        decay_conf = self.hparams.get("lr_decay") or {}
        if not decay_conf.get("enabled"):
            return None
        base_lr = self.hparams["student_lr"]
        gamma = float(decay_conf.get("gamma", 0.5))
        step_epochs = max(1, int(decay_conf.get("step_epochs", 1)))
        min_lr = float(decay_conf.get("min_lr", 0.0))
        total_epochs = self.hparams["num_epochs"]
        steps_per_epoch = max(1, self.hparams["train_batch_size"])

        schedule = [[0, base_lr]]
        current_lr = base_lr
        epoch = step_epochs
        while epoch <= total_epochs:
            current_lr = max(min_lr, current_lr * gamma)
            schedule.append([epoch * steps_per_epoch, current_lr])
            epoch += step_epochs

        return schedule if len(schedule) > 1 else None

    def _setup_teacher_scheduler(self):
        decay_conf = self.hparams.get("lr_decay") or {}
        if not decay_conf.get("enabled"):
            self._teacher_decay = None
            return
        self._teacher_decay = {
            "gamma": float(decay_conf.get("gamma", 0.5)),
            "step": max(1, int(decay_conf.get("step_epochs", 1))),
            "min_lr": float(decay_conf.get("min_lr", 0.0)),
            "next_epoch": max(1, int(decay_conf.get("step_epochs", 1))),
        }

    def _maybe_decay_teacher_lr(self, epoch: int):
        if not getattr(self, "_teacher_decay", None):
            return
        conf = self._teacher_decay
        if epoch < conf["next_epoch"]:
            return
        for group in self.teacher_optimizer.param_groups:
            new_lr = max(conf["min_lr"], group["lr"] * conf["gamma"])
            group["lr"] = new_lr
        conf["next_epoch"] += conf["step"]

    def _evaluate_teacher(self):
        episodes = getattr(self, "teacher_eval_episodes", 0)
        if episodes <= 0:
            return None
        env = gym.make(self.hparams["env_id"])
        total_reward = 0.0
        was_training = self.teacher_model.training
        self.teacher_model.eval()
        try:
            for _ in range(episodes):
                obs, _ = env.reset()
                done = False
                episode_reward = 0.0
                while not done:
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, -1)
                    with torch.no_grad():
                        logits, _ = self.teacher_model(obs_tensor)
                    action = torch.argmax(logits, dim=-1).item()
                    obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                total_reward += episode_reward
        finally:
            if was_training:
                self.teacher_model.train()
            env.close()
        return total_reward / episodes

    def _collect_inference_time(self) -> float:
        total = 0.0

        def _pull(worker):
            def inner(policy, pid):
                model = getattr(policy, "model", None)
                if model is not None and hasattr(model, "consume_inference_time"):
                    return model.consume_inference_time()
                return 0.0

            values = worker.foreach_policy(inner)
            return sum(values)

        workers = self.student_algo.workers.remote_workers()
        worker_count = len(workers)
        if worker_count > 0:
            totals = ray.get([w.apply.remote(_pull) for w in workers])
            total += sum(totals)
        else:
            total += _pull(self.student_algo.workers.local_worker())

        denom = worker_count if worker_count > 0 else 1
        return total / denom

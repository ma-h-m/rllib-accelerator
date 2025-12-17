import os
import ray
import time
import json
import threading
import logging
from enum import Enum
from datetime import datetime
import numpy as np

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.policy.sample_batch import concat_samples

# ---- 控制 Ray 日志 ----
os.environ["RAY_DEDUP_LOGS"] = "0"
logging.getLogger("ray").setLevel(logging.ERROR)

torch, nn = try_import_torch()
F = nn.functional


# ============================================================
# 三种模式
# ============================================================
class CompileMode(Enum):
    NONE = "none"
    SYNC = "sync"
    ASYNC = "async"


# ============================================================
# 纯 PyTorch 的前向骨干（用于被 torch.compile）
# 接口：forward(obs: Tensor) -> (logits, value)
# ============================================================
class PolicyBackbone(nn.Module):
    def __init__(self, in_dim: int, num_outputs: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, num_outputs)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor):
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        logits = self.policy_head(x)
        value = self.value_head(x)  # [B, 1]
        return logits, value


# ============================================================
# 自定义 PPO Policy 模型（RLlib 接口）
# - 训练永远用 self.backbone（未编译版本）
# - 推理可以用 compiled_backbone（由我们控制）
# ============================================================
class CustomPolicyNet(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        in_dim = obs_space.shape[0]
        self.in_dim = in_dim
        self.num_outputs = num_outputs

        # 未编译的训练用 backbone
        self.backbone = PolicyBackbone(in_dim, num_outputs)
        # 可选的编译后 backbone，仅用于推理
        self.compiled_backbone = None
        self.use_compiled = False

        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        # 统一转成 tensor，避免 numpy 参与计算图
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        else:
            obs = obs.float()

        # 选择当前使用的前向模块（训练 or 推理）
        bb = self.compiled_backbone if (self.use_compiled and self.compiled_backbone is not None) else self.backbone

        logits, value = bb(obs)
        # logits: [B, num_outputs], value: [B,1]
        self._value_out = value.view(-1)  # [B]
        return logits, state

    def value_function(self):
        return self._value_out

    # 供我们在 sampler 上切换推理模型用
    def set_compiled_backbone(self, compiled_bb: nn.Module):
        self.compiled_backbone = compiled_bb
        self.use_compiled = compiled_bb is not None


ModelCatalog.register_custom_model("custom_policy", CustomPolicyNet)


# ============================================================
# 编译 Hook：只针对 backbone
# ============================================================
class CompressionHook:
    @staticmethod
    def snapshot_backbone(train_model: CustomPolicyNet):
        """
        从 train_model.backbone 抽 snapshot：
        - 复制 state_dict（detach + cpu + clone）
        - 记录 in_dim / num_outputs
        """
        bb = train_model.backbone
        state_dict_raw = bb.state_dict()
        state_dict = {
            k: v.detach().cpu().clone()
            for k, v in state_dict_raw.items()
        }
        meta = (train_model.in_dim, train_model.num_outputs)
        return state_dict, meta

    @staticmethod
    def build_compiled_backbone(state_dict, meta, backend="inductor"):
        """
        给定 backbone 的 snapshot，构造一个新的 PolicyBackbone 并编译。
        """
        in_dim, num_outputs = meta
        bb = PolicyBackbone(in_dim, num_outputs)
        bb.load_state_dict(state_dict)

        t0 = time.time()
        compiled_bb = torch.compile(bb, backend=backend)
        compile_latency = time.time() - t0

        return compiled_bb, compile_latency


# ============================================================
# PolicyManager：显式维护训练模型 / 推理骨干
# - train_model：local worker 上的 CustomPolicyNet（未编译）
# - compiled_backbone：仅用于 sampler 推理
# ============================================================
class PolicyManager:
    def __init__(self, algo, mode=CompileMode.NONE, trigger_every=5, backend="inductor"):
        self.algo = algo
        self.mode = mode
        self.trigger_every = trigger_every
        self.backend = backend

        self.lock = threading.Lock()

        # local worker 上的训练模型（只用它的 backbone）
        self.train_model: CustomPolicyNet = self.algo.get_policy().model

        # 当前生效的 compiled backbone（用于 sampler）
        self.current_compiled_backbone = None

        # 异步 pending compiled backbone
        self.pending_compiled_backbone = None
        self.pending_compile_latency = None
        self.pending_copy_latency = None

        # 最近一次真正生效的编译统计（用于 log）
        self.last_compile_latency = None
        self.last_copy_latency = None
        self.last_swap_latency = None

    # ---------------------- 编译触发逻辑 ----------------------
    def maybe_trigger_compile(self, epoch: int) -> bool:
        if self.mode == CompileMode.NONE:
            return False
        if epoch % self.trigger_every != 0:
            return False

        if self.mode == CompileMode.SYNC:
            # 同步：直接 snapshot + compile + 下发到 sampler
            t_copy0 = time.time()
            state_dict, meta = CompressionHook.snapshot_backbone(self.train_model)
            copy_latency = time.time() - t_copy0

            compiled_bb, compile_latency = CompressionHook.build_compiled_backbone(
                state_dict, meta, backend=self.backend
            )

            self.current_compiled_backbone = compiled_bb
            self.last_compile_latency = compile_latency
            self.last_copy_latency = copy_latency
            self.last_swap_latency = None

            print(
                f"[SYNC Compile] ✅ Snapshot copy={copy_latency:.3f}s, "
                f"compile={compile_latency:.3f}s"
            )

            self._broadcast_compiled_backbone_to_samplers(compiled_bb)

        elif self.mode == CompileMode.ASYNC:
            # 异步：后台线程 snapshot + compile
            def worker():
                try:
                    print("[AsyncCompile] 🔧 Start background compilation...")

                    t_copy0 = time.time()
                    with self.lock:
                        state_dict, meta = CompressionHook.snapshot_backbone(self.train_model)
                    copy_latency = time.time() - t_copy0

                    compiled_bb, compile_latency = CompressionHook.build_compiled_backbone(
                        state_dict, meta, backend=self.backend
                    )

                    with self.lock:
                        self.pending_compiled_backbone = compiled_bb
                        self.pending_compile_latency = compile_latency
                        self.pending_copy_latency = copy_latency

                    print(
                        f"[AsyncCompile] ✅ Snapshot copy={copy_latency:.3f}s, "
                        f"compile={compile_latency:.3f}s (waiting to swap)"
                    )

                    del state_dict
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"[AsyncCompile] ❌ Failed: {e}")

            threading.Thread(target=worker, daemon=True).start()

        return True

    # ---------------------- 异步 swap：只替换 sampler 使用的 compiled backbone ----------------------
    def maybe_swap_infer_model(self) -> bool:
        if self.mode != CompileMode.ASYNC:
            return False

        with self.lock:
            if self.pending_compiled_backbone is None:
                return False

            compiled_bb = self.pending_compiled_backbone
            compile_latency = self.pending_compile_latency
            copy_latency = self.pending_copy_latency

            self.pending_compiled_backbone = None
            self.pending_compile_latency = None
            self.pending_copy_latency = None

        # 在锁外进行广播，避免长期持锁
        self.current_compiled_backbone = compiled_bb
        t0 = time.time()
        self._broadcast_compiled_backbone_to_samplers(compiled_bb)
        swap_latency = time.time() - t0

        self.last_compile_latency = compile_latency
        self.last_copy_latency = copy_latency
        self.last_swap_latency = swap_latency

        print(
            f"[AsyncCompile] 🔁 Swapped sampler compiled_backbone. "
            f"swap_latency={swap_latency:.3f}s"
        )
        return True

    # ---------------------- 把 compiled_backbone 推送到所有 sampler 上 ----------------------
    def _broadcast_compiled_backbone_to_samplers(self, compiled_bb: nn.Module):
        workers = self.algo.workers.remote_workers()

        def _set_compiled(worker):
            # 在远程 worker 环境中执行
            def _update_policy(policy, pid):
                if hasattr(policy.model, "set_compiled_backbone"):
                    policy.model.set_compiled_backbone(compiled_bb)
                return 1

            worker.foreach_policy(_update_policy)
            return 1

        ray.get([w.apply.remote(_set_compiled) for w in workers])
        print("[Broadcast] 📤 compiled_backbone updated on all sampler workers.")


# ============================================================
# RLTrainer：rollout + train + logging
# ============================================================
class RLTrainer:
    def __init__(self, config, compile_mode=CompileMode.NONE, log_dir="logs", trigger_every=5):
        self.algo = config.build()
        self.compile_mode = compile_mode
        self.manager = PolicyManager(self.algo, compile_mode, trigger_every)

        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"log_{compile_mode.value}_{timestamp}.jsonl")
        self.stats = []
        self.logf = open(self.log_path, "w")

    def _log(self, record):
        json.dump(record, self.logf)
        self.logf.write("\n")
        self.logf.flush()

    def train_epoch(self, epoch: int):
        # 1) ASYNC：看有没有新的 compiled_backbone 可以下发
        swapped = False
        if self.compile_mode == CompileMode.ASYNC:
            swapped = self.manager.maybe_swap_infer_model()

        t0 = time.time()

        # 2) Rollout：remote workers 用当前 compiled_backbone（若有）采样
        workers = self.algo.workers.remote_workers()
        samples = ray.get([w.sample.remote() for w in workers])
        train_batch = concat_samples(samples)
        sample_count = train_batch.count

        # 3) 编译触发
        triggered = self.manager.maybe_trigger_compile(epoch)

        # 4) Train：local worker 用 train_model（未编译）训练
        result = self.algo.workers.local_worker().learn_on_batch(train_batch)

        t1 = time.time()
        step_time = t1 - t0
        throughput = sample_count / step_time

        # 5) 日志：只在真正 compile / swap 的 epoch 记录 latency
        compile_latency = None
        copy_latency = None
        swap_latency = None

        if self.compile_mode == CompileMode.SYNC and triggered:
            compile_latency = self.manager.last_compile_latency
            copy_latency = self.manager.last_copy_latency
        elif self.compile_mode == CompileMode.ASYNC and swapped:
            compile_latency = self.manager.last_compile_latency
            copy_latency = self.manager.last_copy_latency
            swap_latency = self.manager.last_swap_latency

        log_rec = {
            "epoch": epoch,
            "mode": self.compile_mode.value,
            "reward_mean": result.get("episode_reward_mean", 0.0),
            "train_time": step_time,
            "throughput": throughput,
            "compile_latency": compile_latency,
            "copy_latency": copy_latency,
            "swap_latency": swap_latency,
        }
        self._log(log_rec)
        self.stats.append(log_rec)

        print(
            f"[{self.compile_mode.value.upper()}] Epoch {epoch:<3d} | "
            f"Reward={log_rec['reward_mean']:<8.2f} | "
            f"Samples={sample_count:<6d} | "
            f"Time={step_time:<6.2f}s | "
            f"Throughput={throughput:<8.2f} samples/s | "
            f"Compile={compile_latency}"
        )

    def run(self, num_epochs=10):
        for e in range(1, num_epochs + 1):
            self.train_epoch(e)

    def summary(self):
        print(f"\n=== Summary ({self.compile_mode.value}) ===")
        for s in self.stats:
            print(
                f"Epoch {s['epoch']}: reward={s['reward_mean']:.2f}, "
                f"time={s['train_time']:.2f}s, "
                f"thrpt={s['throughput']:.1f}/s, "
                f"compile={s['compile_latency']}"
            )
        self.logf.close()


# ============================================================
# 主流程：对比三种模式
# ============================================================
if __name__ == "__main__":
    ray.init(include_dashboard=False, _metrics_export_port=None)

    base_config = (
        PPOConfig()
        .environment(env="CartPole-v1")
        .framework("torch")
        .training(
            model={"custom_model": "custom_policy"},
            train_batch_size=4000,
            lr=3e-4,
        )
    )
    
    # 兼容不同版本的 Ray API
    try:
        base_config = base_config.env_runners(num_env_runners=2)
    except AttributeError:
        base_config = base_config.rollouts(num_rollout_workers=2)

    trigger_every = 3

    modes = [CompileMode.NONE, CompileMode.SYNC, CompileMode.ASYNC]
    for m in modes:
        print(f"\n=========== Mode = {m.value} ===========")
        trainer = RLTrainer(base_config, compile_mode=m, log_dir="logs", trigger_every=trigger_every)
        trainer.run(num_epochs=5)
        trainer.summary()

    ray.shutdown()
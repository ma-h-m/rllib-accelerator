# RLlib Accelerator

This repository contains the code used in our HPML project on accelerating RLlib
training with PyTorch `torch.compile` and dynamic quantization. The project
focuses on the CartPole-v1 benchmark and provides end-to-end tooling for
benchmarking baseline PPO, synchronous compilation, asynchronous compilation
(with/without warmup), and asynchronous quantization with warmup.

## Quick Start

```bash
pip install -r requirements.txt          # include ray[rllib], torch>=2.1, matplotlib, wandb(optional)
python main.py                           # run all experiments enumerated in config.py
```

Key runtime controls live in `config.py`:

- `DEFAULT_HPARAMS`: environment name, batch sizes, learning rate (+ optional decay),
  model size, resource allocation, device, etc.
- `EXPERIMENTS`: list of scenarios to execute. Each entry specifies compile mode,
  compressor list, trigger cadence, warmup flag, and whether the training backbone
  should be compiled.

## wandb links
### Compile
https://api.wandb.ai/links/mahm/6oytcqih

### Pruning
https://api.wandb.ai/links/fm2859-columbia-university/shlmnaw0

## Results
### Compile
![Timing breakdown](results/none&sync&async/timing_main.png)

Asynchronous compilation reduces steady-state iteration time by overlapping compilation with training. However, both sync and async compilation show early-epoch latency spikes due to first-inference overhead, which must be removed via warmup to achieve stable speedup.

### Quantization
![Reward comparison](results/compile&quant&baseline/quality_comparison_layer=4_dim=512/reward_compare.png)
Quantization accelerates inference but introduces a trade-off between speed and learning stability. High-frequency quantization leads to reward instability, while lower frequency improves stability at the cost of slower convergence.


## Compression Modes

Each experiment is managed by `framework/trainer.py` and `framework/policy_manager.py`.
The policy manager owns a `CompressionController` that snapshots the local training
backbone, applies the configured compressor(s), and broadcasts models/weights to
rollout workers.

### Baseline (no compile)

- `mode = CompileMode.NONE`.
- `compressors = ["compile"]` but the compression pipeline is never triggered.
- Trainer simply performs PPO rollouts/train steps and relies on RLlib's default
  `sync_weights()` to keep workers on-policy.

### Sync Compile

- `mode = CompileMode.SYNC`.
- At the end of each epoch (or whenever the trigger policy fires), the controller
  takes a snapshot, runs `torch.compile` synchronously, and immediately swaps the
  compiled backbone into all rollout workers.
- Since the swap happens inside the training critical path, the epoch includes the
  compile latency; there is no overlap with rollout/learning time.

### Async Compile

- `mode = CompileMode.ASYNC`.
- The controller snapshots the training backbone and launches a background thread
  that compiles the model. At the start of the next epoch, `maybe_swap()` checks
  whether a compiled result is ready:
  - If the compiled structure changed, it swaps the entire module on all workers.
  - Otherwise it reuses the compiled module and **only** pushes the latest weights.
- `PolicyManager.push_weight_update()` guarantees the rollout policy remains on
  the newest parameters even while the compiled module is reused, so async compile
  behaves like the baseline with a faster forward path.

### Async Compile with Warmup

- Same as async compile, but `async_warmup=True`.
- After swapping the module the first time, every worker runs a dummy forward pass
  (`warmup_compiled_backbone()`) so that PyTorch captures the graph before real
  rollouts. This hides the first-iteration compilation overhead.

### Async Quant with Warmup

- `compressors = ["quant"]` (dynamic quantization on linear layers).
- Quantized modules cannot reuse training weights, so every swap transmits the full
  quantized backbone.
- `async_warmup=True` runs a dummy pass to prime CPU kernels. Trigger cadence and
  diff-threshold control how often re-quantization occurs.

## Benchmark & Plot Scripts

| Script                                                                              | Purpose                                                                                                                                                 |
| ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `scripts/benchmark_compile.py`                                                      | Micro-benchmark vanilla vs. `torch.compile` inference (CPU/GPU).                                                                                        |
| `scripts/benchmark_quant.py`                                                        | Compare vanilla vs. dynamic quantized inference throughput (CPU).                                                                                       |
| `results/none&sync&async/plot_times.py`                                             | Plot total/rollout/train/inference time plus compile/swap latency for baseline vs. compile variants. Supports explicit file selection and image export. |
| `results/compile&quant&baseline/quality_comparison_layer=4_dim=512/plot_rewards.py` | Plot smoothed reward curves for any subset of experiments (labels + files configurable).                                                                |
| `results/compile&quant&baseline/plot_inference_bars.py`                             | Generate bar charts comparing rollout/inference time and throughput across the three inference-speed directories (`dim=512/1024/2048`).                 |
| `results/compile&quant&baseline/plot_swap_latency.py`                               | Visualize swap latency measurements recorded in `swap_time_rec.csv`.                                                                                    |

## Logging

- Each experiment writes a JSONL trace under `logs/{exp-name}/` and `results/...`
  directories. Every record contains:
  - `rollout_time`, `train_time`, `total_time`, `inference_time`, `throughput`
  - `compile_latency` (sync mode) or swap metadata (async mode)
  - Average reward per epoch
- Optional Weights & Biases logging is available via `DEFAULT_HPARAMS["use_wandb"]`
  plus `wandb_project`/`wandb_group`.

## Reproducing Custom Runs

1. Edit `config.py` to include the scenarios and hyper-parameters you want.
2. (Optional) set `ACCEL_DEVICE=cuda:0` to run training on GPU.
3. Run `python main.py`. Logs appear under `logs/` and `results/`.
4. Use the plotting scripts above to compare modes or tune quantization triggers.

## Notes

- RLlib PPO is sensitive to learning-rate and batch size. The config includes
  a simple LR decay schedule (`lr_decay`) and manual device selection to keep
  comparisons fair.
- As of this implementation, we focus on CPU rollouts. Training can optionally
  run on GPU; compilation/quantization pipelines run on whichever device the
  backbone resides.

## Pruning Experiments

This repository includes comprehensive pruning experiments that combine **mask-based pruning** with **torch.compile** acceleration. Pruning reduces model size and inference cost while maintaining training quality.

### Quick Start

```bash
# Make sure wandb is installed in your environment
conda activate rllib-env  # or your target environment
pip install wandb

# Run basic pruning experiments
python scripts/run_pruning_experiments.py --experiment basic --seed 43 --wandb --wandb-project your-project-name
```

### Pruning Modes

The pruning system supports two distinct training architectures:

#### 1. Both Pruned Mode (`prune_training_model=True`)

- **Training model**: Pruned
- **Inference model**: Pruned (same masks applied)
- Both training and inference use the pruned structure
- More memory efficient but may impact learning capacity
- Fully on-policy: training and inference use identical pruned weights

### Configuration

All pruning hyperparameters are defined in `config_pruning.py`:

```python
DEFAULT_HPARAMS = {
    # Pruning settings
    "prune_ratio": 0.15,              # Target sparsity (15% of weights pruned)
    "prune_technique": "magnitude",    # Pruning strategy
    "prune_schedule": "iterative",     # Gradual pruning over time
    "prune_steps": 15,                 # Number of pruning steps
    "prune_training_model": True,      # Both Pruned vs Teacher-Student

    # Compression timing
    "trigger_every": 15,               # Prune every N epochs
    "min_epoch_before_compress": 30,   # Wait before first pruning

    # Model architecture
    "hidden_dim": 256,
    "hidden_depth": 4,

    # Wandb logging
    "use_wandb": True,
    "wandb_project": "rllib-accelerator",
    ...
}
```

### Available Experiments

Run different experiment types with the `--experiment` flag:

| Experiment Type | Description                                        | Command                   |
| --------------- | -------------------------------------------------- | ------------------------- |
| `basic`         | Compare baseline, compile, prune, prune+compile    | `--experiment basic`      |
| `ratios`        | Test different pruning ratios (0.1, 0.2, 0.3, 0.4) | `--experiment ratios`     |
| `strategies`    | Compare magnitude vs. random pruning               | `--experiment strategies` |
| `freq`          | Test different trigger frequencies (5, 10, 15, 20) | `--experiment freq`       |
| `sizes`         | Test across model sizes (256, 512, 1024, 2048)     | `--experiment sizes`      |

### Example Commands

```bash
# Basic comparison with wandb logging
python scripts/run_pruning_experiments.py \
    --experiment basic \
    --seed 43 \
    --wandb \
    --wandb-project rllib-accelerator

# Test different pruning ratios
python scripts/run_pruning_experiments.py \
    --experiment ratios \
    --epochs 150 \
    --seed 42

# Custom configuration
python scripts/run_pruning_experiments.py \
    --experiment basic \
    --epochs 200 \
    --hidden-dim 512 \
    --hidden-depth 6 \
    --prune-ratio 0.2
```

### Plotting Results

After running experiments, plot the results:

```bash
python scripts/plot_pruning_results.py --log-dir logs/pruning_basic
```

This generates comparison plots for:

- Reward curves across experiments
- Inference time comparison
- Throughput comparison
- Sparsity progression

### Implementation Details

- **Mask-based pruning**: Uses forward hooks to zero out pruned weights without changing model structure
- **Iterative pruning**: Gradually increases sparsity over multiple compression steps
- **Async architecture**: Pruning happens in background thread, doesn't block training
- **Compile integration**: Pruned models are torch.compile'd for additional speedup
- **Weight synchronization**: Training model weights are continuously synced to inference model
- **Mask broadcasting**: Pruning masks are efficiently broadcast to all rollout workers

## Teacher–Student Training Prototype

For experiments where a lightweight “student” policy interacts with the
environment while a much larger “teacher” policy trains on the same trajectories,
use the standalone script below (it does not touch the compile/quant pipeline):

```bash
python scripts/train_teacher_student.py \
    --env-id CartPole-v1 \
    --num-epochs 300 \
    --student-hidden-dim 64 --student-hidden-depth 2 \
    --teacher-hidden-dim 1024 --teacher-hidden-depth 6
```

Key CLI flags let you control student/teacher model sizes, learning rates,
rollout batch sizes, and the device the teacher trains on. Internally the script
uses RLlib PPO for the student (to generate data) and applies a PPO-style update
to the teacher backbone with the exact same advantages/returns, so both models
improve simultaneously without modifying the existing `main.py` workflow.
Each run writes per-epoch JSONL logs under `logs/teacher_student/`, matching the
format used by the main trainer for easy comparison.

To sweep over multiple student sizes automatically, run:

```bash
python scripts/run_teacher_student_grid.py \
    --student-hidden-dims 64 128 256 \
    --student-hidden-depths 2 4
```

The grid runner executes every dim/depth combination (sequentially, reusing
other hyper-parameters from `teacher_student/config.py` or CLI overrides),
logging each run under `logs/teacher_student/teacher_student_dim{N}_depth{M}_*.jsonl`.

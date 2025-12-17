"""
绘制剪枝实验结果

用法：
    python scripts/plot_pruning_results.py --log-dir logs/pruning_basic
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def load_jsonl(file_path: str) -> List[Dict]:
    """加载 JSONL 文件"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def aggregate_multiple_runs(log_dirs: Dict[str, str]) -> Dict[str, Dict]:
    """
    聚合多次运行的结果（处理带_run0, _run1等后缀的日志）
    
    Returns:
        Dict[method_name, Dict] where Dict contains:
            - 'mean': 平均值数据
            - 'std': 标准差数据
            - 'runs': 所有运行的原始数据列表
    """
    # 识别实验名称（去除_run后缀）
    experiment_groups = {}
    
    for label, log_dir in log_dirs.items():
        # 提取基础实验名（去除_run0, _run1等）
        if '_run' in label:
            base_name = label.rsplit('_run', 1)[0]
        else:
            base_name = label
        
        if base_name not in experiment_groups:
            experiment_groups[base_name] = []
        
        experiment_groups[base_name].append((label, log_dir))
    
    # 聚合每组实验
    aggregated = {}
    
    for base_name, runs in experiment_groups.items():
        if len(runs) == 1:
            # 单次运行，直接使用原始数据
            label, log_dir = runs[0]
            log_files = list(Path(log_dir).glob("*.jsonl"))
            if log_files:
                data = load_jsonl(log_files[0])
                aggregated[base_name] = {
                    'mean': data,
                    'std': None,
                    'runs': [data],
                    'num_runs': 1
                }
        else:
            # 多次运行，计算平均值和标准差
            all_runs_data = []
            
            for label, log_dir in runs:
                log_files = list(Path(log_dir).glob("*.jsonl"))
                if log_files:
                    data = load_jsonl(log_files[0])
                    all_runs_data.append(data)
            
            if not all_runs_data:
                continue
            
            # 确保所有运行有相同的epoch数
            min_epochs = min(len(run) for run in all_runs_data)
            
            # 计算每个epoch的平均值和标准差
            mean_data = []
            std_data = []
            
            for epoch_idx in range(min_epochs):
                # 收集所有运行在这个epoch的数据
                epoch_rewards = [run[epoch_idx]['reward_mean'] for run in all_runs_data]
                epoch_throughputs = [run[epoch_idx].get('throughput', 0) for run in all_runs_data]
                epoch_infer_times = [run[epoch_idx].get('inference_time', 0) for run in all_runs_data]
                epoch_sparsities = [run[epoch_idx].get('sparsity') for run in all_runs_data]
                
                mean_entry = {
                    'epoch': epoch_idx + 1,
                    'reward_mean': np.mean(epoch_rewards),
                    'throughput': np.mean(epoch_throughputs),
                    'inference_time': np.mean(epoch_infer_times),
                }
                
                std_entry = {
                    'epoch': epoch_idx + 1,
                    'reward_std': np.std(epoch_rewards),
                    'throughput_std': np.std(epoch_throughputs),
                    'inference_time_std': np.std(epoch_infer_times),
                }
                
                # 处理sparsity（可能为None）
                valid_sparsities = [s for s in epoch_sparsities if s is not None]
                if valid_sparsities:
                    mean_entry['sparsity'] = np.mean(valid_sparsities)
                    std_entry['sparsity_std'] = np.std(valid_sparsities)
                else:
                    mean_entry['sparsity'] = None
                    std_entry['sparsity_std'] = None
                
                mean_data.append(mean_entry)
                std_data.append(std_entry)
            
            aggregated[base_name] = {
                'mean': mean_data,
                'std': std_data,
                'runs': all_runs_data,
                'num_runs': len(all_runs_data)
            }
    
    return aggregated


def smooth_curve(values, weight=0.9):
    """指数移动平均平滑"""
    smoothed = []
    last = values[0]
    for v in values:
        smoothed_val = last * weight + (1 - weight) * v
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_reward_comparison(log_dirs: Dict[str, str], output_path: str = None):
    """绘制 reward 对比图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = cm.get_cmap('tab10')
    
    for idx, (label, log_dir) in enumerate(log_dirs.items()):
        # 找到第一个 jsonl 文件
        log_files = list(Path(log_dir).glob("*.jsonl"))
        if not log_files:
            print(f"⚠️ No log files found in {log_dir}")
            continue
        
        log_file = log_files[0]
        data = load_jsonl(log_file)
        
        epochs = [d['epoch'] for d in data]
        rewards = [d.get('reward_mean', 0) for d in data]
        
        # 绘制原始曲线（半透明）
        ax.plot(epochs, rewards, alpha=0.2, color=colors(idx))
        
        # 绘制平滑曲线
        if len(rewards) > 1:
            smoothed = smooth_curve(rewards, weight=0.9)
            ax.plot(epochs, smoothed, label=label, linewidth=2, color=colors(idx))
        else:
            ax.plot(epochs, rewards, label=label, linewidth=2, color=colors(idx))
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Average Reward', fontsize=12)
    ax.set_title('Training Reward Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved reward plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_throughput_comparison(log_dirs: Dict[str, str], output_path: str = None):
    """绘制吞吐量对比图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = []
    throughputs = []
    
    for label, log_dir in log_dirs.items():
        log_files = list(Path(log_dir).glob("*.jsonl"))
        if not log_files:
            continue
        
        log_file = log_files[0]
        data = load_jsonl(log_file)
        
        # 计算平均吞吐量（跳过前几个 epoch 的预热）
        skip_epochs = min(5, len(data) // 10)
        throughput_values = [d.get('throughput', 0) for d in data[skip_epochs:]]
        avg_throughput = np.mean(throughput_values)
        
        labels.append(label)
        throughputs.append(avg_throughput)
    
    # 绘制条形图
    x = np.arange(len(labels))
    bars = ax.bar(x, throughputs, color='steelblue', alpha=0.7)
    
    # 添加数值标签
    for bar, val in zip(bars, throughputs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Throughput (samples/s)', fontsize=12)
    ax.set_title('Average Throughput Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved throughput plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_inference_time_comparison(log_dirs: Dict[str, str], output_path: str = None):
    """绘制推理时间对比图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = []
    inference_times = []
    
    for label, log_dir in log_dirs.items():
        log_files = list(Path(log_dir).glob("*.jsonl"))
        if not log_files:
            continue
        
        log_file = log_files[0]
        data = load_jsonl(log_file)
        
        # 计算平均推理时间
        skip_epochs = min(5, len(data) // 10)
        infer_values = [d.get('inference_time', 0) for d in data[skip_epochs:]]
        avg_infer = np.mean(infer_values)
        
        labels.append(label)
        inference_times.append(avg_infer * 1000)  # 转换为毫秒
    
    # 绘制条形图
    x = np.arange(len(labels))
    bars = ax.bar(x, inference_times, color='coral', alpha=0.7)
    
    # 添加数值标签
    for bar, val in zip(bars, inference_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}ms',
                ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Inference Time (ms)', fontsize=12)
    ax.set_title('Average Inference Time Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved inference time plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_compression_ratio(log_dirs: Dict[str, str], output_path: str = None):
    """绘制压缩率变化图（仅适用于剪枝实验）"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = cm.get_cmap('tab10')
    
    for idx, (label, log_dir) in enumerate(log_dirs.items()):
        if 'prune' not in label.lower():
            continue  # 只处理剪枝相关的实验
        
        log_files = list(Path(log_dir).glob("*.jsonl"))
        if not log_files:
            continue
        
        log_file = log_files[0]
        data = load_jsonl(log_file)
        
        # 提取剪枝信息
        epochs = []
        ratios = []
        
        for d in data:
            # 尝试从日志中提取压缩率信息
            # 注意：这需要在压缩时记录到日志中
            epoch = d.get('epoch')
            # 如果有 compression_ratio 字段
            ratio = d.get('compression_ratio')
            if ratio is not None:
                epochs.append(epoch)
                ratios.append(ratio)
        
        if epochs:
            ax.plot(epochs, ratios, label=label, linewidth=2, 
                   color=colors(idx), marker='o', markersize=4)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Compression Ratio (remaining neurons)', fontsize=12)
    ax.set_title('Model Compression Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved compression ratio plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_pruning_ratio_comparison(log_dirs: Dict[str, str], output_path: str = None):
    """绘制不同pruning ratio的对比图（柱状图）"""
    # 提取ratio实验的数据
    ratio_data = {}
    baseline_data = None
    
    for label, log_dir in log_dirs.items():
        log_files = list(Path(log_dir).glob("*.jsonl"))
        if not log_files:
            continue
        
        log_file = log_files[0]
        data = load_jsonl(log_file)
        skip = min(5, len(data) // 10)
        
        # 计算指标
        rewards = [d.get('reward_mean', 0) for d in data[-50:]]
        throughputs = [d.get('throughput', 0) for d in data[skip:]]
        sparsities = [d.get('sparsity', 0) for d in data if d.get('sparsity') is not None]
        
        avg_reward = np.mean(rewards)
        avg_throughput = np.mean(throughputs)
        avg_sparsity = np.mean(sparsities) if sparsities else 0
        
        if 'baseline' in label.lower():
            baseline_data = {
                'reward': avg_reward,
                'throughput': avg_throughput,
                'sparsity': 0
            }
        elif 'ratio=' in label:
            # 提取ratio值
            try:
                ratio_str = label.split('ratio=')[1].split('_')[0]
                ratio = float(ratio_str)
                ratio_data[ratio] = {
                    'reward': avg_reward,
                    'throughput': avg_throughput,
                    'sparsity': avg_sparsity
                }
            except:
                pass
    
    if not ratio_data:
        print("⚠️ No pruning ratio experiments found")
        return
    
    # 排序
    ratios = sorted(ratio_data.keys())
    rewards = [ratio_data[r]['reward'] for r in ratios]
    throughputs = [ratio_data[r]['throughput'] for r in ratios]
    sparsities = [ratio_data[r]['sparsity'] * 100 for r in ratios]  # 转换为百分比
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Pruning Ratio Comparison', fontsize=16, fontweight='bold')
    
    # 1. Avg Reward vs Pruning Ratio
    ax = axes[0, 0]
    bars = ax.bar([f'{r:.1f}' for r in ratios], rewards, color='steelblue', alpha=0.7, edgecolor='black')
    if baseline_data:
        ax.axhline(y=baseline_data['reward'], color='red', linestyle='--', 
                   linewidth=2, label=f"Baseline ({baseline_data['reward']:.1f})")
        ax.legend()
    ax.set_xlabel('Pruning Ratio', fontsize=12)
    ax.set_ylabel('Average Reward (last 50 epochs)', fontsize=12)
    ax.set_title('Performance vs Pruning Ratio', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, reward in zip(bars, rewards):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{reward:.1f}', ha='center', va='bottom', fontsize=10)
    
    # 2. Throughput vs Pruning Ratio
    ax = axes[0, 1]
    bars = ax.bar([f'{r:.1f}' for r in ratios], throughputs, color='coral', alpha=0.7, edgecolor='black')
    if baseline_data:
        ax.axhline(y=baseline_data['throughput'], color='red', linestyle='--', 
                   linewidth=2, label=f"Baseline ({baseline_data['throughput']:.0f})")
        ax.legend()
    ax.set_xlabel('Pruning Ratio', fontsize=12)
    ax.set_ylabel('Throughput (samples/s)', fontsize=12)
    ax.set_title('Throughput vs Pruning Ratio', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Sparsity vs Pruning Ratio
    ax = axes[1, 0]
    bars = ax.bar([f'{r:.1f}' for r in ratios], sparsities, color='mediumseagreen', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Pruning Ratio', fontsize=12)
    ax.set_ylabel('Actual Sparsity (%)', fontsize=12)
    ax.set_title('Achieved Sparsity', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, sparsity in zip(bars, sparsities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{sparsity:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # 4. Reward vs Sparsity Trade-off
    ax = axes[1, 1]
    ax.scatter(sparsities, rewards, s=200, c=ratios, cmap='viridis', 
               alpha=0.7, edgecolors='black', linewidth=2)
    if baseline_data:
        ax.scatter([0], [baseline_data['reward']], s=200, c='red', 
                   marker='*', edgecolors='black', linewidth=2, label='Baseline', zorder=5)
    
    # 添加标签
    for ratio, sparsity, reward in zip(ratios, sparsities, rewards):
        ax.annotate(f'ratio={ratio:.1f}', 
                   (sparsity, reward), 
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, alpha=0.8)
    
    ax.set_xlabel('Sparsity (%)', fontsize=12)
    ax.set_ylabel('Average Reward', fontsize=12)
    ax.set_title('Performance vs Sparsity Trade-off', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if baseline_data:
        ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved pruning ratio comparison to {output_path}")
    else:
        plt.show()
    
    plt.close()


def print_summary_table(log_dirs: Dict[str, str]):
    """打印性能对比表格"""
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print(f"{'Method':<30} {'Avg Reward':<15} {'Throughput':<15} {'Inference Time':<15}")
    print("-"*80)
    
    for label, log_dir in log_dirs.items():
        log_files = list(Path(log_dir).glob("*.jsonl"))
        if not log_files:
            continue
        
        log_file = log_files[0]
        data = load_jsonl(log_file)
        
        skip = min(5, len(data) // 10)
        
        # 计算平均指标
        rewards = [d.get('reward_mean', 0) for d in data[-50:]]  # 最后 50 个 epoch
        throughputs = [d.get('throughput', 0) for d in data[skip:]]
        infer_times = [d.get('inference_time', 0) for d in data[skip:]]
        
        avg_reward = np.mean(rewards)
        avg_throughput = np.mean(throughputs)
        avg_infer = np.mean(infer_times) * 1000  # 转换为毫秒
        
        print(f"{label:<30} {avg_reward:<15.2f} {avg_throughput:<15.1f} {avg_infer:<15.2f}ms")
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Plot pruning experiment results")
    parser.add_argument(
        "--log-dir",
        type=str,
        required=True,
        help="Directory containing experiment logs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots (default: same as log-dir)"
    )
    
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"❌ Log directory not found: {log_dir}")
        return
    
    # 自动发现所有实验子目录
    log_dirs = {}
    for subdir in sorted(log_dir.iterdir()):
        if subdir.is_dir():
            # 使用目录名作为标签
            label = subdir.name
            log_dirs[label] = str(subdir)
    
    if not log_dirs:
        print(f"❌ No experiment subdirectories found in {log_dir}")
        return
    
    print(f"Found {len(log_dirs)} experiments:")
    for label in log_dirs.keys():
        print(f"  - {label}")
    
    # 设置输出目录
    output_dir = Path(args.output_dir) if args.output_dir else log_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检测实验类型
    is_ratio_experiment = any('ratio=' in label for label in log_dirs.keys())
    
    # 生成所有图表
    print("\nGenerating plots...")
    plot_reward_comparison(log_dirs, str(output_dir / "reward_comparison.png"))
    plot_throughput_comparison(log_dirs, str(output_dir / "throughput_comparison.png"))
    plot_inference_time_comparison(log_dirs, str(output_dir / "inference_time_comparison.png"))
    
    # 如果是pruning ratio实验，生成专门的对比图
    if is_ratio_experiment:
        print("\n📊 Detected pruning ratio experiment, generating ratio comparison plots...")
        plot_pruning_ratio_comparison(log_dirs, str(output_dir / "pruning_ratio_comparison.png"))
    
    # 打印性能对比表
    print_summary_table(log_dirs)
    
    print(f"\n✅ All plots saved to: {output_dir}")


if __name__ == "__main__":
    main()


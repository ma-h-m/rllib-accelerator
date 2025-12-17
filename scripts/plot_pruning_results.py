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
    
    # 生成所有图表
    print("\nGenerating plots...")
    plot_reward_comparison(log_dirs, str(output_dir / "reward_comparison.png"))
    plot_throughput_comparison(log_dirs, str(output_dir / "throughput_comparison.png"))
    plot_inference_time_comparison(log_dirs, str(output_dir / "inference_time_comparison.png"))
    
    # 打印性能对比表
    print_summary_table(log_dirs)
    
    print(f"\n✅ All plots saved to: {output_dir}")


if __name__ == "__main__":
    main()


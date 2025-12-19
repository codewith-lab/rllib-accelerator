"""
Plot pruning experiment results.

Usage:
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
    """Load a JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def smooth_curve(values, weight=0.9):
    """Apply exponential moving average smoothing."""
    smoothed = []
    last = values[0]
    for v in values:
        smoothed_val = last * weight + (1 - weight) * v
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_reward_comparison(log_dirs: Dict[str, str], output_path: str = None):
    """Plot reward comparison chart."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = cm.get_cmap('tab10')
    
    for idx, (label, log_dir) in enumerate(log_dirs.items()):
        # Find the first jsonl file
        log_files = list(Path(log_dir).glob("*.jsonl"))
        if not log_files:
            print(f"⚠️ No log files found in {log_dir}")
            continue
        
        log_file = log_files[0]
        data = load_jsonl(log_file)
        
        epochs = [d['epoch'] for d in data]
        rewards = [d.get('reward_mean', 0) for d in data]
        
        # Plot the raw curve (semi-transparent)
        ax.plot(epochs, rewards, alpha=0.2, color=colors(idx))
        
        # Plot the smoothed curve
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
    """Plot throughput comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = []
    throughputs = []
    
    for label, log_dir in log_dirs.items():
        log_files = list(Path(log_dir).glob("*.jsonl"))
        if not log_files:
            continue
        
        log_file = log_files[0]
        data = load_jsonl(log_file)
        
        # Compute average throughput (skip early warm-up epochs)
        skip_epochs = min(5, len(data) // 10)
        throughput_values = [d.get('throughput', 0) for d in data[skip_epochs:]]
        avg_throughput = np.mean(throughput_values)
        
        labels.append(label)
        throughputs.append(avg_throughput)
    
    # Plot bar chart
    x = np.arange(len(labels))
    bars = ax.bar(x, throughputs, color='steelblue', alpha=0.7)
    
    # Add value labels
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
    """Plot inference time comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = []
    inference_times = []
    
    for label, log_dir in log_dirs.items():
        log_files = list(Path(log_dir).glob("*.jsonl"))
        if not log_files:
            continue
        
        log_file = log_files[0]
        data = load_jsonl(log_file)
        
        # Compute average inference time
        skip_epochs = min(5, len(data) // 10)
        infer_values = [d.get('inference_time', 0) for d in data[skip_epochs:]]
        avg_infer = np.mean(infer_values)
        
        labels.append(label)
        inference_times.append(avg_infer * 1000)  # Convert to milliseconds
    
    # Plot bar chart
    x = np.arange(len(labels))
    bars = ax.bar(x, inference_times, color='coral', alpha=0.7)
    
    # Add value labels
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
    """Plot compression ratio over time (pruning experiments only)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = cm.get_cmap('tab10')
    
    for idx, (label, log_dir) in enumerate(log_dirs.items()):
        if 'prune' not in label.lower():
            continue  # Only handle pruning-related experiments
        
        log_files = list(Path(log_dir).glob("*.jsonl"))
        if not log_files:
            continue
        
        log_file = log_files[0]
        data = load_jsonl(log_file)
        
        # Extract pruning info
        epochs = []
        ratios = []
        
        for d in data:
            # Try to extract compression ratio from logs
            # Note: requires logging compression_ratio during compression
            epoch = d.get('epoch')
            # If compression_ratio exists
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
    """Print performance comparison table."""
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
        
        # Compute average metrics
        rewards = [d.get('reward_mean', 0) for d in data[-50:]]  # Last 50 epochs
        throughputs = [d.get('throughput', 0) for d in data[skip:]]
        infer_times = [d.get('inference_time', 0) for d in data[skip:]]
        
        avg_reward = np.mean(rewards)
        avg_throughput = np.mean(throughputs)
        avg_infer = np.mean(infer_times) * 1000  # Convert to milliseconds
        
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
    
    # Auto-discover all experiment subdirectories
    log_dirs = {}
    for subdir in sorted(log_dir.iterdir()):
        if subdir.is_dir():
            # Use directory name as label
            label = subdir.name
            log_dirs[label] = str(subdir)
    
    if not log_dirs:
        print(f"❌ No experiment subdirectories found in {log_dir}")
        return
    
    print(f"Found {len(log_dirs)} experiments:")
    for label in log_dirs.keys():
        print(f"  - {label}")
    
    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else log_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all plots
    print("\nGenerating plots...")
    plot_reward_comparison(log_dirs, str(output_dir / "reward_comparison.png"))
    plot_throughput_comparison(log_dirs, str(output_dir / "throughput_comparison.png"))
    plot_inference_time_comparison(log_dirs, str(output_dir / "inference_time_comparison.png"))
    
    # Print performance comparison table
    print_summary_table(log_dirs)
    
    print(f"\n✅ All plots saved to: {output_dir}")


if __name__ == "__main__":
    main()

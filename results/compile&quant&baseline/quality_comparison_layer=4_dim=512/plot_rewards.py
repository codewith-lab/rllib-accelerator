#!/usr/bin/env python3
"""
Plot reward curves for compile & quant comparison experiments.

Usage:
    python plot_rewards.py \
        --file baseline=baseline_....jsonl \
        --file async_compile=async_compile_....jsonl \
        --file async_quant=async_quant_....jsonl \
        --file async_quant2=async_quant2_....jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot reward curves.")
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory containing jsonl result files.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=[],
        help="Curve labels. If --file not provided, use prefix search.",
    )
    parser.add_argument(
        "--file",
        action="append",
        nargs=2,
        metavar=("LABEL", "FILENAME"),
        help="Explicit mapping from label to filename.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reward_compare.png",
        help="Output image name saved under --dir.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable plt.show() (useful for headless runs).",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Moving average window size for smoothing reward curves.",
    )
    return parser.parse_args()


def parse_file_mapping(pairs):
    mapping = {}
    if not pairs:
        return mapping
    for label, filename in pairs:
        mapping[label.strip()] = filename.strip()
    return mapping


def find_latest_file(directory: Path, prefix: str) -> Path:
    exact = directory / prefix
    if exact.exists():
        return exact
    files = sorted(directory.glob(f"{prefix}_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No file for prefix '{prefix}' in {directory}")
    return files[-1]


def load_rewards(path: Path) -> Dict[str, List[float]]:
    epochs, rewards = [], []
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            epochs.append(rec.get("epoch"))
            rewards.append(rec.get("reward_mean"))
    return {"epoch": epochs, "reward": rewards}


def smooth_series(values: List[float], window: int) -> List[float]:
    if window <= 1 or window > len(values):
        return values
    smoothed = []
    half = window // 2
    for i in range(len(values)):
        start = max(0, i - half)
        end = min(len(values), i - half + window)
        segment = values[start:end]
        smoothed.append(sum(segment) / len(segment))
    return smoothed


def main():
    args = parse_args()
    file_map = parse_file_mapping(args.file)
    if args.labels:
        labels = args.labels
    elif file_map:
        labels = list(file_map.keys())
    else:
        labels = ["baseline", "async_compile", "async_quant", "async_quant2"]
    data: Dict[str, Dict[str, List[float]]] = {}

    for label in labels:
        if label in file_map:
            path = args.dir / file_map[label]
            if not path.exists():
                raise FileNotFoundError(f"{path} not found for label {label}")
        else:
            path = find_latest_file(args.dir, label)
        data[label] = load_rewards(path)
        print(f"[plot_rewards] Loaded {path.name} for label '{label}'.")

    plt.figure(figsize=(10, 6))
    for label, rec in data.items():
        rewards = smooth_series(rec["reward"], args.smooth)
        plt.plot(rec["epoch"], rewards, label=label, linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Reward Mean")
    plt.title("Reward Comparison")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    out_path = args.dir / args.output
    plt.savefig(out_path, dpi=200)
    print(f"[plot_rewards] Saved to {out_path}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()

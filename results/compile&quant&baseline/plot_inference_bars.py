#!/usr/bin/env python3
"""
Plot bar charts for rollout/train/inference time across compile&quant experiments.

Example:
    python plot_inference_bars.py \
        --experiments inference_speed_comparison_layer=8_dim=512 \
                     inference_speed_comparison_layer=8_dim=1024 \
                     inference_speed_comparison_layer=8_dim=2048 \
        --labels baseline async_compile async_quant \
        --output-prefix bars_layer8
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_PATTERNS = {
    "baseline": ["none.jsonl", "baseline.jsonl", "none_*.jsonl"],
    "async_compile": ["async_compile.jsonl", "async.jsonl", "async_compile_*.jsonl"],
    "async_quant": ["async_quant.jsonl", "async_quant_freq=*.jsonl"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot bar charts comparing inference speed.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Base directory containing experiment subfolders.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help="Relative subdirectories to include (e.g. inference_speed_comparison_layer=8_dim=512).",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["baseline", "async_compile", "async_quant"],
        help="Experiment labels to plot.",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        metavar=("LABEL", "FILENAME"),
        help="Optional explicit mapping label filename (applied to all directories).",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="inference_bars",
        help="Prefix for output images saved under base-dir.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable plt.show() for headless environments.",
    )
    return parser.parse_args()


def parse_file_mapping(items):
    if not items:
        return {}
    if len(items) % 2 != 0:
        raise ValueError("--files expects LABEL FILENAME pairs.")
    mapping = {}
    for i in range(0, len(items), 2):
        mapping[items[i]] = items[i + 1]
    return mapping


def resolve_file(directory: Path, label: str, file_map: Dict[str, str]) -> Path:
    if label in file_map:
        path = directory / file_map[label]
        if not path.exists():
            raise FileNotFoundError(f"{path} not found for label '{label}'.")
        return path

    patterns = DEFAULT_PATTERNS.get(label, [f"{label}.jsonl", f"{label}_*.jsonl"])
    for pattern in patterns:
        if "*" in pattern:
            matches = sorted(directory.glob(pattern))
            if matches:
                return matches[-1]
        else:
            candidate = directory / pattern
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"No matching file for label '{label}' in {directory}")


def aggregate_times(path: Path) -> Dict[str, float]:
    buckets: Dict[str, List[float]] = {
        "rollout_time": [],
        "train_time": [],
        "inference_time": [],
        "throughput": [],
    }
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            for key in buckets:
                val = rec.get(key)
                if val is not None:
                    buckets[key].append(float(val))
    return {k: float(np.mean(v)) if v else 0.0 for k, v in buckets.items()}


def main():
    args = parse_args()
    file_map = parse_file_mapping(args.files)
    metrics = ["rollout_time", "inference_time", "throughput"]

    # Aggregate data
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for exp in args.experiments:
        directory = args.base_dir / exp
        if not directory.exists():
            raise FileNotFoundError(f"Experiment directory {directory} not found.")
        label_data = {}
        for label in args.labels:
            path = resolve_file(directory, label, file_map)
            label_data[label] = aggregate_times(path)
            print(f"[plot_inference_bars] {exp} | {label} -> {path.name}")
        results[exp] = label_data

    x = np.arange(len(args.experiments))
    bar_width = 0.8 / max(1, len(args.labels))

    def format_label(name: str) -> str:
        if "dim=" in name:
            return name.split("dim=")[-1]
        return name

    formatted_labels = [format_label(exp) for exp in args.experiments]

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8, 4))
        for idx, label in enumerate(args.labels):
            values = [
                results[exp][label].get(metric, 0.0)
                for exp in args.experiments
            ]
            offsets = x - 0.4 + bar_width / 2 + idx * bar_width
            ax.bar(offsets, values, width=bar_width, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels(formatted_labels, rotation=0)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.replace('_', ' ').title()} Comparison")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        out_path = args.base_dir / f"{args.output_prefix}_{metric}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        print(f"[plot_inference_bars] Saved {out_path}")
        if not args.no_show:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Plot timing metrics (total/rollout/train/inference/compile/swap) for
baseline vs. compile variants under results/none&sync&async.

Usage:
    python plot_times.py --dir results/none&sync&async
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot RL timing metrics.")
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory containing *.jsonl result files.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["none", "sync", "async", "async_with_warmup"],
        help="File prefixes (ignoring timestamps) to include.",
    )
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        metavar="LABEL=FILENAME",
        help="Explicit mapping from label to file name (e.g. none=none_20251203_204904.jsonl).",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="timing",
        help="Prefix for saved plot files. Images are saved into --dir.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="If set, do not open the matplotlib window (useful for headless runs).",
    )
    return parser.parse_args()


def parse_file_mapping(pairs):
    mapping = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Invalid --file entry '{item}', expected LABEL=FILENAME.")
        label, filename = item.split("=", 1)
        mapping[label.strip()] = filename.strip()
    return mapping


def find_latest_file(directory: Path, prefix: str) -> Path:
    exact = directory / prefix
    if exact.exists():
        return exact
    files = sorted(directory.glob(f"{prefix}_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No files found for prefix '{prefix}' in {directory}")
    return files[-1]


def load_metrics(path: Path) -> Dict[str, List[float]]:
    metrics = {
        "epoch": [],
        "total_time": [],
        "rollout_time": [],
        "train_time": [],
        "inference_time": [],
        "compile_latency": [],
        "swap_latency": [],
    }
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            metrics["epoch"].append(rec.get("epoch"))
            for key in metrics.keys():
                if key == "epoch":
                    continue
                metrics[key].append(rec.get(key))
    return metrics


def plot_metric(ax, label_data: Dict[str, Dict[str, List[float]]], metric: str):
    for label, data in label_data.items():
        ax.plot(data["epoch"], data[metric], marker="o", linewidth=1.5, label=label)
    ax.set_title(metric.replace("_", " ").title())
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)


def main():
    args = parse_args()
    explicit = parse_file_mapping(args.file)
    label_data: Dict[str, Dict[str, List[float]]] = {}

    for label in args.labels:
        if label in explicit:
            path = args.dir / explicit[label]
            if not path.exists():
                raise FileNotFoundError(f"Explicit file '{explicit[label]}' for '{label}' not found.")
        else:
            path = find_latest_file(args.dir, label)
        label_data[label] = load_metrics(path)
        print(f"[plot_times] Loaded {path.name} for label '{label}'.")

    metrics_to_plot = ["total_time", "rollout_time", "train_time", "inference_time"]
    compile_metrics = ["compile_latency", "swap_latency"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric in zip(axes.flat, metrics_to_plot):
        plot_metric(ax, label_data, metric)

    fig.tight_layout()
    main_path = args.dir / f"{args.output_prefix}_main.png"
    fig.savefig(main_path, dpi=200)
    print(f"[plot_times] Saved main plot to {main_path}")

    if any(any(label_data[label][m]) for label in label_data for m in compile_metrics):
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
        for ax, metric in zip(axes2, compile_metrics):
            plot_metric(ax, label_data, metric)
        fig2.tight_layout()
        compile_path = args.dir / f"{args.output_prefix}_compile.png"
        fig2.savefig(compile_path, dpi=200)
        print(f"[plot_times] Saved compile plot to {compile_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()

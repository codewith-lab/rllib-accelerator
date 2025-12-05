#!/usr/bin/env python3
"""
Plot bar chart for swap latency measurements stored in swap_time_rec.csv.
Each line of the CSV represents one experiment (baseline/compile/quant order configurable).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Plot swap latency bar chart.")
    parser.add_argument(
        "--file",
        type=Path,
        default=Path(__file__).parent / "swap_time_rec.csv",
        help="CSV file containing one latency value per line.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["dim=512", "dim=1024", "dim=2048"],
        help="Labels for each line in the CSV.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Swap Latency (Quant Models)",
        help="Chart title.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "swap_latency.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable plt.show().",
    )
    return parser.parse_args()


def load_latencies(path: Path):
    latencies = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            latencies.append(float(line))
    return latencies


def main():
    args = parse_args()
    latencies = load_latencies(args.file)
    if len(latencies) != len(args.labels):
        raise ValueError("Number of labels must match number of entries in CSV.")

    x = np.arange(len(latencies))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x, latencies, color="teal", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(args.labels)
    ax.set_ylabel("Swap Latency (s)")
    ax.set_title(args.title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    print(f"[plot_swap_latency] Saved to {args.output}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()

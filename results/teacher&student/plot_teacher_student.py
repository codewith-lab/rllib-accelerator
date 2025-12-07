import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(path: Path):
    epochs = []
    student_rewards = []
    teacher_rewards = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            epochs.append(rec.get("epoch"))
            student_rewards.append(rec.get("reward_mean"))
            teacher_rewards.append(rec.get("teacher_eval_reward"))
    return epochs, student_rewards, teacher_rewards


def smooth_series(values, window):
    if window <= 1 or not values:
        return values
    smoothed = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        window_vals = [v for v in values[start : idx + 1] if v is not None]
        if window_vals:
            smoothed.append(sum(window_vals) / len(window_vals))
        else:
            smoothed.append(None)
    return smoothed


def band_series(values, window):
    if window <= 1 or not values:
        return values, values
    mins = []
    maxs = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        window_vals = [v for v in values[start : idx + 1] if v is not None]
        if window_vals:
            mins.append(min(window_vals))
            maxs.append(max(window_vals))
        else:
            mins.append(None)
            maxs.append(None)
    return mins, maxs


def plot_with_band(epochs, series, label, color, window):
    smoothed = smooth_series(series, window)
    mins, maxs = band_series(series, window)
    plt.plot(epochs, smoothed, label=label, color=color, linewidth=1.5)
    band_epochs = [e for e, lo, hi in zip(epochs, mins, maxs) if lo is not None and hi is not None]
    band_lows = [lo for lo in mins if lo is not None]
    band_highs = [hi for hi in maxs if hi is not None]
    if band_epochs:
        plt.fill_between(
            band_epochs[: len(band_lows)],
            band_lows,
            band_highs[: len(band_lows)],
            color=color,
            alpha=0.15,
            linewidth=0,
        )


def plot_file(path: Path, title: str | None = None, smooth_window: int = 1):
    epochs, student, teacher = load_metrics(path)
    plt.figure(figsize=(10, 5))
    # reward_mean corresponds to the large learner (teacher) target
    plot_with_band(epochs, student, "Teacher Reward", "#1f77b4", smooth_window)
    if any(t is not None for t in teacher):
        plot_with_band(epochs, teacher, "Student Eval Reward", "#ff7f0e", smooth_window)
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.title(title or path.stem)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def main():
    parser = argparse.ArgumentParser(description="Plot student vs teacher rewards from a JSONL log.")
    parser.add_argument("jsonl_path", type=str, help="Path to teacher-student JSONL log.")
    parser.add_argument("--title", type=str, default=None, help="Optional plot title.")
    parser.add_argument("--show", action="store_true", help="Display plot interactively.")
    parser.add_argument("--output", type=str, default=None, help="Save plot to this file (png/pdf).")
    parser.add_argument("--smooth", type=int, default=1, help="Moving-average window for smoothing.")
    args = parser.parse_args()

    path = Path(args.jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}")

    plot_file(path, args.title, smooth_window=max(1, args.smooth))

    output_path = None
    if args.output:
        user_path = Path(args.output)
        output_path = user_path if user_path.is_absolute() else path.parent / user_path
        plt.savefig(output_path, dpi=200)
        print(f"Saved plot to {output_path}")

    if args.show or output_path is None:
        plt.show()


if __name__ == "__main__":
    main()

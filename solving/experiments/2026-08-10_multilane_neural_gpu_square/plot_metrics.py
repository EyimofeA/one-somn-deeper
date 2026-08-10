"""Plot train and unseen-x learning curves from the capability run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.metrics.read_text().splitlines()]
    steps = [row["step"] for row in rows]

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    colors = {"train": "#2457A6", "unseen_x": "#D17724"}
    styles = {"train": "-", "unseen_x": "--"}
    labels = {"train": "Train x", "unseen_x": "Unseen x"}

    for split in ("train", "unseen_x"):
        axes[0].plot(
            steps,
            [100 * row[split]["exact_accuracy"] for row in rows],
            color=colors[split],
            linestyle=styles[split],
            linewidth=2.2,
            label=labels[split],
        )
        axes[1].plot(
            steps,
            [100 * row[split]["digit_accuracy"] for row in rows],
            color=colors[split],
            linestyle=styles[split],
            linewidth=2.2,
            label=labels[split],
        )

    axes[0].axhline(90, color="#3E4651", linewidth=1.2, linestyle=":", label="90% pass gate")
    axes[0].set_title("Whole-square exact accuracy")
    axes[1].set_title("Per-digit accuracy")
    for axis in axes:
        axis.set_xlabel("Optimizer updates")
        axis.set_ylabel("Accuracy (%)")
        axis.set_xlim(0, 12000)
        axis.set_ylim(0, 100)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="upper left")
    axes[1].legend(frameon=False, loc="lower right")
    figure.suptitle("Multi-lane Neural GPU raw-square capability", fontsize=14, fontweight="bold")
    figure.text(
        0.5,
        -0.01,
        "Seed 0; 8,000 training x and 2,000 disjoint unseen x; final-label supervision",
        ha="center",
        fontsize=9,
        color="#545B66",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()

"""Plot the matched 50k answer-only and carry-supervised curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("answer_only", type=Path)
    parser.add_argument("carry_aux", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    conditions = {
        "Answer only": read(args.answer_only),
        "Carry supervised": read(args.carry_aux),
    }
    colors = {"Answer only": "#2457A6", "Carry supervised": "#D17724"}
    styles = {"Answer only": "-", "Carry supervised": "--"}

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for label, rows in conditions.items():
        steps = [row["step"] for row in rows]
        axes[0].plot(
            steps,
            [100 * row["unseen_x"]["exact_accuracy"] for row in rows],
            color=colors[label],
            linestyle=styles[label],
            linewidth=2.2,
            label=label,
        )
        axes[1].plot(
            steps,
            [100 * row["train"]["exact_accuracy"] for row in rows],
            color=colors[label],
            linestyle=styles[label],
            linewidth=2.2,
            label=label,
        )

    axes[0].axhline(20, color="#3E4651", linestyle=":", linewidth=1.2, label="20% kill gate")
    axes[0].set_title("Unseen-x exact accuracy")
    axes[0].set_ylim(0, 20)
    axes[1].set_title("Training exact accuracy")
    axes[1].set_ylim(0, 100)
    for axis in axes:
        axis.set_xlim(0, 50000)
        axis.set_xlabel("Optimizer updates")
        axis.set_ylabel("Exact accuracy (%)")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="upper left")
    axes[1].legend(frameon=False, loc="upper left")
    figure.suptitle("Neural GPU carry-supervision duration test", fontsize=14, fontweight="bold")
    figure.text(
        0.5,
        -0.01,
        "Seed 0; same 8,000 train / 2,000 unseen x; only carry head and loss differ",
        ha="center",
        fontsize=9,
        color="#545B66",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()

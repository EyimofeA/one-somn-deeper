"""Compare first-rung evidence used for the 2026-08-08 Hard selection."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "solving/experiments/figures/t1_hard_selection_2026-08-08.png"

ROWS = [
    ("Full register · local s74", 5, 2, False),
    ("Full register · hosted", 3, 0, False),
    ("Compact · local s74", 11, 1, False),
    ("Compact · hosted A", 2, 3, True),
    ("Compact · hosted B", 5, 4, True),
    ("Width 128 · local", 6, 1, False),
    ("Width 128 · hosted", 2, 0, False),
    ("Batch 256 · local", 2, 0, False),
    ("LR 6e-3 · local", 0, 2, False),
]


def main() -> None:
    labels = [row[0] for row in ROWS][::-1]
    seen_counts = np.array([row[1] for row in ROWS][::-1])
    ood_counts = np.array([row[2] for row in ROWS][::-1])
    selected = np.array([row[3] for row in ROWS][::-1])
    y = np.arange(len(labels))
    colors = np.where(selected, "#276FBF", "#A7B0BC")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.4), sharey=True)
    fig.patch.set_facecolor("#FAFBFC")
    panels = [
        (axes[0], seen_counts, "Seen-N T=1", 6 / 768 * 100, "Hard leader: 6/768"),
        (axes[1], ood_counts, "OOD-N T=1", 3 / 768 * 100, "Hard leader: 3/768"),
    ]
    for axis, counts, title, leader, leader_label in panels:
        values = counts / 512 * 100
        bars = axis.barh(y, values, color=colors, edgecolor="#53606F", height=0.66)
        axis.axvline(leader, color="#C47A23", linestyle="--", linewidth=1.8)
        axis.text(
            leader,
            len(labels) - 0.15,
            leader_label,
            color="#8A5115",
            fontsize=8,
            ha="left",
            va="top",
        )
        for bar, count in zip(bars, counts):
            axis.text(
                bar.get_width() + 0.025,
                bar.get_y() + bar.get_height() / 2,
                f"{count}/512",
                va="center",
                fontsize=8,
                color="#202631",
            )
        axis.set_title(title, loc="left", weight="bold")
        axis.set_xlabel("Exact accuracy (%)")
        axis.set_xlim(0, max(2.35, values.max() + 0.3))
        axis.grid(axis="x", color="#DDE2E8", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)

    axes[0].set_yticks(y, labels, fontsize=9)
    for tick, is_selected in zip(axes[0].get_yticklabels(), selected):
        if is_selected:
            tick.set_weight("bold")
            tick.set_color("#174A7E")

    fig.suptitle(
        "T=1 Hard candidate selection",
        x=0.08,
        y=0.985,
        ha="left",
        fontsize=16,
        weight="bold",
        color="#202631",
    )
    fig.text(
        0.08,
        0.935,
        "Public e5 profiles; blue rows are the two exact hosted runs of frozen SHA 5b622f06. Hard leader lines use 768-example profiles.",
        fontsize=9,
        color="#5B6573",
    )
    fig.text(
        0.98,
        0.985,
        "✦ ONE LAYER DEEPER RESEARCH",
        ha="right",
        va="top",
        fontsize=8,
        color="#276FBF",
        weight="bold",
    )
    fig.text(
        0.08,
        0.02,
        "Source: evaluator-owned public e5 profiles and live Hard leaderboard read on 2026-08-08. No candidate certified T=1.",
        fontsize=8.5,
        color="#5B6573",
    )
    fig.tight_layout(rect=(0.06, 0.06, 0.99, 0.90), w_pad=2.2)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(OUTPUT)


if __name__ == "__main__":
    main()

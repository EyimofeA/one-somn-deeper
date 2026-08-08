"""Plot seed-level T=1 final-label diagnostics from raw JSON reports."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARMS = {
    "Factored\nN hidden in square": (
        ROOT / "diagnostics/artifacts/t1_phase_square_reduce_2026-08-08",
        "factored_seed{}",
        "#3568A8",
    ),
    "Entangled\nN visible in square": (
        ROOT / "diagnostics/artifacts/t1_phase_square_reduce_2026-08-08",
        "entangled_seed{}",
        "#D97724",
    ),
    "Pair-fold\nstructured square": (
        ROOT / "diagnostics/artifacts/t1_pairfold_square_reduce_2026-08-08",
        "seed{}",
        "#6B7D32",
    ),
}


def values(root: Path, pattern: str, split: str) -> list[float]:
    return [
        100 * json.loads((root / pattern.format(seed) / "eval_report.json").read_text())["result"][split]["exact"]
        for seed in range(3)
    ]


def main() -> None:
    output = ROOT / "solving/experiments/figures/t1_final_label_comparison_2026-08-08.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), sharey=True)
    splits = (("held_out_x", "Held-out x exact"), ("unseen_N", "Unseen N exact"))
    for axis, (split, title) in zip(axes, splits):
        for index, (label, (root, pattern, color)) in enumerate(ARMS.items()):
            scores = values(root, pattern, split)
            jitter = np.array([-0.08, 0.0, 0.08])
            axis.scatter(index + jitter, scores, s=48, facecolor="white", edgecolor=color,
                         linewidth=1.8, zorder=3)
            median = float(np.median(scores))
            axis.hlines(median, index - 0.24, index + 0.24, color=color, linewidth=5, zorder=2)
            axis.text(index, median + 0.75, f"{median:.2f}%", ha="center", va="bottom",
                      color="#20252B", fontsize=10, fontweight="bold")
        axis.set_title(title, loc="left", fontsize=13, fontweight="bold", color="#20252B")
        axis.set_xticks(range(len(ARMS)), ARMS.keys(), fontsize=9)
        axis.set_ylim(0, 25)
        axis.grid(axis="y", color="#D9DEE5", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#7B838C")
    axes[0].set_ylabel("Exact accuracy (%)", color="#20252B")
    fig.suptitle("T=1 final-label architecture comparison", x=0.07, ha="left",
                 fontsize=17, fontweight="bold", color="#20252B")
    fig.text(0.07, 0.91, "Three L40 seeds per arm; dots are seeds and thick lines are medians. All runs reached 100% train exact.",
             fontsize=10, color="#525A64")
    fig.text(0.07, 0.02, "Source: diagnostics/artifacts/*_2026-08-08/eval_report.json  •  Final-label-only research diagnostic",
             fontsize=8.5, color="#68717C")
    fig.tight_layout(rect=(0.04, 0.08, 0.99, 0.88), w_pad=2.5)
    fig.savefig(output, dpi=180, facecolor="white")
    print(output)


if __name__ == "__main__":
    main()

"""Plot the hosted public-dataset ladder from frozen structured results."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS = json.loads((ROOT / "results.json").read_text())["datasets"]


def panel(axis, names, title, color):
    values = [100 * RESULTS[name]["mean_exact"] for name in names]
    bars = axis.bar(names, values, color=color, edgecolor="#243447", linewidth=0.8)
    axis.set_title(title, loc="left", fontsize=12, weight="bold")
    axis.set_ylabel("Mean exact accuracy (%)")
    axis.grid(axis="y", color="#DCE3EA", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )


fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0))
fig.subplots_adjust(left=0.075, right=0.985, bottom=0.16, top=0.79, wspace=0.15)
panel(axes[0], ["e1", "e2", "e3", "e4", "e5"], "Easy: 60-second training", "#2F80ED")
panel(axes[1], ["m1", "m2", "m3", "m4", "m5"], "Medium: 600-second training", "#F2994A")
fig.suptitle(
    "One exact Fable T-cap/AdamW source across every public dataset",
    fontsize=14,
    weight="bold",
)
fig.text(
    0.5,
    0.045,
    "SHA-1 aa75819a878fab6c03c6a23d979f6234560f6e3d · no dataset certified a T rung",
    ha="center",
    fontsize=8,
    color="#52606D",
)
fig.savefig(ROOT / "dataset_ladder.png", dpi=180, facecolor="white")

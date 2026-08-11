"""Plot the hosted Hard training loss and final evaluation references."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
METRICS = ROOT / "diagnostics/artifacts/hosted_hard_05f53719/metrics.jsonl"
OUTPUT = Path(__file__).with_name("hard_loss_curve.png")


rows = [json.loads(line) for line in METRICS.read_text().splitlines() if line]
train = [row for row in rows if row.get("type") == "train"]
evaluation = {
    row["split"]: row["loss"]
    for row in rows
    if row.get("type") == "evaluation"
}

minutes = np.asarray([row["elapsed_seconds"] / 60 for row in train])
loss = np.asarray([row["loss"] for row in train])
window = 9
smooth = np.convolve(loss, np.ones(window) / window, mode="valid")
smooth_minutes = minutes[window - 1 :]

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#d7dce2",
        "grid.linewidth": 0.7,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

fig, ax = plt.subplots(figsize=(10.5, 5.8))
ax.plot(minutes, loss, color="#8bbde0", linewidth=0.8, alpha=0.55, label="logged train batch")
ax.plot(smooth_minutes, smooth, color="#0072B2", linewidth=2.2, label="train loss, 9-point mean")

references = [
    ("test", "final test", "#D55E00", "--"),
    ("ood_t", "final OOD-T", "#6b7280", "-."),
    ("ood_n_t", "final OOD-N", "#111827", ":"),
]
for split, label, color, linestyle in references:
    ax.axhline(evaluation[split], color=color, linestyle=linestyle, linewidth=1.5,
               label=f"{label}: {evaluation[split]:.3f}")

uniform = math.log(17)
ax.axhline(uniform, color="#9ca3af", linewidth=1.1, linestyle=(0, (2, 3)),
           label=f"uniform over 17 tokens: {uniform:.3f}")

ax.scatter([minutes[-1]], [loss[-1]], color="#0072B2", s=28, zorder=5)
ax.annotate(
    f"final logged train = {loss[-1]:.3f}\n45,376 updates",
    (minutes[-1], loss[-1]),
    xytext=(-150, 34),
    textcoords="offset points",
    arrowprops={"arrowstyle": "-", "color": "#0072B2"},
    color="#004f7c",
)

ax.set_title("Hard submission loss over the 60-minute training budget", loc="left", weight="bold")
ax.set_xlabel("elapsed training time (minutes)")
ax.set_ylabel("token cross-entropy loss (nats; lower is better)")
ax.set_xlim(0, 60.5)
ax.set_ylim(2.10, 3.02)
ax.legend(frameon=False, ncol=2, loc="upper right")
fig.text(
    0.01,
    0.01,
    "Final evaluation losses are single end-of-run measurements, not curves. "
    "Source: hosted job 05f53719-7717-4923-88d5-a3cafe373167; 229 train logs, seed 74.",
    fontsize=7.5,
    color="#4b5563",
)
fig.tight_layout(rect=(0, 0.045, 1, 1))
fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
print(OUTPUT)

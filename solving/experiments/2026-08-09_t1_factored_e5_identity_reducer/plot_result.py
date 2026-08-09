"""Plot the learned reduction gate and training loss from the verified run."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
REPORT = (
    ROOT
    / "diagnostics/artifacts/prime-0c1aba701be94af3bb8494f88e962a53"
    / "runs/t1_factored_e5_identity_reducer/seed0/eval_report.json"
)
OUTPUT = Path(__file__).resolve().parent / "gate_trajectory.png"

result = json.loads(REPORT.read_text())["result"]
curve = result["curve"]
seconds = [point["seconds"] for point in curve]
gates = [point["reduce_gate"] for point in curve]
losses = [max(point["loss"], 1e-8) for point in curve]

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
fig.subplots_adjust(left=0.075, right=0.98, bottom=0.18, top=0.77, wspace=0.25)

axes[0].plot(seconds, gates, color="#2F80ED", linewidth=2.4)
axes[0].axhline(
    0.05, color="#C0392B", linestyle="--", linewidth=1.2, label="promotion gate"
)
axes[0].set(
    title="Reducer gate opened",
    xlabel="Training seconds",
    ylabel="sigmoid(gate)",
)
axes[0].legend(frameon=False)

axes[1].plot(seconds, losses, color="#F2994A", linewidth=1.6)
axes[1].set_yscale("log")
axes[1].set(
    title="Training still memorized",
    xlabel="Training seconds",
    ylabel="Batch CE loss",
)

for axis in axes:
    axis.grid(color="#DCE3EA", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)

fig.suptitle(
    "Identity initialization changed optimization, not generalization",
    fontsize=14,
    weight="bold",
)
fig.text(
    0.5,
    0.045,
    "gate 0.010 → 0.759 · train 1600/1600 · seen-N 7/512 · OOD-N 1/512",
    ha="center",
    fontsize=10,
    color="#52606D",
)
fig.savefig(OUTPUT, dpi=180, facecolor="white")

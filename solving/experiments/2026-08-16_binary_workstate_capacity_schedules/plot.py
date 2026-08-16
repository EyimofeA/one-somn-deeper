import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
BACKUP = ROOT / "diagnostics/artifacts/prime-7072f85e48094888bcf3893db897ea54"
REPORTS = {
    "128 constant 3e-4": BACKUP / "binary-workstate-adamw-2026-08-16/exact_square_clean/eval_report.json",
    "192 constant 3e-4": BACKUP / "binary-workstate-capacity-schedules-2026-08-16/width192/eval_report.json",
    "128 warmup + cosine": BACKUP / "binary-workstate-capacity-schedules-2026-08-16/warmup_cosine/eval_report.json",
    "128 warmup + inverse sqrt": BACKUP / "binary-workstate-capacity-schedules-2026-08-16/warmup_inverse_sqrt/eval_report.json",
}
COLORS = ["#2563eb", "#dc2626", "#d97706", "#059669"]


def load_curve(path):
    report = json.loads(path.read_text())
    points = [point for point in report["curve"] if point["step"] >= 500]
    return report, points


plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.0))

for (label, path), color in zip(REPORTS.items(), COLORS):
    report, points = load_curve(path)
    steps = [point["step"] for point in points]
    seconds = [point["seconds"] for point in points]
    exact = [100 * point["validation_exact"] for point in points]
    axes[0].plot(steps, exact, marker="o", markersize=3, linewidth=2, label=label, color=color)
    axes[1].plot(seconds, exact, marker="o", markersize=3, linewidth=2, label=label, color=color)

axes[0].set_title("Same examples: width helps")
axes[0].set_xlabel("Optimizer steps (batch 512)")
axes[0].set_ylabel("Validation exact accuracy (%)")
axes[0].set_xlim(500, 10000)

axes[1].set_title("Same wall time: width is slower")
axes[1].set_xlabel("Elapsed seconds (includes compilation)")
axes[1].set_ylabel("Validation exact accuracy (%)")
axes[1].set_xlim(0, 1650)

for axis in axes:
    axis.axhline(25, color="#6b7280", linestyle="--", linewidth=1.2, label="25% diagnostic gate")
    axis.set_ylim(0, 26)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=3, frameon=False)
fig.suptitle("Exact-square reduction: capacity and AdamW schedules", fontsize=15, fontweight="bold")
fig.text(
    0.5,
    0.018,
    "Deterministic seed-74 split • 44 tied updates • 5.12M examples • checkpoint selected on validation",
    ha="center",
    fontsize=9,
    color="#4b5563",
)
fig.tight_layout(rect=(0, 0.19, 1, 0.93))

output = ROOT / "solving/figures/binary_workstate_capacity_schedules_2026-08-16.png"
fig.savefig(output, dpi=180, bbox_inches="tight")
print(output)

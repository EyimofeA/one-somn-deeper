import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = (
    ROOT
    / "diagnostics/artifacts/prime-7072f85e48094888bcf3893db897ea54"
)
SCREEN = ARTIFACTS / "binary-workstate-fused-tuned-muon-screen-2026-08-16"
REPORTS = {
    "AdamW 3e-4": ARTIFACTS
    / "binary-workstate-fused-varying-n-adamw-2026-08-16/full/eval_report.json",
    "Muon 0.001": SCREEN / "lr001/eval_report.json",
    "Muon 0.003": SCREEN / "lr003/eval_report.json",
    "Muon 0.006": SCREEN / "lr006/eval_report.json",
}
COLORS = ["#2563eb", "#6b7280", "#d97706", "#059669"]


def load(path: Path) -> dict:
    return json.loads(path.read_text())


plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))

for (label, path), color in zip(REPORTS.items(), COLORS):
    report = load(path)
    points = [point for point in report["curve"] if point["step"] <= 3_000]
    steps = [point["step"] for point in points]
    axes[0].plot(
        steps,
        [100 * point["train_exact"] for point in points],
        marker="o",
        markersize=4,
        linewidth=2.2,
        color=color,
        label=label,
    )
    axes[1].plot(
        steps,
        [100 * point["validation_exact"] for point in points],
        marker="o",
        markersize=4,
        linewidth=2.2,
        color=color,
        label=label,
    )

axes[0].set_title("Training exact accuracy")
axes[1].set_title("Validation exact accuracy")
for axis in axes:
    axis.set_xlabel("Optimizer steps (batch 512)")
    axis.set_ylabel("Exact accuracy (%)")
    axis.set_xlim(0, 3_000)
    axis.set_ylim(0, 7)
    axis.legend(frameon=False, loc="upper left")

fig.suptitle(
    "Full fused x² mod N: tuned Muon learning-rate screen",
    fontsize=15,
    fontweight="bold",
)
fig.text(
    0.5,
    0.025,
    "Width 128 • seed 74 • 44 tied updates • 9% dropout • Muon warmup 250 steps",
    ha="center",
    fontsize=9.5,
    color="#4b5563",
)
fig.tight_layout(rect=(0, 0.08, 1, 0.93))
output = ROOT / "solving/figures/binary_workstate_fused_muon_screen_2026-08-16.png"
fig.savefig(output, dpi=180, bbox_inches="tight")
print(output)

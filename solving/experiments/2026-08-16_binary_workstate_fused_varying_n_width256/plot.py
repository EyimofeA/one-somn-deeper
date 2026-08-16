import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = (
    ROOT
    / "diagnostics/artifacts/prime-7072f85e48094888bcf3893db897ea54"
)
REPORTS = {
    "width 128": ARTIFACTS
    / "binary-workstate-fused-varying-n-adamw-2026-08-16/full/eval_report.json",
    "width 256": ARTIFACTS
    / "binary-workstate-fused-varying-n-width256-2026-08-16/full/eval_report.json",
}
COLORS = {"width 128": "#2563eb", "width 256": "#dc2626"}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


reports = {label: load(path) for label, path in REPORTS.items()}
plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))

for label, report in reports.items():
    curve = [point for point in report["curve"] if point["step"] >= 500]
    exact = [100 * point["validation_exact"] for point in curve]
    axes[0].plot(
        [point["step"] for point in curve],
        exact,
        marker="o",
        markersize=3,
        linewidth=2.3,
        color=COLORS[label],
        label=label,
    )
    axes[1].plot(
        [point["seconds"] for point in curve],
        exact,
        marker="o",
        markersize=3,
        linewidth=2.3,
        color=COLORS[label],
        label=label,
    )

axes[0].set_title("Capacity improves learning per optimizer step")
axes[0].set_xlabel("Optimizer steps (batch 512)")
axes[0].set_ylabel("Validation exact accuracy (%)")
axes[0].set_xlim(500, 10_000)

axes[1].set_title("The gain mostly disappears per wall-clock second")
axes[1].set_xlabel("Elapsed seconds (includes compilation)")
axes[1].set_ylabel("Validation exact accuracy (%)")
axes[1].set_xlim(0, 2_250)

for axis in axes:
    axis.axhline(10, color="#6b7280", linestyle="--", linewidth=1.2, label="10% gate")
    axis.set_ylim(0, 16)
    axis.legend(frameon=False, loc="upper left")

fig.suptitle(
    "Full fused x² mod N: width 128 versus 256",
    fontsize=15,
    fontweight="bold",
)
fig.text(
    0.5,
    0.025,
    "Same seed-74 data, 44 tied updates, AdamW 3e-4, 9% dropout, and 5.12M examples",
    ha="center",
    fontsize=9.5,
    color="#4b5563",
)
fig.tight_layout(rect=(0, 0.08, 1, 0.93))
output = ROOT / "solving/figures/binary_workstate_fused_width256_2026-08-16.png"
fig.savefig(output, dpi=180, bbox_inches="tight")
print(output)

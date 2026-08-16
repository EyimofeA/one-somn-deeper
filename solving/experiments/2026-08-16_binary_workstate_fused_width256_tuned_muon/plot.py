import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = (
    ROOT
    / "diagnostics/artifacts/prime-7072f85e48094888bcf3893db897ea54"
)
REPORTS = {
    "128 + AdamW": ARTIFACTS
    / "binary-workstate-fused-varying-n-adamw-2026-08-16/full/eval_report.json",
    "256 + AdamW": ARTIFACTS
    / "binary-workstate-fused-varying-n-width256-2026-08-16/full/eval_report.json",
    "128 + tuned Muon": ARTIFACTS
    / "binary-workstate-fused-tuned-muon-full-2026-08-16/full/eval_report.json",
    "256 + tuned Muon": ARTIFACTS
    / "binary-workstate-fused-width256-tuned-muon-2026-08-16/full/eval_report.json",
}
COLORS = ["#2563eb", "#dc2626", "#059669", "#7c3aed"]


def load(path: Path) -> dict:
    return json.loads(path.read_text())


plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))

for (label, path), color in zip(REPORTS.items(), COLORS):
    report = load(path)
    points = [point for point in report["curve"] if point["step"] >= 500]
    exact = [100 * point["validation_exact"] for point in points]
    axes[0].plot(
        [point["step"] for point in points],
        exact,
        marker="o",
        markersize=3,
        linewidth=2.2,
        color=color,
        label=label,
    )
    axes[1].plot(
        [point["seconds"] for point in points],
        exact,
        marker="o",
        markersize=3,
        linewidth=2.2,
        color=color,
        label=label,
    )

axes[0].set_title("Learning per optimizer step")
axes[0].set_xlabel("Optimizer steps (batch 512)")
axes[0].set_ylabel("Validation exact accuracy (%)")
axes[0].set_xlim(500, 10_000)
axes[1].set_title("Learning per wall-clock second")
axes[1].set_xlabel("Elapsed seconds (includes compilation)")
axes[1].set_ylabel("Validation exact accuracy (%)")

for axis in axes:
    axis.axhline(10, color="#6b7280", linestyle="--", linewidth=1.2, label="10% gate")
    axis.legend(frameon=False, loc="upper left")

fig.suptitle(
    "Full fused x² mod N: capacity × optimizer factorial",
    fontsize=15,
    fontweight="bold",
)
fig.text(
    0.5,
    0.025,
    "Same seed-74 data, 44 tied updates, 9% dropout, 5.12M examples, and final-residue-only loss",
    ha="center",
    fontsize=9.5,
    color="#4b5563",
)
fig.tight_layout(rect=(0, 0.08, 1, 0.93))
output = ROOT / "solving/figures/binary_workstate_fused_capacity_optimizer_2026-08-16.png"
fig.savefig(output, dpi=180, bbox_inches="tight")
print(output)

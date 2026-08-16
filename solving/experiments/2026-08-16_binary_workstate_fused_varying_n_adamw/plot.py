import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "diagnostics/artifacts/prime-7072f85e48094888bcf3893db897ea54"
REPORTS = {
    "Fused + Muon": ARTIFACTS / "binary-workstate-matched-2026-08-16/fused_x/eval_report.json",
    "Fused + AdamW": ARTIFACTS / "binary-workstate-fused-varying-n-adamw-2026-08-16/full/eval_report.json",
    "Exact square + AdamW": ARTIFACTS / "binary-workstate-adamw-2026-08-16/exact_square_clean/eval_report.json",
}
COLORS = ["#dc2626", "#2563eb", "#059669"]


def load(path):
    return json.loads(path.read_text())


plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))

for (label, path), color in zip(REPORTS.items(), COLORS):
    report = load(path)
    curve = report["curve"]
    steps = [point["step"] for point in curve]
    validation = [100 * point["validation_exact"] for point in curve]
    axes[0].plot(steps, validation, color=color, linewidth=2.5, marker="o", markersize=3, label=label)

fused_muon = load(REPORTS["Fused + Muon"])["selected"]
fused_adamw = load(REPORTS["Fused + AdamW"])["selected"]
groups = ["Train", "Unseen x\nseen N", "Seen x\nunseen N", "Unseen x\nand N"]
keys = ["train", "validation_unseen_x_seen_n", "audit_seen_x_unseen_n", "audit_unseen_x_unseen_n"]
x_positions = range(len(groups))
width = 0.34
axes[1].bar([x - width / 2 for x in x_positions], [100 * fused_muon[key]["exact"] for key in keys], width, label="Muon", color="#dc2626")
axes[1].bar([x + width / 2 for x in x_positions], [100 * fused_adamw[key]["exact"] for key in keys], width, label="AdamW", color="#2563eb")

axes[0].axhline(5, color="#6b7280", linestyle="--", linewidth=1.2, label="5% material gate")
axes[0].set_title("AdamW reveals stable fused learning")
axes[0].set_xlabel("Optimizer steps (batch 512)")
axes[0].set_ylabel("Validation exact accuracy (%)")
axes[0].set_ylim(0, 16)
axes[0].legend(frameon=False, loc="upper left")

axes[1].set_title("AdamW transfers to unseen x and N")
axes[1].set_ylabel("Selected-checkpoint exact accuracy (%)")
axes[1].set_xticks(list(x_positions), groups)
axes[1].set_ylim(0, 10)
axes[1].legend(frameon=False, loc="upper left")

fig.suptitle("Full fused binary work-state: optimizer-controlled T=1 comparison", fontsize=15, fontweight="bold")
fig.text(
    0.5,
    0.025,
    "Same seed-74 split • 128 channels • 44 tied updates • 9% dropout • 5.12M examples",
    ha="center",
    fontsize=9.5,
    color="#4b5563",
)
fig.tight_layout(rect=(0, 0.08, 1, 0.93))
output = ROOT / "solving/figures/binary_workstate_fused_varying_n_adamw_2026-08-16.png"
fig.savefig(output, dpi=180, bbox_inches="tight")
print(output)

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "diagnostics/artifacts/prime-7072f85e48094888bcf3893db897ea54"
SHORT = ARTIFACTS / "binary-workstate-fused-varying-n-adamw-2026-08-16/full/eval_report.json"
LONG = ARTIFACTS / "binary-workstate-fused-varying-n-adamw-20k-2026-08-16/full/eval_report.json"
EXACT = ARTIFACTS / "binary-workstate-adamw-2026-08-16/exact_square_clean/eval_report.json"


def load(path):
    return json.loads(path.read_text())


short = load(SHORT)
long = load(LONG)
exact = load(EXACT)

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))

for report, label, color in [
    (short, "10k run", "#2563eb"),
    (long, "20k rerun", "#7c3aed"),
]:
    curve = report["curve"]
    axes[0].plot(
        [point["step"] for point in curve],
        [100 * point["validation_exact"] for point in curve],
        linewidth=2.5,
        marker="o",
        markersize=3,
        color=color,
        label=label,
    )

axes[0].axhline(10, color="#6b7280", linestyle="--", linewidth=1.2, label="10% gate")
axes[0].axhline(100 * exact["selected"]["validation_unseen_x_seen_n"]["exact"], color="#059669", linestyle=":", linewidth=2, label="Exact-square 10k")
axes[0].set_title("More compute helps gradually")
axes[0].set_xlabel("Optimizer steps (batch 512)")
axes[0].set_ylabel("Validation exact accuracy (%)")
axes[0].set_ylim(0, 16)
axes[0].legend(frameon=False, loc="upper left")

groups = ["Train", "Unseen x\nseen N", "Seen x\nunseen N", "Unseen x\nand N"]
keys = ["train", "validation_unseen_x_seen_n", "audit_seen_x_unseen_n", "audit_unseen_x_unseen_n"]
x_positions = range(len(groups))
width = 0.34
axes[1].bar([x - width / 2 for x in x_positions], [100 * short["selected"][key]["exact"] for key in keys], width, color="#2563eb", label="10k")
axes[1].bar([x + width / 2 for x in x_positions], [100 * long["selected"][key]["exact"] for key in keys], width, color="#7c3aed", label="20k")
axes[1].axhline(10, color="#6b7280", linestyle="--", linewidth=1.2)
axes[1].set_title("20k improves every selected split")
axes[1].set_ylabel("Selected-checkpoint exact accuracy (%)")
axes[1].set_xticks(list(x_positions), groups)
axes[1].set_ylim(0, 16)
axes[1].legend(frameon=False, loc="upper left")

fig.suptitle("Full fused varying-N AdamW: doubling the example budget", fontsize=15, fontweight="bold")
fig.text(
    0.5,
    0.025,
    "Same seed and configuration, but compiled BF16 execution is not bitwise deterministic",
    ha="center",
    fontsize=9.5,
    color="#4b5563",
)
fig.tight_layout(rect=(0, 0.08, 1, 0.93))
output = ROOT / "solving/figures/binary_workstate_fused_varying_n_adamw_20k_2026-08-16.png"
fig.savefig(output, dpi=180, bbox_inches="tight")
print(output)

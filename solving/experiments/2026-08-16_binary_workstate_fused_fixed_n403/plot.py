import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
REPORT = (
    ROOT
    / "diagnostics/artifacts/prime-7072f85e48094888bcf3893db897ea54"
    / "binary-workstate-fused-fixed-n403-2026-08-16/full/eval_report.json"
)
OUTPUT = ROOT / "solving/figures/binary_workstate_fused_fixed_n403_2026-08-16.png"

report = json.loads(REPORT.read_text())
curve = report["curve"]
steps = [point["step"] for point in curve]
train = [100 * point["train_exact"] for point in curve]
validation = [100 * point["validation_exact"] for point in curve]
loss = [point["loss"] for point in curve]

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))

axes[0].plot(steps, train, color="#2563eb", linewidth=2.5, marker="o", markersize=3, label="Train x")
axes[0].plot(steps, validation, color="#dc2626", linewidth=2.5, marker="o", markersize=3, label="Held-out x")
axes[0].axvline(1250, color="#6b7280", linestyle="--", linewidth=1.2, label="100% train")
axes[0].set_title("Perfect memorization, no held-out rule")
axes[0].set_xlabel("Optimizer steps")
axes[0].set_ylabel("Exact accuracy (%)")
axes[0].set_ylim(-3, 105)
axes[0].legend(frameon=False, loc="center right")

axes[1].semilogy(steps, loss, color="#059669", linewidth=2.5, marker="o", markersize=3)
axes[1].set_title("Training objective keeps improving")
axes[1].set_xlabel("Optimizer steps")
axes[1].set_ylabel("Last-minibatch BCE loss (log scale)")

fig.suptitle("Full fused x² mod 403: terminal labels select a lookup solution", fontsize=15, fontweight="bold")
fig.text(
    0.5,
    0.025,
    "282 train x • 60 validation x • 61 untouched audit x • final-label-only supervision",
    ha="center",
    fontsize=9.5,
    color="#4b5563",
)
fig.tight_layout(rect=(0, 0.08, 1, 0.93))
fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
print(OUTPUT)

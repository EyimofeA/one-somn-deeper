"""Plot matched learning curves and the quotient frontier shift."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "solving" / "experiments"
HERE = Path(__file__).resolve().parent

CURVES = {
    "fused x, 33 clocks": EXP
    / "2026-08-17_binary_local_kernel_width128_updates33_t1/full/eval_report.json",
    "exact square, 33 clocks": EXP
    / "2026-08-17_binary_local_width128_updates33_exact_square_t1/full/eval_report.json",
    "exact square, 44 clocks": EXP
    / "2026-08-17_binary_local_width128_updates44_exact_square_t1/full/eval_report.json",
    "exact square, 55 clocks": EXP
    / "2026-08-17_binary_local_width128_updates55_exact_square_t1/full/eval_report.json",
}
FRONTIERS = {
    "33 clocks": HERE / "exact_square.json",
    "44 clocks": HERE / "exact_square_updates44.json",
    "55 clocks": HERE / "exact_square_updates55.json",
}
BUCKETS = ["q=0", "q=1", "q=2..3", "q=4..7", "q=8..15", "q=16..31", "q=32..63", "q>=64"]


def read(path: Path) -> dict:
    return json.loads(path.read_text())


fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for label, path in CURVES.items():
    if not path.exists():
        continue
    curve = read(path)["curve"]
    axes[0].plot(
        [row["examples"] / 1e6 for row in curve],
        [100 * row["validation_exact"] for row in curve],
        marker="o",
        markersize=2.5,
        linewidth=1.8,
        label=label,
    )
axes[0].set(title="Held-out x, seen N", xlabel="Training examples (millions)", ylabel="Exact accuracy (%)")
axes[0].grid(alpha=0.25)
axes[0].legend(fontsize=8)

for label, path in FRONTIERS.items():
    if not path.exists():
        continue
    groups = read(path)["validation_unseen_x_seen_n"]["by_quotient"]
    axes[1].plot(
        range(len(BUCKETS)),
        [100 * groups[bucket]["exact"] for bucket in BUCKETS],
        marker="o",
        linewidth=2,
        label=label,
    )
axes[1].set(title="Reducer frontier by raw quotient", xlabel="floor(x^2 / N)", ylabel="Exact accuracy (%)")
axes[1].set_xticks(range(len(BUCKETS)), BUCKETS, rotation=35, ha="right")
axes[1].grid(alpha=0.25)
axes[1].legend(fontsize=8)

fig.suptitle("One Layer Deeper T=1: squaring drag and recurrent reduction horizon")
fig.tight_layout()
fig.savefig(HERE / "t1_reduction_frontier.png", dpi=180, bbox_inches="tight")

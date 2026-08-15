import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
COLORS = {"E1": "#2f80ed", "E7": "#27ae60", "E10": "#9b51e0", "M6": "#eb5757"}


def read(name):
    return [json.loads(line) for line in (ROOT / "metrics" / f"{name.lower()}.jsonl").read_text().splitlines()]


fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
scores = {}
for name in ("E1", "E7", "E10", "M6"):
    rows = read(name)
    train = [row for row in rows if row.get("type") == "train"]
    axes[0].plot([row["elapsed_seconds"] for row in train],
                 [100 * row["exact_accuracy"] for row in train],
                 label=name, color=COLORS[name], linewidth=2)
    evaluations = {row["split"]: 100 * row["exact_accuracy"] for row in rows
                   if row.get("type") == "evaluation"}
    scores[name] = evaluations

axes[0].set(xlabel="Hosted training seconds", ylabel="Train exact (%)",
            title="Learning then Medium collapse")
axes[0].legend(frameon=False)
axes[0].grid(alpha=.2)

easy_names = [f"E{i}" for i in range(1, 11)]
easy_scores = {}
for name in easy_names:
    rows = read(name)
    easy_scores[name] = {row["split"]: 100 * row["exact_accuracy"] for row in rows
                         if row.get("type") == "evaluation"}
x = range(10)
axes[1].bar([i - .18 for i in x], [easy_scores[name]["test"] for name in easy_names],
            width=.36, label="Test", color="#56ccf2")
axes[1].bar([i + .18 for i in x], [easy_scores[name]["ood"] for name in easy_names],
            width=.36, label="OOD", color="#f2c94c")
axes[1].set_xticks(list(x), easy_names)
axes[1].axvspan(1.5, 4.5, color="#eb5757", alpha=.08, label="Varying N")
axes[1].set(ylabel="Exact accuracy (%)", title="Frozen-source hosted transfer")
axes[1].legend(frameon=False)
axes[1].grid(axis="y", alpha=.2)
fig.tight_layout()
fig.savefig(ROOT / "hosted_transfer.png", dpi=180)

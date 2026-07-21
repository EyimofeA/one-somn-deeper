"""Regenerate Easy e1 baseline ladder + train-curve figures from metrics JSONL."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "metrics"
FIGDIR = Path(__file__).resolve().parent

MODELS = {
    "b0 Transformer": "b0_e1.jsonl",
    "b1 MLP×3": "b1_e1.jsonl",
    "b2 BiGRU": "b2_e1.jsonl",
}
COLORS = ["#0072B2", "#E69F00", "#009E73"]


def load(name: str):
    rows = [json.loads(line) for line in (METRICS / name).read_text().splitlines() if line.strip()]
    train = [row for row in rows if row.get("type") == "train"]
    evaluation = {row["split"]: row for row in rows if row.get("type") == "evaluation"}
    summary = next(row for row in rows if row.get("type") == "summary")
    return train, evaluation, summary


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    labels, means, tests, oods = [], [], [], []
    for label, filename in MODELS.items():
        _, evaluation, summary = load(filename)
        labels.append(label)
        means.append(100 * summary["mean_exact_accuracy"])
        tests.append(100 * evaluation["test"]["exact_accuracy"])
        oods.append(100 * evaluation["ood"]["exact_accuracy"])

    xs = list(range(len(labels)))
    width = 0.25
    ax.bar([x - width for x in xs], means, width, label="mean (score)", color=COLORS[0])
    ax.bar(xs, tests, width, label="test", color=COLORS[1])
    ax.bar([x + width for x in xs], oods, width, label="ood", color=COLORS[2])
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Exact accuracy (%)")
    ax.set_ylim(0, max(means + tests + oods) * 1.25 + 0.5)
    ax.set_title("Easy e1 baseline ladder (N=323, T∈{1,2,3}, 60s H100)")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_baseline_ladder_e1.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for (label, filename), color in zip(MODELS.items(), COLORS):
        train, _, summary = load(filename)
        ax.plot(
            [row["step"] for row in train],
            [100 * row["exact_accuracy"] for row in train],
            label=f"{label} ({summary['completed_steps']} steps)",
            color=color,
            lw=2,
        )
    ax.set_xlabel("Train step")
    ax.set_ylabel("Train exact accuracy (%)")
    ax.set_ylim(bottom=0)
    ax.set_title("Easy e1 train exact accuracy vs step")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_baseline_train_curves_e1.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()

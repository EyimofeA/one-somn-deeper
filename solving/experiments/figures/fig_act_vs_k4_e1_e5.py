"""ACT soft-halt vs fixed K=4 on Easy e1/e5."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "metrics"
OUT = Path(__file__).with_suffix(".png")

C_BASE = "#0072B2"
C_ACT = "#009E73"


def load(name: str) -> dict:
    rows = [
        json.loads(line)
        for line in (METRICS / name).read_text().splitlines()
        if line.strip()
    ]
    summary = next(r for r in rows if r["type"] == "summary")
    evals = {
        r["split"]: r["exact_accuracy"] for r in rows if r["type"] == "evaluation"
    }
    return {
        "mean": 100.0 * summary["mean_exact_accuracy"],
        "test": 100.0 * evals["test"],
        "ood": 100.0 * evals["ood"],
        "steps": summary["completed_steps"],
    }


def main() -> None:
    pairs = [
        ("Easy e1", "depth_d32_k4_e1.jsonl", "depth_d32_act_e1.jsonl"),
        ("Easy e5", "depth_d32_k4_e5.jsonl", "depth_d32_act_e5.jsonl"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    metrics = ["mean", "test", "ood"]
    x = range(len(metrics))
    width = 0.35
    for ax, title, base_n, act_n in [
        (axes[0], pairs[0][0], pairs[0][1], pairs[0][2]),
        (axes[1], pairs[1][0], pairs[1][1], pairs[1][2]),
    ]:
        b, a = load(base_n), load(act_n)
        ax.bar(
            [i - width / 2 for i in x],
            [b[m] for m in metrics],
            width,
            label="fixed K=4",
            color=C_BASE,
        )
        ax.bar(
            [i + width / 2 for i in x],
            [a[m] for m in metrics],
            width,
            label="ACT K_max=8",
            color=C_ACT,
        )
        ax.set_xticks(list(x), metrics)
        ax.set_ylabel("exact accuracy (%)")
        ax.set_title(title)
        ax.set_ylim(0, max([b[m] for m in metrics] + [a[m] for m in metrics]) * 1.3 + 0.2)
        ax.text(
            0.02,
            0.98,
            f"steps {b['steps']} vs {a['steps']}",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )
    axes[0].legend(frameon=False)
    fig.suptitle("Adaptive loops (soft ACT) vs fixed K=4", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

"""Midloop (pre/mid×4/post) vs UT K4 on Easy e1/e5."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "metrics"
OUT = Path(__file__).with_suffix(".png")

C_UT = "#E69F00"
C_MID = "#CC79A7"


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
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    metrics = ["mean", "test", "ood"]
    x = range(len(metrics))
    w = 0.35
    for ax, title, ut_n, mid_n in [
        (axes[0], "Easy e1", "depth_d32_k4_ut_e1.jsonl", "depth_d32_midloop_k4_e1.jsonl"),
        (axes[1], "Easy e5", "depth_d32_k4_ut_e5.jsonl", "depth_d32_midloop_k4_e5.jsonl"),
    ]:
        ut, mid = load(ut_n), load(mid_n)
        ax.bar([i - w / 2 for i in x], [ut[m] for m in metrics], w, label="UT K4", color=C_UT)
        ax.bar(
            [i + w / 2 for i in x],
            [mid[m] for m in metrics],
            w,
            label="midloop K4",
            color=C_MID,
        )
        ax.set_xticks(list(x), metrics)
        ax.set_ylabel("exact (%)")
        ax.set_title(title)
        ax.set_ylim(0, max([ut[m] for m in metrics] + [mid[m] for m in metrics]) * 1.3 + 0.2)
        ax.text(
            0.02,
            0.98,
            f"steps {ut['steps']} vs {mid['steps']}",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )
    axes[0].legend(frameon=False)
    fig.suptitle("Middle-only loop vs full UT depth loop", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

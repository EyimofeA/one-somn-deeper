"""Compare depth_d32_k4 vs N-conditioned FiLM on Easy e1 and e5."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "metrics"
OUT = Path(__file__).with_suffix(".png")

# Okabe–Ito-ish
C_BASE = "#0072B2"
C_NCOND = "#D55E00"


def load_summary(name: str) -> dict:
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
    base_e1 = load_summary("depth_d32_k4_e1.jsonl")
    ncond_e1 = load_summary("ncond_d32_k4_e1.jsonl")
    base_e5 = load_summary("depth_d32_k4_e5.jsonl")
    ncond_e5 = load_summary("ncond_d32_k4_e5.jsonl")

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), sharey=False)
    metrics = ["mean", "test", "ood"]
    x = range(len(metrics))
    width = 0.35

    for ax, title, base, ncond in [
        (axes[0], "Easy e1 (fixed N=323)", base_e1, ncond_e1),
        (axes[1], "Easy e5 (10–11 bit N)", base_e5, ncond_e5),
    ]:
        b = [base[m] for m in metrics]
        n = [ncond[m] for m in metrics]
        ax.bar([i - width / 2 for i in x], b, width, label="d32×K4", color=C_BASE)
        ax.bar(
            [i + width / 2 for i in x], n, width, label="d32×K4 + N-FiLM", color=C_NCOND
        )
        ax.set_xticks(list(x), metrics)
        ax.set_ylabel("exact accuracy (%)")
        ax.set_title(title)
        ax.set_ylim(0, max(b + n) * 1.25 + 0.5)
        ax.axhline(0, color="black", lw=0.6)
        note = f"steps {base['steps']} vs {ncond['steps']}"
        ax.text(0.02, 0.98, note, transform=ax.transAxes, va="top", fontsize=8)

    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("N-conditioning FiLM vs tied-loop baseline", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

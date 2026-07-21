"""Train loss curves for Easy e1 baselines."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt

METRICS = Path(__file__).resolve().parents[1] / "metrics"
FIGDIR = Path(__file__).resolve().parent
MODELS = {"b0 Transformer": "b0_e1.jsonl", "b1 MLP×3": "b1_e1.jsonl", "b2 BiGRU": "b2_e1.jsonl"}
COLORS = ["#0072B2", "#E69F00", "#009E73"]

def main() -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for (label, filename), color in zip(MODELS.items(), COLORS):
        rows = [json.loads(l) for l in (METRICS / filename).read_text().splitlines() if l.strip()]
        train = [r for r in rows if r["type"] == "train"]
        summary = next(r for r in rows if r["type"] == "summary")
        ax.plot(
            [r["step"] for r in train],
            [r["loss"] for r in train],
            color=color, lw=2, marker="o",
            label=f"{label} (hit {summary['training_seconds']:.0f}s wall)",
        )
    ax.set_xlabel("Train step")
    ax.set_ylabel("Train loss (logged batch)")
    ax.set_title("Easy e1 train loss vs step (stop = 60s clock)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_baseline_train_loss_e1.png", dpi=160)
    plt.close(fig)

if __name__ == "__main__":
    main()

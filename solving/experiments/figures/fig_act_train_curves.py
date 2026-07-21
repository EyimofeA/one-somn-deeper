"""Train loss + batch exact for ACT vs fixed K=4."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "metrics"
OUT = Path(__file__).with_suffix(".png")

C_BASE = "#0072B2"
C_ACT = "#009E73"


def load_train(name: str):
    rows = [
        json.loads(line)
        for line in (METRICS / name).read_text().splitlines()
        if line.strip()
    ]
    train = [r for r in rows if r["type"] == "train"]
    return (
        [r["step"] for r in train],
        [r["loss"] for r in train],
        [100.0 * r["exact_accuracy"] for r in train],
    )


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.0))
    for col, tag, base_n, act_n in [
        (0, "e1", "depth_d32_k4_e1.jsonl", "depth_d32_act_e1.jsonl"),
        (1, "e5", "depth_d32_k4_e5.jsonl", "depth_d32_act_e5.jsonl"),
    ]:
        bs, bl, be = load_train(base_n)
        as_, al, ae = load_train(act_n)
        axes[0, col].plot(bs, be, color=C_BASE, marker="o", label="K=4")
        axes[0, col].plot(as_, ae, color=C_ACT, marker="s", label="ACT")
        axes[0, col].set_title(f"Easy {tag} — train exact (batch)")
        axes[0, col].set_ylabel("exact (%)")
        axes[0, col].legend(frameon=False, fontsize=8)
        axes[1, col].plot(bs, bl, color=C_BASE, marker="o")
        axes[1, col].plot(as_, al, color=C_ACT, marker="s")
        axes[1, col].set_title(f"Easy {tag} — train loss")
        axes[1, col].set_xlabel("step")
        axes[1, col].set_ylabel("loss")
    fig.suptitle("ACT vs fixed K=4 train curves", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

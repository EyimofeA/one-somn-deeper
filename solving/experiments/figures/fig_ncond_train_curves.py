"""Train exact curves: baseline vs N-cond on e1 and e5."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "metrics"
OUT = Path(__file__).with_suffix(".png")

C_BASE = "#0072B2"
C_NCOND = "#D55E00"


def load_train(name: str) -> tuple[list[int], list[float], list[float]]:
    rows = [
        json.loads(line)
        for line in (METRICS / name).read_text().splitlines()
        if line.strip()
    ]
    train = [r for r in rows if r["type"] == "train"]
    steps = [r["step"] for r in train]
    exact = [100.0 * r["exact_accuracy"] for r in train]
    loss = [r["loss"] for r in train]
    return steps, exact, loss


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.0), sharex=False)
    pairs = [
        (axes[0, 0], axes[1, 0], "e1", "depth_d32_k4_e1.jsonl", "ncond_d32_k4_e1.jsonl"),
        (axes[0, 1], axes[1, 1], "e5", "depth_d32_k4_e5.jsonl", "ncond_d32_k4_e5.jsonl"),
    ]
    for ax_ex, ax_loss, tag, base_name, ncond_name in pairs:
        bs, be, bl = load_train(base_name)
        ns, ne, nl = load_train(ncond_name)
        ax_ex.plot(bs, be, color=C_BASE, marker="o", label="d32×K4")
        ax_ex.plot(ns, ne, color=C_NCOND, marker="s", label="N-FiLM")
        ax_ex.set_title(f"Easy {tag} — train exact (batch)")
        ax_ex.set_ylabel("exact (%)")
        ax_ex.legend(frameon=False, fontsize=8)
        ax_loss.plot(bs, bl, color=C_BASE, marker="o")
        ax_loss.plot(ns, nl, color=C_NCOND, marker="s")
        ax_loss.set_title(f"Easy {tag} — train loss")
        ax_loss.set_xlabel("step")
        ax_loss.set_ylabel("loss")
    fig.suptitle("N-cond vs baseline train curves", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

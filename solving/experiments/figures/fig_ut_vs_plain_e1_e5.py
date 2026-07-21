"""UT depth-emb vs plain tied loops on e1/e5."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "metrics"
OUT = Path(__file__).with_suffix(".png")


def mean(name: str) -> float:
    rows = [
        json.loads(line)
        for line in (METRICS / name).read_text().splitlines()
        if line.strip()
    ]
    return 100.0 * next(r for r in rows if r["type"] == "summary")["mean_exact_accuracy"]


def main() -> None:
    labels = ["K=2", "K=4"]
    plain_e1 = [mean("depth_d32_k2_e1.jsonl"), mean("depth_d32_k4_e1.jsonl")]
    ut_e1 = [mean("depth_d32_k2_ut_e1.jsonl"), mean("depth_d32_k4_ut_e1.jsonl")]
    plain_e5 = [mean("depth_d32_k2_e5.jsonl"), mean("depth_d32_k4_e5.jsonl")]
    ut_e5 = [mean("depth_d32_k2_ut_e5.jsonl"), mean("depth_d32_k4_ut_e5.jsonl")]

    x = np.arange(len(labels))
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    for ax, title, plain, ut in [
        (axes[0], "Easy e1 mean exact (%)", plain_e1, ut_e1),
        (axes[1], "Easy e5 mean exact (%)", plain_e5, ut_e5),
    ]:
        ax.bar(x - w / 2, plain, w, label="tied loops", color="#0072B2")
        ax.bar(x + w / 2, ut, w, label="+ depth emb (UT)", color="#E69F00")
        ax.set_xticks(x, labels)
        ax.set_ylabel("mean exact (%)")
        ax.set_title(title)
        ax.set_ylim(0, max(plain + ut) * 1.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Universal Transformer depth emb vs plain tied loops", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

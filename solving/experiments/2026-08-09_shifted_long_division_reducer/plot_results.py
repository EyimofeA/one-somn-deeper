"""Render the autonomous exactness comparison from saved run reports."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).parent
RUNS = {
    "Initial shifted training": ("shifted", "#315b7d", "o", "-"),
    "Boundary-only repair": ("boundary", "#d77a1f", "s", "--"),
    "Mixed consolidation": ("mixed", "#637939", "^", "-."),
}


def main() -> None:
    reports = {
        label: json.loads((ROOT / "runs" / folder / "eval_report.json").read_text())["autonomous"]
        for label, (folder, *_style) in RUNS.items()
    }
    quotients = list(next(iter(reports.values())))
    x = range(len(quotients))
    fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=160)
    for label, (_folder, color, marker, linestyle) in RUNS.items():
        y = [100 * reports[label][q]["remainder_exact"] for q in quotients]
        ax.plot(x, y, label=label, color=color, marker=marker, linestyle=linestyle, linewidth=2.2, markersize=6)
    ax.axhline(99.9, color="#3f464d", linewidth=1.2, linestyle=":", label="99.9% gate")
    fig.suptitle("Shifted reducer autonomous exactness by quotient", x=0.09, y=0.97, ha="left", fontsize=15, weight="bold")
    fig.text(0.09, 0.925, "16 unseen four-digit semiprimes; 1,024 examples per quotient and run", fontsize=10, color="#50565c")
    ax.set_ylabel("Exact final remainder (%)")
    ax.set_xlabel("Quotient q (ordered test scale)")
    ax.set_xticks(list(x), [f"{int(q):,}" for q in quotients], rotation=32, ha="right")
    ax.set_ylim(-2, 103)
    ax.grid(axis="y", color="#d9dde1", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="lower left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(ROOT / "autonomous_exactness.png", bbox_inches="tight")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--muon", type=Path, required=True)
    parser.add_argument("--adamw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = ((load(args.muon), "Muon warmdown", "#dc3912"),
               (load(args.adamw), "AdamW", "#3366cc"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for report, label, color in reports:
        curve = report["curve"]
        steps = [row["step"] for row in curve]
        axes[0].plot(steps, [100 * row["validation_exact"] for row in curve], color=color, label=label)
        axes[1].plot(steps, [row["loss"] for row in curve], color=color, label=label)
    axes[0].set(title="Held-out-x, seen-N exact accuracy", xlabel="Optimization step", ylabel="Exact accuracy (%)")
    axes[1].set(title="Training loss", xlabel="Optimization step", ylabel="Binary cross-entropy")
    for axis in axes:
        axis.legend(frameon=False)
        axis.grid(alpha=0.2)
    fig.suptitle("AdamW preserves reduction learning that Muon destroys")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    exact = load(args.artifact_root / "exact_square" / "eval_report.json")
    fused = load(args.artifact_root / "fused_x" / "eval_report.json")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for report, label, color in (
        (exact, "Exact square → reducer", "#3366cc"),
        (fused, "Fused x,N", "#dc3912"),
    ):
        curve = report["curve"]
        steps = [row["step"] for row in curve]
        axes[0].plot(steps, [100 * row["train_exact"] for row in curve], "--", color=color, alpha=0.7)
        axes[0].plot(steps, [100 * row["validation_exact"] for row in curve], color=color, label=label)
        axes[1].plot(steps, [row["loss"] for row in curve], color=color, label=label)
    axes[0].set(title="Exact accuracy", xlabel="Optimization step", ylabel="Exact accuracy (%)")
    axes[1].set(title="Training loss", xlabel="Optimization step", ylabel="Binary cross-entropy")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    axes[0].grid(alpha=0.2)
    axes[1].grid(alpha=0.2)
    fig.suptitle("Matched binary work-state processor: both arms collapse after early signal")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()

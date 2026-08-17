#!/usr/bin/env python3
"""Plot H13 learning and frozen per-prefix state-probe results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
BASELINE = (
    REPO / "diagnostics" / "artifacts"
    / "prime-7072f85e48094888bcf3893db897ea54"
    / "binary-workstate-fused-width256-tuned-muon-2026-08-16"
    / "full" / "eval_report.json"
)
OUTPUT = REPO / "solving" / "figures" / "binary_prefix_residue_h13_2026-08-17.png"


def load(path: Path):
    return json.loads(path.read_text())


def main() -> None:
    h13 = load(ROOT / "eval_report.json")
    curriculum = load(ROOT / "curriculum_eval_report.json")
    probe = load(ROOT / "probe_report.json")
    baseline = load(BASELINE)
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    plt.style.use("seaborn-v0_8-whitegrid")

    for report, label, style in (
        (baseline, "All-at-once fused", "-"),
        (h13, "H13 bit schedule", "--"),
        (curriculum, "H13 length curriculum", "-."),
    ):
        steps = [point["step"] for point in report["curve"]]
        axes[0].plot(
            steps, [100 * point["validation_exact"] for point in report["curve"]],
            linestyle=style, linewidth=2.4, label=f"{label}: validation",
        )
        axes[0].plot(
            steps, [100 * point["train_exact"] for point in report["curve"]],
            linestyle=style, linewidth=1.5, alpha=0.55,
            label=f"{label}: train",
        )
    axes[0].axhline(15, color="#555555", linestyle=":", label="H13 kill boundary")
    axes[0].set(
        title="Serializing x regresses full-transition generalization",
        xlabel="Training step", ylabel="Exact match (%)",
    )
    axes[0].set_ylim(0, 28)
    axes[0].legend(fontsize=8)

    validation = probe["selected"]["validation_unseen_x_seen_n"]
    joint = probe["selected"]["audit_unseen_x_unseen_n"]
    prefix_bits = list(range(1, 12))
    for split, label, color in (
        (validation, "Seen N", "#00798c"),
        (joint, "Unseen N", "#d1495b"),
    ):
        axes[1].plot(
            prefix_bits,
            [100 * item["exact"] for item in split["prefix_residue"]],
            marker="o", linewidth=2.4, color=color, label=f"Residue: {label}",
        )
        axes[1].plot(
            prefix_bits,
            [100 * item["exact"] for item in split["prefix_value"]],
            marker="s", linewidth=1.7, linestyle="--", alpha=0.75,
            color=color, label=f"Prefix value: {label}",
        )
    axes[1].axvspan(5, 11, color="#f4a261", alpha=0.10)
    axes[1].text(5.15, 7, "wrapping becomes common", fontsize=8, color="#8a4f08")
    axes[1].set(
        title="Frozen state is strong before wrap, then loses residue",
        xlabel="Input prefix bits consumed", ylabel="Linear-probe exact match (%)",
        xticks=prefix_bits,
    )
    axes[1].set_ylim(0, 104)
    axes[1].legend(fontsize=8, loc="lower left")

    figure.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180)
    print(OUTPUT)


if __name__ == "__main__":
    main()

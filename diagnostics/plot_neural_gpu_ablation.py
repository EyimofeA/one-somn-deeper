"""Plot the 2026-08-15 Neural GPU multiplication ablation tournament."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = json.loads(args.summary.read_text())
    rows.sort(key=lambda row: row["audit_exact"])
    labels = [row["variant"].replace("_", " ") for row in rows]
    audit = [100 * row["audit_exact"] for row in rows]
    validation = [100 * row["validation_exact"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    colors = ["#cf5c5c" if value < 63.31 else "#4f81bd" for value in audit]
    axes[0].barh(labels, audit, color=colors)
    axes[0].axvline(63.31, color="black", linestyle="--", linewidth=1, label="baseline audit")
    axes[0].set_xlabel("Exact accuracy (%)")
    axes[0].set_title("Untouched audit split")
    axes[0].legend()
    by_name = {row["variant"]: row for row in rows}
    for name, color in (("baseline", "#333333"), ("muon", "#d98324"),
                        ("muon_decay", "#2e8b57")):
        curve = by_name[name]["curve"]
        axes[1].plot([point["step"] for point in curve],
                     [100 * point["validation_exact"] for point in curve],
                     marker="o", markersize=3, label=name.replace("_", " "), color=color)
    axes[1].set_xlabel("Optimizer updates")
    axes[1].set_ylabel("Validation exact (%)")
    axes[1].set_title("Muon discovers faster; warmdown prevents collapse")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()

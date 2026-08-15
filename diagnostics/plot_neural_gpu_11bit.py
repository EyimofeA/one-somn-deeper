"""Plot diagonal control and eleven-bit multiplication scaling."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    diagonal = read(args.artifact_root/"runs_v3/muon_dropout_diagonal/eval_report.json")
    scale = read(args.artifact_root/"runs_v4/muon_dropout_11bit/eval_report.json")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.3), constrained_layout=True)

    dc = diagonal["curve"]
    axes[0].plot([x["examples"]/1e6 for x in dc], [100*x["validation_exact"] for x in dc],
                 color="#d95f02", marker="o", markersize=3, label="+ diagonal")
    axes[0].axhline(97.01, color="#7a3db8", linestyle="--", label="winner without diagonal")
    axes[0].set_title("Diagonal transport regresses")
    axes[0].set_xlabel("Examples processed (millions)")
    axes[0].set_ylabel("Validation exact (%)")
    axes[0].legend()
    axes[0].grid(alpha=.2)

    curve = scale["curve"]
    examples = [x["examples"]/1e6 for x in curve]
    axes[1].plot(examples, [100*x["train_exact"] for x in curve], linestyle="--",
                 color="#7a3db8", label="train monitor")
    axes[1].plot(examples, [100*x["validation_exact"] for x in curve], marker="o",
                 markersize=3, color="#7a3db8", label="validation")
    axes[1].set_title("11-bit multiplication keeps improving")
    axes[1].set_xlabel("Examples processed (millions)")
    axes[1].set_ylabel("Exact accuracy (%)")
    axes[1].legend()
    axes[1].grid(alpha=.2)

    audit = scale["selected"]["audit"]
    lengths = sorted((int(k), v["accuracy"]*100) for k, v in audit["product_length"].items()
                     if int(k) >= 14)
    axes[2].bar([str(k) for k, _ in lengths], [v for _, v in lengths], color="#4f81bd")
    axes[2].set_title("Failure grows with product length")
    axes[2].set_xlabel("Output product bit length")
    axes[2].set_ylabel("Untouched audit exact (%)")
    axes[2].grid(axis="y", alpha=.2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()

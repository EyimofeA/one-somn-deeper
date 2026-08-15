"""Plot fixed-compute, fit-matched, grokking, and depth diagnostics."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    old, new = args.artifact_root / "runs", args.artifact_root / "runs_v2"
    fixed = {name: read(old/name/"eval_report.json") for name in
             ("baseline", "hard", "gradient_noise", "wide", "sharing_relaxation",
              "muon", "sparse_memory", "microprogram", "dropout", "muon_decay")}
    extended = {"dropout": read(new/"dropout_fit"/"eval_report.json"),
                "gradient noise": read(new/"gradient_noise_fit"/"eval_report.json"),
                "Muon warmdown": read(new/"muon_decay_grokking"/"eval_report.json"),
                "Muon + dropout": read(new/"muon_dropout"/"eval_report.json")}
    mechanism = {name: read(args.artifact_root/"diagnostics"/f"{name}_mechanism.json")
                 for name in ("baseline", "muon_decay", "muon_dropout")}

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5), constrained_layout=True)
    labels = ["hard", "wide", "sharing", "sparse memory", "microprogram",
              "dropout", "gradient noise", "Muon warmdown", "Muon + dropout"]
    fixed_audit = [fixed["hard"]["selected"]["audit"]["exact"],
                   fixed["wide"]["selected"]["audit"]["exact"],
                   fixed["sharing_relaxation"]["selected"]["audit"]["exact"],
                   fixed["sparse_memory"]["selected"]["audit"]["exact"],
                   fixed["microprogram"]["selected"]["audit"]["exact"],
                   fixed["dropout"]["selected"]["audit"]["exact"],
                   fixed["gradient_noise"]["selected"]["audit"]["exact"],
                   fixed["muon_decay"]["selected"]["audit"]["exact"], float("nan")]
    fit_audit = [fixed["hard"]["selected"]["audit"]["exact"],
                 fixed["wide"]["selected"]["audit"]["exact"],
                 fixed["sharing_relaxation"]["final"]["audit"]["exact"],
                 fixed["sparse_memory"]["selected"]["audit"]["exact"],
                 fixed["microprogram"]["final"]["audit"]["exact"],
                 extended["dropout"]["selected"]["audit"]["exact"],
                 extended["gradient noise"]["selected"]["audit"]["exact"],
                 extended["Muon warmdown"]["selected"]["audit"]["exact"],
                 extended["Muon + dropout"]["selected"]["audit"]["exact"]]
    x = np.arange(len(labels)); width = 0.38
    axes[0].bar(x-width/2, np.array(fixed_audit)*100, width, label="5.12M examples")
    axes[0].bar(x+width/2, np.array(fit_audit)*100, width, label="fit-matched / extended")
    axes[0].axhline(63.31, color="black", linestyle="--", linewidth=1)
    axes[0].set_xticks(x, labels, rotation=45, ha="right")
    axes[0].set_ylabel("Untouched audit exact (%)")
    axes[0].set_title("Fit matching changes dropout")
    axes[0].legend(fontsize=8)

    colors = {"dropout": "#5b8ff9", "gradient noise": "#e8684a",
              "Muon warmdown": "#2e8b57", "Muon + dropout": "#7a3db8"}
    for name, report in extended.items():
        curve = report["curve"]
        examples = [point["examples"]/1e6 for point in curve]
        axes[1].plot(examples, [100*point["train_exact"] for point in curve],
                     color=colors[name], alpha=.25, linestyle="--")
        axes[1].plot(examples, [100*point["validation_exact"] for point in curve],
                     color=colors[name], marker="o", markersize=2.5, label=name)
    axes[1].axvline(5.12, color="black", linestyle=":", linewidth=1)
    axes[1].set_xlabel("Examples processed (millions)")
    axes[1].set_ylabel("Exact accuracy (%)")
    axes[1].set_title("Training and validation trajectories")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=.2)

    for name, label, color in (("baseline", "AdamW baseline", "#333333"),
                               ("muon_decay", "Muon warmdown", "#2e8b57"),
                               ("muon_dropout", "Muon + dropout", "#7a3db8")):
        curve = mechanism[name]["audit_diagnostics"]["decode_every_step"]
        steps = sorted(map(int, curve))
        axes[2].plot(steps, [100*curve[str(step)] for step in steps], marker="o",
                     markersize=3, label=label, color=color)
    axes[2].axvline(14, color="black", linestyle=":", linewidth=1)
    axes[2].set_xlabel("Inference recurrent updates")
    axes[2].set_ylabel("Untouched audit exact (%)")
    axes[2].set_title("A timed 14-step circuit, not convergence")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=.2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()

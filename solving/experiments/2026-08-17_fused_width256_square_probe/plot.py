#!/usr/bin/env python3
"""Plot validation exact-match curves for the frozen representation probes."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "report.json"
OUTPUT = ROOT.parents[1] / "figures" / "fused_width256_square_probe_2026-08-17.png"


def main() -> None:
    report = json.loads(REPORT.read_text())
    steps = [point["step"] for point in report["curve"]]
    series = {
        "Global square readout": [
            100 * point["validation"]["global_square"]["exact"]
            for point in report["curve"]
        ],
        "Local square readout": [
            100 * point["validation"]["local_square"]["exact"]
            for point in report["curve"]
        ],
        "Input x control": [
            100 * point["validation"]["global_x"]["exact"]
            for point in report["curve"]
        ],
    }

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(9.0, 5.4))
    colors = ("#d1495b", "#edae49", "#00798c")
    for (label, values), color in zip(series.items(), colors):
        axis.plot(steps, values, marker="o", linewidth=2.2, label=label, color=color)
    axis.axhline(25, color="#555555", linestyle="--", linewidth=1.2,
                 label="Preregistered square kill boundary")
    axis.set(title="What is linearly recoverable from the fused model's final work tape?",
             xlabel="Probe training step", ylabel="Validation exact match (%)")
    axis.set_ylim(0, 100)
    axis.legend(loc="center right", frameon=True)
    axis.text(0.01, 0.98,
              "Frozen width-256 tuned-Muon processor; unseen x, seen N",
              transform=axis.transAxes, va="top", fontsize=9)
    figure.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180)
    print(OUTPUT)


if __name__ == "__main__":
    main()

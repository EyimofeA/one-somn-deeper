"""Plot the public e5 canonical-register ablation from preserved runner logs."""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
LOG_ROOT = ROOT / "diagnostics/runs/t1_identifiability/l40s"
OUTPUT = ROOT / "solving/experiments/figures/t1_canonical_ablation_2026-08-08.png"

ARMS = {
    "Prompt reinjection\n+ curriculum": "prompt_reinject_curriculum.log",
    "Canonical\n+ plain": "canonical_plain.log",
    "Canonical\n+ curriculum": "canonical_t1_curriculum.log",
}
COLORS = ["#9AA4B2", "#C47A23", "#276FBF"]


def parse_log(path: Path) -> tuple[list[float], list[float], dict]:
    text = path.read_text()
    curve = [
        (float(elapsed), float(loss))
        for loss, elapsed in re.findall(
            r"step=\d+ loss=([0-9.]+).*?elapsed=([0-9.]+)s", text
        )
    ]
    result_line = next(
        line.removeprefix("RESULT_JSON=")
        for line in text.splitlines()
        if line.startswith("RESULT_JSON=")
    )
    result = json.loads(result_line)
    return [x for x, _ in curve], [y for _, y in curve], result


def main() -> None:
    parsed = {label: parse_log(LOG_ROOT / name) for label, name in ARMS.items()}
    fig, (loss_ax, profile_ax) = plt.subplots(
        1, 2, figsize=(12.8, 5.2), gridspec_kw={"width_ratios": [1.45, 1]}
    )
    fig.patch.set_facecolor("#FAFBFC")

    for (label, (elapsed, loss, _)), color in zip(parsed.items(), COLORS):
        loss_ax.plot(elapsed, loss, color=color, linewidth=2.2, label=label.replace("\n", " "))
    loss_ax.set_title("Training cross-entropy", loc="left", weight="bold", y=1.10)
    loss_ax.text(
        0,
        1.025,
        "Public Easy e5, seed 74, 60-second L40S budget; logarithmic loss axis",
        transform=loss_ax.transAxes,
        color="#5B6573",
        fontsize=9,
    )
    loss_ax.set_xlabel("Elapsed training time (seconds)")
    loss_ax.set_ylabel("Final-label cross-entropy")
    loss_ax.set_yscale("log")
    loss_ax.grid(axis="y", color="#DDE2E8", linewidth=0.8)
    loss_ax.spines[["top", "right"]].set_visible(False)
    loss_ax.legend(frameon=False, fontsize=8, loc="upper right")

    seen, ood = [], []
    for _, (_, _, result) in parsed.items():
        profile = result["seeds"][0]["depth_profile"]
        seen.append(profile["rungs"][0]["correct_examples"])
        ood.append(profile["ood_n_rungs"][0]["correct_examples"])
    positions = np.arange(len(ARMS))
    width = 0.34
    seen_bars = profile_ax.bar(
        positions - width / 2,
        seen,
        width,
        color="#276FBF",
        edgecolor="#173E69",
        label="Seen N",
    )
    ood_bars = profile_ax.bar(
        positions + width / 2,
        ood,
        width,
        color="#F4C78B",
        edgecolor="#8A5115",
        hatch="//",
        label="OOD N",
    )
    profile_ax.bar_label(seen_bars, padding=3, fontsize=9)
    profile_ax.bar_label(ood_bars, padding=3, fontsize=9)
    profile_ax.set_title("Exact T=1 profile", loc="left", weight="bold", y=1.10)
    profile_ax.text(
        0,
        1.025,
        "Correct examples out of 512; certification requires 512/512",
        transform=profile_ax.transAxes,
        color="#5B6573",
        fontsize=9,
    )
    profile_ax.set_ylabel("Correct examples")
    profile_ax.set_xticks(positions, ARMS.keys(), fontsize=8)
    profile_ax.set_ylim(0, max(seen + ood) + 1.8)
    profile_ax.grid(axis="y", color="#DDE2E8", linewidth=0.8)
    profile_ax.spines[["top", "right"]].set_visible(False)
    profile_ax.legend(frameon=False, fontsize=8, loc="upper left")

    fig.suptitle(
        "Canonical-register T=1 ablation",
        x=0.06,
        y=0.99,
        ha="left",
        fontsize=15,
        weight="bold",
        color="#202631",
    )
    fig.text(
        0.985,
        0.985,
        "✦ ONE LAYER DEEPER RESEARCH",
        ha="right",
        va="top",
        fontsize=8,
        color="#276FBF",
        weight="bold",
    )
    fig.text(
        0.06,
        -0.01,
        "Source: evaluator-owned public e5 reports. All arms use final labels only. No arm certified T=1.",
        fontsize=8.5,
        color="#5B6573",
    )
    fig.tight_layout(rect=(0.04, 0.06, 0.99, 0.86), w_pad=2.8)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(OUTPUT)


if __name__ == "__main__":
    main()

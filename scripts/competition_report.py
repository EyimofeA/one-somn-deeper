#!/usr/bin/env python3
"""Render evaluator-owned competition metrics as Markdown plus two SVG charts."""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path


def points(values: list[tuple[float, float]], width: int, height: int, pad: int) -> str:
    if not values:
        return ""
    maximum_x = max(value[0] for value in values) or 1.0
    return " ".join(
        f"{pad + (width - 2 * pad) * x / maximum_x:.1f},{height - pad - (height - 2 * pad) * y:.1f}"
        for x, y in values
    )


def svg_chart(title: str, series: list[tuple[str, list[tuple[float, float]], str]], x_label: str, path: Path) -> None:
    width, height, pad = 900, 360, 54
    legend = "".join(
        f'<text x="{pad + index * 210}" y="28" fill="{colour}" font-size="15">{escape(label)}</text>'
        for index, (label, _, colour) in enumerate(series)
    )
    polylines = "".join(
        f'<polyline fill="none" stroke="{colour}" stroke-width="3" points="{points(data, width, height, pad)}"/>'
        for _, data, colour in series if data
    )
    path.write_text(
        f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">
<rect width="100%" height="100%" fill="#10131a"/><text x="{pad}" y="{height - 14}" fill="#d7dde8" font-size="14">{escape(x_label)}</text>
<text x="8" y="{pad}" fill="#d7dde8" font-size="14">exact accuracy / normalized loss</text>
<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#778"/><line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#778"/>
{legend}{polylines}</svg>\n'''
    )


def percent(value: float) -> str:
    return f"{100 * value:.4f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir
    result = json.loads((run_dir / "result.json").read_text())
    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line]
    seed = result["seeds"][0]
    profile = seed["depth_profile"]
    train = [row for row in metrics if row["type"] == "train"]
    evaluations = {row["split"]: row for row in metrics if row["type"] == "evaluation"}
    rung_rows = profile["rungs"]
    ood_rung_rows = profile["ood_n_rungs"]
    report = [
        "# Competition-style run report",
        "",
        f"- Run: `{run_dir.name}`",
        f"- Completed training steps: {seed['completed_training_steps']}",
        f"- Training time: {seed['training_seconds']:.1f}s",
        f"- Mean final exact: {percent(result['score']['mean_exact_accuracy'])}",
        "",
        "## Final splits",
        "",
        "| Split | Exact | Loss |",
        "| --- | ---: | ---: |",
        *[f"| {name} | {percent(row['exact_accuracy'])} | {row['loss']:.3f} |" for name, row in evaluations.items()],
        "",
        "## Seen-N depth ladder",
        "",
        "| T | Exact | Correct / total | Certification |",
        "| ---: | ---: | ---: | --- |",
        *[f"| {row['time_steps']} | {percent(row['exact_accuracy'])} | {row['correct_examples']} / {row['example_count']} | {row['status']} |" for row in rung_rows],
        "",
        "## OOD-N depth ladder",
        "",
        "| T | Exact | Correct / total | Certification |",
        "| ---: | ---: | ---: | --- |",
        *[f"| {row['time_steps']} | {percent(row['exact_accuracy'])} | {row['correct_examples']} / {row['example_count']} | {row['status']} |" for row in ood_rung_rows],
        "",
        "## Training dynamics",
        "",
        f"The evaluator retained {len(train)} bounded training observations. See `training_curve.svg` and `depth_profile.svg`.",
        "",
    ]
    (run_dir / "competition_report.md").write_text("\n".join(report))
    max_loss = max((row["loss"] for row in train), default=1.0) or 1.0
    svg_chart(
        "Training dynamics",
        [
            ("loss", [(row["step"], row["loss"] / max_loss) for row in train], "#fbad3d"),
            ("train exact", [(row["step"], row["exact_accuracy"]) for row in train], "#5bd6b0"),
        ],
        "training step", run_dir / "training_curve.svg",
    )
    svg_chart(
        "Depth exact-match profile",
        [
            ("seen N", [(row["time_steps"], row["exact_accuracy"]) for row in rung_rows], "#6ea8fe"),
            ("OOD N", [(row["time_steps"], row["exact_accuracy"]) for row in ood_rung_rows], "#f2789f"),
        ],
        "T", run_dir / "depth_profile.svg",
    )


if __name__ == "__main__":
    main()

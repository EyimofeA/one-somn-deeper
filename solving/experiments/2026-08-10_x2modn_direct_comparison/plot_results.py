"""Render the direct MLP/Transformer generalization comparison as SVG.

Chart contract:
- Question: do direct models learn x2 mod N or memorize sparse final labels?
- Takeaway: both reach 100% train exact while unseen-N stays near 4%.
- Family: two-panel line plus dot/interval comparison.
- Data: 25 checkpoints for seed 0 and three final seeds per architecture.
- Palette: blue MLP, orange Transformer; solid train, dashed unseen.
- Output: 1200x640 static SVG for repository/chat delivery.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "diagnostics/artifacts/prime-a6eb7c97e54d4174a9b265674758a383/runs"
OUT = Path(__file__).with_name("direct_model_generalization.svg")
COLORS = {"MLP": "#2563EB", "Transformer": "#D97706"}


def metrics(architecture: str, seed: int):
    path = ARTIFACTS / f"2026-08-10_x2modn_direct_{architecture.lower()}" / f"seed{seed}"
    curve = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
    final = json.loads((path / "eval_report.json").read_text())
    return curve, final


def line_path(points):
    return " ".join(("M" if index == 0 else "L") + f" {x:.1f} {y:.1f}" for index, (x, y) in enumerate(points))


def main():
    curves, finals = {}, {}
    for label, directory in (("MLP", "mlp"), ("Transformer", "transformer")):
        curves[label], _ = metrics(directory, 0)
        finals[label] = [metrics(directory, seed)[1] for seed in range(3)]

    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="640" viewBox="0 0 1200 640">',
        '<rect width="1200" height="640" fill="#FFFFFF"/>',
        '<style>text{font-family:Inter,Arial,sans-serif;fill:#172033}.mono{font-family:ui-monospace,SFMono-Regular,monospace}.axis{stroke:#536071;stroke-width:1}.grid{stroke:#E5E7EB;stroke-width:1}.small{font-size:12px}.label{font-size:14px}.panel{font-size:17px;font-weight:700}.title{font-size:25px;font-weight:750}</style>',
        '<text x="60" y="42" class="title">Direct x² mod N model generalization</text>',
        '<text x="60" y="66" class="label" fill="#536071">Three-digit semiprimes · 185 train N · 41 exhaustive unseen test N · final labels only</text>',
        '<text x="70" y="102" class="panel">A. Exact accuracy during training (seed 0)</text>',
        '<text x="680" y="102" class="panel">B. Held-out exactness across three seeds</text>',
        '<text x="680" y="123" class="small" fill="#536071">Zoomed 0–6% scale; horizontal marks are means</text>',
    ]

    left, top, width, height = 70, 130, 530, 390
    for value in (0, 25, 50, 75, 100):
        y = top + height * (1 - value / 100)
        svg += [f'<line x1="{left}" y1="{y}" x2="{left+width}" y2="{y}" class="grid"/>', f'<text x="{left-12}" y="{y+4}" text-anchor="end" class="small mono">{value}%</text>']
    for step in (0, 3000, 6000, 9000, 12000):
        x = left + width * step / 12000
        svg += [f'<line x1="{x}" y1="{top+height}" x2="{x}" y2="{top+height+5}" class="axis"/>', f'<text x="{x}" y="{top+height+22}" text-anchor="middle" class="small mono">{step//1000}k</text>']
    svg += [f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+height}" class="axis"/>', f'<line x1="{left}" y1="{top+height}" x2="{left+width}" y2="{top+height}" class="axis"/>', f'<text x="{left+width/2}" y="{top+height+47}" text-anchor="middle" class="label">optimizer updates</text>']

    for label, records in curves.items():
        color = COLORS[label]
        train = [(left + width * row["step"] / 12000, top + height * (1 - row["train"]["exact_accuracy"])) for row in records]
        unseen = [(left + width * row["step"] / 12000, top + height * (1 - row["unseen_n_test"]["exact_accuracy"])) for row in records]
        svg += [f'<path d="{line_path(train)}" fill="none" stroke="{color}" stroke-width="2.7"/>', f'<path d="{line_path(unseen)}" fill="none" stroke="{color}" stroke-width="2.7" stroke-dasharray="7 5"/>']

    legend_y = 575
    for index, label in enumerate(("MLP", "Transformer")):
        x = 80 + index * 150
        svg += [f'<line x1="{x}" y1="{legend_y}" x2="{x+28}" y2="{legend_y}" stroke="{COLORS[label]}" stroke-width="3"/>', f'<text x="{x+36}" y="{legend_y+5}" class="label">{label}</text>']
    svg += ['<line x1="375" y1="575" x2="403" y2="575" stroke="#536071" stroke-width="3"/>', '<text x="411" y="580" class="label">train</text>', '<line x1="480" y1="575" x2="508" y2="575" stroke="#536071" stroke-width="3" stroke-dasharray="7 5"/>', '<text x="516" y="580" class="label">unseen N</text>']

    right, right_top, right_width, right_height = 680, 145, 460, 375
    for value in (0, 2, 4, 6):
        y = right_top + right_height * (1 - value / 6)
        svg += [f'<line x1="{right}" y1="{y}" x2="{right+right_width}" y2="{y}" class="grid"/>', f'<text x="{right-12}" y="{y+4}" text-anchor="end" class="small mono">{value}%</text>']
    svg += [f'<line x1="{right}" y1="{right_top}" x2="{right}" y2="{right_top+right_height}" class="axis"/>', f'<line x1="{right}" y1="{right_top+right_height}" x2="{right+right_width}" y2="{right_top+right_height}" class="axis"/>']

    categories = []
    for label in ("MLP", "Transformer"):
        categories += [(label, "seen N / new x", "seen_n_unseen_x"), (label, "unseen N", "unseen_n_test")]
    jitter = (-9, 0, 9)
    for index, (label, split_label, key) in enumerate(categories):
        x = right + 60 + index * 112
        values = [100 * report["metrics"][key]["exact_accuracy"] for report in finals[label]]
        mean = sum(values) / len(values)
        for offset, value in zip(jitter, values):
            y = right_top + right_height * (1 - value / 6)
            fill = COLORS[label] if key == "seen_n_unseen_x" else "#FFFFFF"
            svg.append(f'<circle cx="{x+offset}" cy="{y}" r="5" fill="{fill}" stroke="{COLORS[label]}" stroke-width="2"/>')
        mean_y = right_top + right_height * (1 - mean / 6)
        svg += [f'<line x1="{x-22}" y1="{mean_y}" x2="{x+22}" y2="{mean_y}" stroke="{COLORS[label]}" stroke-width="3"/>', f'<text x="{x}" y="{mean_y-11}" text-anchor="middle" class="small mono" font-weight="700">{mean:.2f}%</text>', f'<text x="{x}" y="{right_top+right_height+20}" text-anchor="middle" class="small">{escape(label)}</text>', f'<text x="{x}" y="{right_top+right_height+37}" text-anchor="middle" class="small" fill="#536071">{escape(split_label)}</text>']

    svg += ['<text x="60" y="624" class="small" fill="#6B7280">Source: frozen synthetic split; exact-match over 11,840 train rows and 21,404 exhaustive unseen-N test rows.</text>', '</svg>']
    OUT.write_text("\n".join(svg) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()

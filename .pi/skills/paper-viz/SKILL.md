---
name: paper-viz
description: Create publication-quality plots and visualizations for ML experiments. Covers T-extrapolation curves, loss landscapes, attention maps, metric dashboards, and wandb-style tracking. Use when the user wants to "plot something", "visualize metrics", "make a figure", or "see what's happening."
---

# Paper-Quality Visualization

## Philosophy

Every plot should tell one story. Before writing code, state: *what should this plot convince the reader of?*

## Quick plots (REPL)

For fast inspection, use `studio_repl_send` with matplotlib:

```python
import json, matplotlib.pyplot as plt

# Load metrics
with open("solving/experiments/metrics/some_run.jsonl") as f:
    data = [json.loads(line) for line in f]

# Plot
steps = [d["step"] for d in data]
loss = [d["loss"] for d in data]
plt.plot(steps, loss)
plt.xlabel("step")
plt.ylabel("loss")
plt.show()
```

## The T-extrapolation curve (frozen)

The frozen measurement script is `scripts/extrapolation_curve.py`. Do not regenerate it — it's read-only per RESEARCH_PROTOCOL.md §9.

To plot a new run's curve:
```bash
python scripts/extrapolation_curve.py solving/experiments/<card>/metrics.jsonl
```

## wandb-style dashboards

For multi-run comparisons, plot side-by-side or overlaid:

```python
import json, matplotlib.pyplot as plt
from pathlib import Path

runs = Path("solving/experiments/metrics").glob("*.jsonl")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

for path in runs:
    data = [json.loads(line) for line in open(path)]
    steps = [d["step"] for d in data]
    ax1.plot(steps, [d.get("loss", float("nan")) for d in data], label=path.stem, alpha=0.7)
    ax2.plot(steps, [d.get("exact_match", 0) for d in data], label=path.stem, alpha=0.7)

ax1.set_title("Loss"); ax1.set_xlabel("step"); ax1.legend(fontsize=6)
ax2.set_title("Exact Match"); ax2.set_xlabel("step")
plt.tight_layout()
plt.savefig("dashboard.png", dpi=150)
```

## Paper-quality style

For figures going into notes or submissions:

```python
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})
```

Use a consistent color palette:
```python
COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
```

## What to visualize (priority order)

1. **Loss curve** — train + eval on same axes
2. **Exact match %** — on eval set
3. **T-extrapolation** — accuracy vs problem size T
4. **OOD generalization** — fixed T vs varying T
5. **Attention/state maps** — if applicable

## Output

Always save figures to `solving/experiments/figures/<name>.png` and report the path. Use descriptive filenames: `depth_d32_k4_ut_t_curve.png`, not `fig1.png`.
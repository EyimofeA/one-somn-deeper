"""Plot training curves from one or more diagnostic runs.

Presentation figures (research-viz / Tufte): one message per panel, shared
scales across overlays, muted non-data ink, source + n in the footnote.
Never mixes loss and accuracy on one axis.

Usage:
    python plot_metrics.py runs/square_transformer
    python plot_metrics.py runs/square_transformer runs/mod_transformer
    python plot_metrics.py runs/                 # every child with metrics.jsonl
    python plot_metrics.py runs/square_transformer --only loss weights

Writes under <first-run>/plots/ (or --out):
    fig_overview.png     2×2 small multiples
    fig_loss.png
    fig_accuracy.png
    fig_weights.png      ‖θ‖₂ and Δ‖θ‖₂ as stacked multiples
    fig_optimizer.png    lr and ‖g‖₂ as stacked multiples
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Okabe–Ito
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]

PANELS = ("loss", "accuracy", "weights", "optimizer")
RunData = tuple[str, Path, list[dict], list[dict]]  # name, dir, train, eval


def apply_theme() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#d0d0d0",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#444444",
            "text.color": "#222222",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "lines.linewidth": 1.4,
        }
    )


def resolve_runs(paths: list[str]) -> list[Path]:
    runs: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_file() and p.name == "metrics.jsonl":
            runs.append(p.parent)
            continue
        if (p / "metrics.jsonl").is_file():
            runs.append(p)
            continue
        if p.is_dir():
            kids = sorted(
                c for c in p.iterdir() if c.is_dir() and (c / "metrics.jsonl").is_file()
            )
            if kids:
                runs.extend(kids)
                continue
        raise FileNotFoundError(f"no metrics.jsonl under {p}")
    seen: set[Path] = set()
    out: list[Path] = []
    for r in runs:
        r = r.resolve()
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def load_metrics(run_dir: Path) -> tuple[list[dict], list[dict]]:
    train, eval_rows = [], []
    with (run_dir / "metrics.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("type") == "train":
                train.append(row)
            elif row.get("type") == "eval":
                eval_rows.append(row)
    return train, eval_rows


def series(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for r in rows:
        if key in r and r[key] is not None:
            xs.append(r["step"])
            ys.append(r[key])
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def footnote(fig: Figure, runs: list[RunData]) -> None:
    bits = []
    for name, rd, train, ev in runs:
        try:
            src = (rd / "metrics.jsonl").relative_to(Path.cwd())
        except ValueError:
            src = rd / "metrics.jsonl"
        bits.append(f"{name}: n_train={len(train)} n_eval={len(ev)} · {src}")
    fig.text(
        0.01,
        0.005,
        "  ·  ".join(bits),
        ha="left",
        va="bottom",
        fontsize=6.5,
        color="#555555",
    )


def shared_xlim(runs: list[RunData]) -> tuple[float, float] | None:
    xs = []
    for _, _, train, ev in runs:
        for rows in (train, ev):
            for r in rows:
                if "step" in r:
                    xs.append(r["step"])
    if not xs:
        return None
    lo, hi = min(xs), max(xs)
    pad = max(1.0, 0.02 * (hi - lo))
    return lo - pad, hi + pad


def empty_msg(ax: Axes, msg: str) -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, color="#666666")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(False)


def draw_loss(ax: Axes, runs: list[RunData]) -> None:
    plotted = False
    for i, (name, _, train, _) in enumerate(runs):
        x, y = series(train, "loss")
        if not len(x):
            continue
        ax.plot(x, y, color=PALETTE[i % len(PALETTE)], label=name)
        plotted = True
    if not plotted:
        empty_msg(ax, "no train loss")
        return
    ax.set_title("Train cross-entropy loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_ylim(bottom=0.0)
    if len(runs) > 1:
        ax.legend(frameon=False, loc="upper right")
    elif len(runs) == 1:
        ax.annotate(runs[0][0], xy=(0.98, 0.95), xycoords="axes fraction",
                    ha="right", va="top", fontsize=8, color=PALETTE[0])


def draw_accuracy(ax: Axes, runs: list[RunData]) -> None:
    plotted = False
    for i, (name, _, train, ev) in enumerate(runs):
        c = PALETTE[i % len(PALETTE)]
        x, y = series(train, "exact_accuracy")
        if len(x):
            ax.plot(x, y, color=c, linestyle="-", label=f"{name} train exact")
            plotted = True
        x, y = series(train, "token_accuracy")
        if len(x):
            ax.plot(x, y, color=c, linestyle=":", label=f"{name} train token")
            plotted = True
        x, y = series(ev, "exact_accuracy")
        if len(x):
            ax.plot(x, y, color=c, linestyle="--", marker="o", markersize=3.5,
                    markevery=max(1, len(x) // 12), label=f"{name} val exact")
            plotted = True
    if not plotted:
        empty_msg(ax, "no accuracy fields")
        return
    ax.set_title("Exact-match and token accuracy")
    ax.set_xlabel("step")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, loc="lower right", ncols=1)


def draw_weights(fig: Figure, axes: list[Axes], runs: list[RunData]) -> None:
    """Two stacked small multiples: ‖θ‖₂ and |Δ‖θ‖₂| between log steps."""
    ax_n, ax_d = axes
    any_n = any_d = False
    for i, (name, _, train, _) in enumerate(runs):
        c = PALETTE[i % len(PALETTE)]
        x, y = series(train, "weight_norm")
        if len(x):
            ax_n.plot(x, y, color=c, label=name)
            any_n = True
        x, y = series(train, "weight_update")
        if len(x):
            ax_d.plot(x, y, color=c, label=name)
            any_d = True
    if not any_n:
        empty_msg(ax_n, "no weight_norm (re-train)")
    else:
        ax_n.set_title("Parameter L2 norm ‖θ‖₂")
        ax_n.set_ylabel("‖θ‖₂")
        ax_n.tick_params(labelbottom=False)
        if len(runs) > 1:
            ax_n.legend(frameon=False, loc="best")
    if not any_d:
        empty_msg(ax_d, "no weight_update (re-train)")
    else:
        ax_d.set_title("‖θ‖₂ change between log steps")
        ax_d.set_xlabel("step")
        ax_d.set_ylabel("Δ‖θ‖₂")
        ax_d.set_ylim(bottom=0.0)
    xlim = shared_xlim(runs)
    if xlim:
        ax_n.set_xlim(*xlim)
        ax_d.set_xlim(*xlim)


def draw_optimizer(fig: Figure, axes: list[Axes], runs: list[RunData]) -> None:
    ax_lr, ax_g = axes
    any_lr = any_g = False
    for i, (name, _, train, _) in enumerate(runs):
        c = PALETTE[i % len(PALETTE)]
        x, y = series(train, "lr")
        if len(x):
            ax_lr.plot(x, y, color=c, label=name)
            any_lr = True
        x, y = series(train, "grad_norm")
        if len(x):
            ax_g.plot(x, y, color=c, label=name)
            any_g = True
    if not any_lr:
        empty_msg(ax_lr, "no lr (re-train)")
    else:
        ax_lr.set_title("Learning rate schedule")
        ax_lr.set_ylabel("lr")
        ax_lr.set_ylim(bottom=0.0)
        ax_lr.tick_params(labelbottom=False)
        if len(runs) > 1:
            ax_lr.legend(frameon=False, loc="best")
    if not any_g:
        empty_msg(ax_g, "no grad_norm (re-train)")
    else:
        ax_g.set_title("Gradient L2 norm (pre-clip)")
        ax_g.set_xlabel("step")
        ax_g.set_ylabel("‖g‖₂")
        ax_g.set_ylim(bottom=0.0)
    xlim = shared_xlim(runs)
    if xlim:
        ax_lr.set_xlim(*xlim)
        ax_g.set_xlim(*xlim)


def save_fig_loss(runs: list[RunData], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.6), constrained_layout=True)
    draw_loss(ax, runs)
    xlim = shared_xlim(runs)
    if xlim:
        ax.set_xlim(*xlim)
    footnote(fig, runs)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_fig_accuracy(runs: list[RunData], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.6), constrained_layout=True)
    draw_accuracy(ax, runs)
    xlim = shared_xlim(runs)
    if xlim:
        ax.set_xlim(*xlim)
    footnote(fig, runs)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_fig_weights(runs: list[RunData], out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.0), sharex=True, constrained_layout=True)
    draw_weights(fig, list(axes), runs)
    footnote(fig, runs)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_fig_optimizer(runs: list[RunData], out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.0), sharex=True, constrained_layout=True)
    draw_optimizer(fig, list(axes), runs)
    footnote(fig, runs)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_fig_overview(runs: list[RunData], out: Path) -> None:
    """2×2 small multiples; weights/optimizer each get an internal stack."""
    names = " vs ".join(r[0] for r in runs)
    fig = plt.figure(figsize=(11.0, 8.2), constrained_layout=True)
    fig.suptitle(f"Diagnostic training curves — {names}", fontsize=11, y=1.01)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.35])

    ax_loss = fig.add_subplot(gs[0, 0])
    ax_acc = fig.add_subplot(gs[0, 1])
    draw_loss(ax_loss, runs)
    draw_accuracy(ax_acc, runs)
    xlim = shared_xlim(runs)
    if xlim:
        ax_loss.set_xlim(*xlim)
        ax_acc.set_xlim(*xlim)

    gs_w = gs[1, 0].subgridspec(2, 1, hspace=0.25)
    ax_wn = fig.add_subplot(gs_w[0])
    ax_wd = fig.add_subplot(gs_w[1], sharex=ax_wn)
    draw_weights(fig, [ax_wn, ax_wd], runs)

    gs_o = gs[1, 1].subgridspec(2, 1, hspace=0.25)
    ax_lr = fig.add_subplot(gs_o[0])
    ax_g = fig.add_subplot(gs_o[1], sharex=ax_lr)
    draw_optimizer(fig, [ax_lr, ax_g], runs)

    footnote(fig, runs)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


SAVERS = {
    "loss": save_fig_loss,
    "accuracy": save_fig_accuracy,
    "weights": save_fig_weights,
    "optimizer": save_fig_optimizer,
}


def main() -> None:
    apply_theme()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="run dir(s), metrics.jsonl, or a parent of runs/")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: <first-run>/plots)")
    ap.add_argument("--only", nargs="+", choices=PANELS, default=list(PANELS),
                    help="which separate panels to write")
    ap.add_argument("--no-combined", action="store_true", help="skip fig_overview.png")
    ap.add_argument("--no-separate", action="store_true", help="skip per-panel PNGs")
    args = ap.parse_args()

    run_dirs = resolve_runs(args.paths)
    runs: list[RunData] = []
    for rd in run_dirs:
        train, eval_rows = load_metrics(rd)
        if not train and not eval_rows:
            print(f"warning: empty metrics in {rd}")
            continue
        runs.append((rd.name, rd, train, eval_rows))
    if not runs:
        raise SystemExit("no usable metrics found")

    out_dir = args.out or (run_dirs[0] / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    if not args.no_combined:
        path = out_dir / "fig_overview.png"
        save_fig_overview(runs, path)
        written.append(path)
    if not args.no_separate:
        for panel in args.only:
            path = out_dir / f"fig_{panel}.png"
            SAVERS[panel](runs, path)
            written.append(path)

    for p in written:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()

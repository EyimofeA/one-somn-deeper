# Modular-Squaring Diagnostic Suite

Offline research benchmark for `(N, x) -> x^2 mod N`. Separates three possible
failure modes: decimal squaring, modular reduction, and composing the two.
This is **not** a competition submission — it shares no code or weights with
`competition/` or `solving/`, and nothing here should be copied into a
submission file. See `../solving/experiments/predictions.md` (2026-07-24) for
the isolated-mechanism experiments that motivated this suite.

## Setup

```bash
cd diagnostics
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python numpy pyyaml pytest matplotlib
```

CPU only by design — every model here is small (d_model=128, 2-4 layers).

## 1. Generate data

```bash
# fast, tiny (hundreds of rows) — for the smoke test and local iteration
.venv/bin/python -m data.generate --task all --out data/generated --scale small

# full scale (100k train / 10k val / 10k per diagnostic split, per spec)
.venv/bin/python -m data.generate --task all --out data/generated --scale full

# a single task only
.venv/bin/python -m data.generate --task square --out data/generated/square --scale full

# reversed-digit-order ablation (write to a separate directory)
.venv/bin/python -m data.generate --task square --out data/generated_reversed/square \
    --scale full --reverse-digits
```

Each task directory gets these split files (task B/C/D also get
`heldout_modulus`, `heldout_factor`, `length12`, `length13`):

| File | Split | What it tests |
|---|---|---|
| `train.jsonl` | training data | — |
| `val_iid.jsonl` | IID prompt split | trainability only |
| `heldout_x.jsonl` (A/C/D) or `heldout_u.jsonl` (B) | held-out operand | unseen-operand generalization, familiar modulus |
| `heldout_modulus.jsonl` | held-out modulus | genuine modular-arithmetic learning vs. memorization |
| `heldout_factor.jsonl` | held-out prime factor | stricter than held-out modulus |
| `length12.jsonl` / `length13.jsonl` | length extrapolation | train on 10-11 bit N, test on 12/13-bit — **do not average with IID** |
| `hard.jsonl` | hard arithmetic | long carry chains, near-boundary remainders, x near N/sqrt(N) |

Task A (`square`) has no modulus, so it only has `train` / `val_iid` /
`heldout_x` / `hard`.

## 2. Train each task

Recommended order (per spec): A, then B, then C from random init, then
compare C against D, then multitask only after that.

```bash
# Task A: squaring, standard Transformer baseline
.venv/bin/python train.py configs/square.yaml

# Task B: modular reduction
.venv/bin/python train.py configs/mod.yaml

# Task C: fused modular squaring
.venv/bin/python train.py configs/square_mod.yaml

# Task D: fused with intermediate (auxiliary) supervision, diagnosis only
.venv/bin/python train.py configs/square_mod_trace.yaml
```

Swap in the recurrent workspace model (the main architecture) with an
override — no config file edits needed:

```bash
.venv/bin/python train.py configs/square_mod.yaml \
    --override model.type=recurrent_workspace \
    --override model.workspace_size=8 \
    --override model.num_output_slots=4 \
    --override model.num_loops=12
```

Any config value can be overridden the same way, e.g.
`--override optim.lr=1e-3 --override optim.epochs=40`. Checkpoints
(`peak.pt`, the best-val_iid checkpoint, and `final.pt`) and `metrics.jsonl`
land in `out_dir` (default `runs/<task>_<model>`); `config_used.yaml` is
written alongside them so `evaluate.py` can rebuild the exact model.

### Plot curves

`metrics.jsonl` records loss, exact/token accuracy, lr, pre-clip grad norm,
weight L2, and Δweight L2 each `log_every` steps. Plot any run (or overlay
several, or everything under `runs/`):

```bash
.venv/bin/python plot_metrics.py runs/square_transformer
.venv/bin/python plot_metrics.py runs/square_transformer runs/mod_transformer
.venv/bin/python plot_metrics.py runs/                    # all child runs
.venv/bin/python plot_metrics.py runs/square_transformer --only loss optimizer
```

Writes presentation PNGs under `<run>/plots/` (or `--out DIR`):

| File | Message |
|---|---|
| `fig_overview.png` | 2×2 small multiples of the panels below |
| `fig_loss.png` | train cross-entropy only (y includes 0) |
| `fig_accuracy.png` | train/val exact + token accuracy (y in [0, 1]) |
| `fig_weights.png` | stacked: ‖θ‖₂ and Δ‖θ‖₂ (no dual axis) |
| `fig_optimizer.png` | stacked: lr schedule and pre-clip ‖g‖₂ |

Footnote on each figure lists run name, n train/eval points, and the
`metrics.jsonl` path. Re-train once after this logging change so
weight/optimizer fields exist; older jsonl still plots loss/accuracy.


## 3. Evaluate every split

```bash
.venv/bin/python evaluate.py runs/square_transformer \
    --data data/generated/square --splits val_iid heldout_x hard

.venv/bin/python evaluate.py runs/square_mod_transformer \
    --data data/generated/square_mod \
    --splits val_iid heldout_x heldout_modulus heldout_factor length12 length13 hard
```

Writes `<run_dir>/eval_report.json` with, per split: exact-match, token
accuracy, and (wherever the field applies) exact-match stratified by
modulus bit length, x/output digit length, carry-chain length, quotient
size, remainder bucket, seen-vs-unseen modulus, and — for the recurrent
workspace model only — accuracy at every recurrence depth up to
`num_loops` (`by_recurrence_depth`), from one trained checkpoint (the
transition is weight-tied, so running fewer loops at eval time is valid).

**Task C vs. A x B** (does composition cost more than the component error
rates predict — the accuracy-compounding question):

```bash
.venv/bin/python -c "
from evaluate import compare_product
import json
a = json.load(open('runs/square_transformer/eval_report.json'))['splits']['heldout_x']['exact_match']
b = json.load(open('runs/mod_transformer/eval_report.json'))['splits']['heldout_modulus']['exact_match']
c = json.load(open('runs/square_mod_transformer/eval_report.json'))['splits']['heldout_modulus']['exact_match']
compare_product(a, b, c)
"
```

## 4. CPU smoke test

```bash
.venv/bin/python -m pytest tests/ -q
```

30 tests, ~4s on CPU: dataset generation (dedup, zero modulus/factor
overlap, labels match plain Python arithmetic), split-building (prime pools,
coprime sampling, overlap assertions actually catch overlap), tokenization
(shapes, round-trip, no answer leakage into `<OUT>` tokens), and both model
architectures (forward+backward shapes, gradients finite, a 16-example
dataset overfits to >=90% exact-match, `override_loops` actually changes the
recurrent model's output).

To reproduce the full small-scale pipeline end to end in under a minute:

```bash
.venv/bin/python -m data.generate --task all --out data/generated --scale small
.venv/bin/python train.py configs/square.yaml --override optim.epochs=40 \
    --override log_every=6 --override eval_every=6
.venv/bin/python evaluate.py runs/square_transformer --data data/generated/square \
    --splits val_iid heldout_x hard
```

(`--scale small` has only ~400 train rows, so don't expect real
generalization signal from it — it exists to prove the pipeline runs, not to
produce a result. Use `--scale full` for anything you intend to draw a
conclusion from.)

## Layout

```
diagnostics/
  data/
    tokens.py      vocabulary + fixed-width digit encoders for Tasks A-D
    splits.py       prime pools, semiprime construction, coprime sampling, the 6 split builders
    generate.py     CLI: writes train/val/diagnostic-split jsonl per task
    dataset.py      jsonl -> torch Dataset (padding, attention mask, IGNORE_INDEX labels)
  models/
    transformer.py          Baseline 1: standard bidirectional Transformer (control)
    recurrent_workspace.py  Baseline 2: immutable context + weight-tied recurrent workspace (main architecture)
  train.py          generic trainer, reads a yaml config + --override key=value
  plot_metrics.py   loss / accuracy / weight / optimizer curves from metrics.jsonl
  evaluate.py        stratified evaluation, writes eval_report.json; compare_product() for the C-vs-A*B question
  configs/           one yaml per task, model.type selects the baseline
  tests/             pytest: generator, splits, tokenization, model shapes/smoke
```

## Notes on scope

- No `%`/closed-form solver is used anywhere in `models/` — labels are
  computed with plain Python `%` only inside `data/generate.py` (data
  generation, not part of any model's forward pass).
- Digit widths are fixed globally (N,x: 4 digits; u, x^2: 8 digits; mod
  result: 4 digits) so tensor shapes never depend on which split produced a
  row, including the 12/13-bit length-extrapolation splits.
- `--scale full`'s held-out-modulus / held-out-factor / length-extrapolation
  splits draw from a finite pool of small primes (needed to keep 10-13 bit
  semiprimes enumerable); if you raise the requested split sizes high enough
  to exhaust that pool, `data/splits.py` raises `ValueError` rather than
  silently reusing a modulus or prime across train/test.

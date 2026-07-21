# What a scored run returns

## You get

From `one-layer submit … --wait` / `one-layer metrics <id>`:

| Field | Where | Meaning |
|-------|--------|---------|
| `score` | CLI summary | = `mean_exact_accuracy` (Easy: avg test+ood) |
| `type=train` rows | JSONL | every `log_every` steps: `step`, `loss`, `exact_accuracy` (batch), `elapsed_seconds` |
| `type=evaluation` | JSONL | per split `test` / `ood`: `loss`, `exact_accuracy` |
| `type=summary` | JSONL | `completed_steps`, `training_seconds`, `mean_exact_accuracy` |
| job page | onelayerdeeper.ai | status UI |

Saved under `solving/experiments/metrics/*.jsonl`.

## You do **not** get

- Model weights / checkpoints
- LR schedule traces, grad norms, weight histograms
- Per-example predictions
- Optimizer-state plots

So **no native “optim plots.”** We only plot what is in the JSONL (loss, exact, steps, eval splits). Scheduler behavior is inferred from code + step count, not logged.

## After every experiment — read in order

1. This file (what came back)
2. `solving/experiments/metrics/<run>.jsonl` — raw facts
3. `solving/experiments/figures/PLOTS_INDEX.md` — open PNGs in IDE
4. `solving/experiments/EXPERIMENT_LOOP.md` — card row
5. `solving/RESEARCH_LOG.md` — append-only narrative

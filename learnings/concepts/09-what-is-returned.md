# What a scored run returns

Pinned to upstream evaluator behavior @ `79f0a09` (2026-07-24).

## You get

From `one-layer submit … --wait` / `one-layer metrics <id>`:

| Field | Where | Meaning |
|-------|--------|---------|
| `score` / `mean_exact_accuracy` | CLI + JSONL summary | Easy/Medium **ranking** score = mean exact accuracy over fixed eval splits. Still computed on Hard, but **does not rank** Hard. |
| `depth_profile.max_certified_time_steps` | RESULT_JSON / UI **Max T** | Largest T∈{1,2,4,8,16,32,64} with consecutive 100%-exact rungs on seen-modulus depth profile. Hard primary rank key. Diagnostic on Easy/Medium. |
| `depth_profile.ood_n_max_certified_time_steps` | RESULT_JSON / UI **OOD N Max T** | Same certification on unseen-modulus depth profile. Hard tie-break. Diagnostic on Easy/Medium. |
| `type=train` rows | JSONL | every `log_every` steps: `step`, `loss`, `exact_accuracy` (batch), `elapsed_seconds` |
| `type=evaluation` rows | JSONL | per ordinary split (`test` / `ood` / …): `loss`, `exact_accuracy` |
| `type=summary` | JSONL | `completed_steps`, `training_seconds`, `mean_exact_accuracy` |
| job page | onelayerdeeper.ai | status UI; Hard LB sorts Max T → OOD N Max T → earlier time |

Saved under `solving/experiments/metrics/*.jsonl` when we download them.

## You do **not** get

- Model weights / checkpoints
- Whatever you stuffed in **`auxiliary`** — aux is only for your custom `training_loss` during the run; it is **not** logged in metrics JSONL
- LR schedule traces, grad norms, weight histograms
- Per-example predictions / intermediate step strings
- Optimizer-state plots
- On Hard: public exact-% as the leaderboard number (those measurements stay private diagnostics)

So: you “return” `(logits, aux)` every forward, but the API only reports **loss + exact + steps + eval splits + depth certification**. Aux is invisible after the job unless you fold it into the scalar loss (which still only shows up as that one loss number).

## After every experiment — read in order

1. This file (what came back)
2. `solving/experiments/metrics/<run>.jsonl` — raw facts
3. `solving/experiments/figures/PLOTS_INDEX.md` — open PNGs in IDE
4. `solving/experiments/EXPERIMENT_LOOP.md` — card row
5. `solving/RESEARCH_LOG.md` — append-only narrative

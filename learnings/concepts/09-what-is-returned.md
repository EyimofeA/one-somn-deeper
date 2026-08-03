# What a scored run returns

Pinned to upstream evaluator behavior @ `8a3c78d` (2026-08-03).

## You get

From `one-layer submit … --wait` / `one-layer metrics <id>` / `one-layer leaderboard`:

| Field | Where | Meaning |
|-------|--------|---------|
| `score` / `mean_exact_accuracy` | CLI + JSONL summary | Easy/Medium **ranking** score = mean exact accuracy over fixed eval splits. Still computed on Hard, but **does not rank** Hard. |
| `depth_profile.max_certified_time_steps` | RESULT_JSON / UI **Max T** | Largest T∈{1,2,4,8,16,32,64} with consecutive 100%-exact rungs on seen-modulus depth profile. Hard primary rank key. Diagnostic on Easy/Medium. |
| `depth_profile.ood_n_max_certified_time_steps` | RESULT_JSON / UI **OOD N Max T** | Same certification on unseen-modulus depth profile. Hard second key. Diagnostic on Easy/Medium. |
| `seen_tiebreak_accuracy_percent` / `ood_n_tiebreak_accuracy_percent` | leaderboard / status | Exact accuracy at the first *uncertified* rung on each profile. Hard third key (UI shows 4 decimals; ranking uses full precision). |
| `type=train` rows | JSONL | every `log_every` steps: `step`, `loss`, `exact_accuracy` (batch), `elapsed_seconds` |
| `type=evaluation` rows | JSONL | per ordinary split (`test` / `ood` / …): `loss`, `exact_accuracy` |
| `type=summary` | JSONL | `completed_steps`, `training_seconds`, `mean_exact_accuracy` |
| job page | onelayerdeeper.ai | status UI; Hard LB sorts Max T → OOD N Max T → next-rung accuracies → earlier time |

Saved under `solving/experiments/metrics/*.jsonl` when we download them.

## Custom loss callbacks (Rule 12)

Mutually exclusive on `Submission`:

- **`training_loss(logits, labels, auxiliary)`** — legacy. Only *valid* tokens, flattened to `[valid_tokens, vocab]` / `[valid_tokens]`.
- **`token_training_loss(batch: TokenLossBatch)`** — sequence-aware. `logits` / `labels` / `valid_mask` keep `[batch, target_length, …]` boundaries; `target_positions` is set for separate-output tasks and `None` for causal; ignore pads via `valid_mask`.

Both must return one differentiable finite scalar; the evaluator owns backward. Existing cards that only set `training_loss` remain legal.

## You do **not** get

- Model weights / checkpoints
- Whatever you stuffed in **`auxiliary`** — aux is only for your custom loss during the run; it is **not** logged in metrics JSONL
- LR schedule traces, grad norms, weight histograms
- Per-example predictions / intermediate step strings
- Optimizer-state plots
- On Hard: public exact-% as the leaderboard number (those measurements stay private diagnostics except the published next-rung tie-break)

So: you “return” `(logits, aux)` every forward, but the API only reports **loss + exact + steps + eval splits + depth certification (+ next-rung tie-break on Hard LB)**. Aux is invisible after the job unless you fold it into the scalar loss (which still only shows up as that one loss number).

## After every experiment — read in order

1. This file (what came back)
2. `solving/experiments/metrics/<run>.jsonl` — raw facts
3. `solving/experiments/figures/PLOTS_INDEX.md` — open PNGs in IDE
4. `solving/experiments/EXPERIMENT_LOOP.md` — card row
5. `solving/RESEARCH_LOG.md` — append-only narrative

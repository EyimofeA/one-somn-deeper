# Experiment registry

Append-only. Each entry is written before a run starts (hypothesis/changed
variable/controls/predicted outcome) and filled in after (result/
interpretation/next decision). Never edit a filled-in entry in place --
add a new entry if a run is repeated or corrected, and link back with
`[[id]]`. Raw outputs for each ID live under `runs/<id>/` (gitignored;
`eval_report.json`/`learning_curve.jsonl`/`run_config.json` are the
committed summary of record via `analysis_out/`).

Status values: `planned` (not yet run, GPU not started as of 2026-07-25) |
`running` | `done`.

## Task B serial Phase 1: input-conditioned workspace initialization — Codex

- **Status:** done (2026-08-03; serial L40 queue, source commit `3cde93d`)
- **Prediction:** a one-time ordered-input read will improve fixed-N=1349
  held-out-u exact match, particularly for `q >= 10`; the shuffled-context
  control will remain near the fixed-workspace result.
- **Changed variable:** `workspace_init_mode`: fixed learned registers versus
  one cross-attention read from the encoded ordered input. The read reuses the
  recurrent transition's existing cross-attention parameters, so parameter
  count and K=8 tied transition are unchanged. `shuffled_context` changes only
  that read's source through a deterministic non-identity row-level derangement.
- **Controls held fixed:** existing 8k/2k N=1349 split, tokenizer, d=144,
  K=8, eight workspace tokens, optimizer, batch size 512, 50k step budget,
  decoder, no auxiliary loss.
- **Conditions/seeds:** correct input context and shuffled context, seeds 0, 1,
  2; compare against the completed fixed-workspace and deep controls.
- **Pre-registered interpretation:** correct > fixed and shuffled supports
  semantic initialization. Correct ≈ shuffled implies an optimization-path
  effect. Neither improves closes fixed workspace initialization as the main
  explanation.
- **Result:** correct input context reached held-out-u exact match
  **39.22±3.33%** across seeds (35.45%, 40.45%, 41.75%); the matched
  shuffled-context control reached **14.67±4.43%** (18.65%, 15.45%, 9.90%).
  This clears the fixed-workspace final baseline (33.65±2.95%) by 5.57 points.
  Raw reports, curves, and per-run metadata:
  `analysis_out/task_b_workspace_init_phase1/`.
- **Interpretation:** **confirmed.** Correct > fixed and correct >> shuffled
  supports semantic initialization; the gain cannot be explained by the extra
  encoder pass or initialization attention alone. The predicted `q >= 10`
  comparison was not emitted: this evaluator records relative
  small/mid/large quotient buckets, so that subprediction remains untested.
- **Next decision:** input-conditioned initialization is now the recurrent
  reference. Before changing K or capacity, add the missing absolute-quotient
  slice to evaluation and compare the retained checkpoints by that slice.

**Deviation applied to all of A1/A2/A3/B2 (2026-07-25, explicit user request):**
`batch_size` raised from the established 64 to **512** for every run below,
and all runs launched concurrently on one GPU rather than sequentially.
Measured throughput at batch=64/512/2048 on this box: 5,188 / 20,049 /
29,416 examples/sec -- 512 was chosen as a real utilization gain (~4x)
without the steep diminishing returns seen at 2048 (+46% over 512 for 4x
the batch). Learning rate was **not** re-tuned for the larger batch (no
linear/sqrt scaling applied) -- explicitly out of scope for this pass per
the user ("don't care about reproducibility like that, want to be fast").
This means: (a) none of the runs below are directly comparable to the
original 15-run ablation's batch=64 numbers (`carry`/`both`/`diagonal`/
`baseline` at 71.4%/80.1%/11.1%/0.75% final exact match) without accounting
for the batch-size change, and (b) convergence may be somewhat suppressed
relative to what a batch-matched LR would give, since a fixed LR typically
under-drives a larger batch. Each experiment below is still internally
valid (all conditions within one experiment share the same batch size), and
A1 in particular is entirely self-contained (compares its own step-50k
checkpoint against its own step-100k checkpoint within the same run, not
against the original ablation's numbers).

---

## A1-carry: carry-aux, seed 1, single continuous run to 100k steps

- **Status:** planned
- **Hypothesis:** the remaining Task A error under carry-aux supervision
  (val exact ~75.6% at 50k, seed 1, the best of the 3 original carry seeds)
  is caused mainly by insufficient training time, not a hard architectural
  ceiling.
- **Changed variable:** optimizer steps only (50,000 -> 100,000).
- **Deviation from the original plan, documented:** originally scoped as
  "resume `runs/aux_ablation/carry_seed1/peak.pt`", but the GPU box that
  produced it was torn down and only its JSON summaries (not `.pt` weights)
  were pulled before that happened -- the checkpoint no longer exists.
  Retraining seed=1 from step 0 straight through 100,000 steps (one
  continuous cosine schedule, 5% warmup of 100k) is used instead. This
  is not a compromise -- it avoids the fresh-Adam-state discontinuity and
  schedule-splicing judgment call a true resume would have needed, so the
  100k number is actually cleaner to interpret than a resumed run would
  have been, at the cost of repeating the first 50k steps of compute.
- **Controls held fixed:** data (`data/generated/square`), architecture
  (d_model=128, 4 layers, 4 heads, d_ff=512), optimizer (AdamW, lr=3e-4,
  wd=0.01), batch_size=64, aux loss weight (carry, fixed 1.0, not annealed),
  eval cadence (every 1000 steps, fixed 2000-example train subset),
  evaluation code (`train_aux_ablation.py`'s `evaluate_full` /
  `per_position_and_bucket_analysis`, unchanged), seed=1.
- **Predicted outcome:** [to be written by a human before the run, per
  RESEARCH_PROTOCOL -- not pre-filled by the agent]
- **Result:** [pending]
- **Interpretation:** [pending]
- **Next decision:** [pending]

## A1-both: both-aux, seed 1, single continuous run to 100k steps

- **Status:** planned
- **Hypothesis:** same as A1-carry, for the combined carry+diagonal
  condition (val exact 0.6981/0.8948/0.8111 at 50k across the 3 original
  seeds -> seed 1 chosen, 89.5%, the best).
- **Changed variable / controls / deviation:** identical to A1-carry
  (same checkpoint-loss situation, same fix), seed=1, both aux weights
  fixed at 1.0 throughout (not annealed -- this is the "both" condition,
  not "both_annealed").
- **Predicted outcome:** [pending -- human-written]
- **Result:** [pending]
- **Interpretation:** [pending]
- **Next decision:** [pending]

## A2: shuffled-carry-label control

- **Status:** planned
- **Hypothesis:** if the carry-aux gain (0.75% -> 71.4%) comes from genuine
  carry information, training with a fixed per-example derangement of the
  carry targets (same shape/marginal distribution, guaranteed not to match
  the true target for any example) should perform close to baseline
  (~0.75%), not close to genuine carry-aux (~71%).
- **Changed variable:** carry auxiliary target only -- replaced with
  `carry_target[permutation[i]]` for a fixed derangement `permutation`
  drawn once per seed before training starts (not reshuffled per epoch/step).
- **Controls held fixed:** backbone, carry head, aux loss weight (1.0,
  fixed), optimizer, 50,000-step budget, parameter count, training example
  set (same x's, same order structure) -- identical to the `carry`
  condition in every respect except which carry values are attached to
  which row.
- **Validation:** unchanged -- val exact-match/token-accuracy is always
  computed on the REAL digit-prediction task; only the auxiliary training
  target is shuffled, and only during training.
- **Seeds:** 3, each with its own fixed derangement.
- **Predicted outcome:** [pending -- human-written]
- **Result:** [pending]
- **Interpretation rule (pre-registered):** shuffled-carry near baseline
  (~1-5%) -> confirms genuine carry information caused the original gain.
  Shuffled-carry still strong (double-digit+, closer to 71% than to 1%) ->
  the gain is NOT specifically about carry semantics -- investigate generic
  regularization/loss-scaling effects or an implementation leak (e.g. the
  permutation accidentally correlating with position/magnitude, or the aux
  loss acting as an unintended auxiliary regularizer regardless of target
  content).
- **Next decision:** [pending]

## A3: modest scale check (d_model=256, 6 layers)

- **Status:** planned
- **Hypothesis:** once supervision (carry+diagonal aux) is fixed, a modest
  capacity increase does NOT close much further of the remaining gap --
  i.e. the bottleneck sections 5-8 of the error-analysis report identified
  is a supervision/precision problem, not a raw-capacity problem, so scaling
  without also fixing supervision (baseline condition) should show only a
  small improvement, and scaling WITH supervision (both condition) should
  show a small further improvement on top of the already-large aux gain,
  not a qualitative jump to near-100%.
- **Changed variable:** d_model 128->256, n_layers 4->6. n_heads set to 8
  (keeps head_dim=32, matching the original 128/4=32); d_ff set to 1024
  (keeps the 4x d_model ratio the original 512/128=4x used).
- **Conditions x seeds:** {baseline, both} x 2 seeds = 4 runs (not a broad
  sweep -- exactly the two conditions needed to answer the question above).
- **Controls held fixed:** data, optimizer hyperparameters (lr/wd/warmup/
  grad_clip), batch_size=64, 50,000-step budget, aux loss weight (both
  condition, fixed 1.0), evaluation code.
- **Predicted outcome:** [pending -- human-written]
- **Result:** [pending]
- **Interpretation rule (pre-registered):** modest scaling causes a very
  large, clean improvement (per the Task A stopping rule) -> do NOT freeze
  Task A, scaling is the missing piece. Small/no improvement on top of the
  aux gain -> confirms supervision, not capacity, was the bottleneck;
  proceed to freeze Task A per the stopping rule.
- **Next decision:** [pending]

---

## B1: Task B (mod) pipeline audit + 32-example memorization test

- **Status:** planned (memorization test can run locally/CPU immediately,
  does not need GPU -- everything else in B1 is a code-reading audit, also
  GPU-independent)
- **Hypothesis:** the existing `mod` task generator/dataset/train/evaluate
  pipeline (already used for the original `mod_transformer` 20-epoch run,
  final val_iid exact 27.6%) is mechanically correct -- i.e. Task B's low
  score reflects genuine task difficulty, not a tokenization/masking/
  leakage bug.
- **Changed variable:** none -- this is a verification pass, not a training
  change.
- **Predicted outcome:** [pending -- human-written]
- **Result:** [pending]
- **Interpretation:** [pending]
- **Next decision:** [pending]

## B2: Task B standard baseline (1 seed, then 3 if B1 passes)

- **Status:** planned
- **Hypothesis:** [to be written after B1 confirms the pipeline is sound --
  no point predicting an outcome for a pipeline not yet verified]
- **Changed variable:** none vs. the existing `configs/mod.yaml` recipe
  (d_model=128, 4 layers, 4 heads, d_ff=512, AdamW lr=3e-4 wd=0.01, cosine
  5% warmup, grad_clip=1.0, batch_size=64, 50,000 steps) -- this re-run
  exists to get a clean, current-pipeline number with the full
  reproducibility metadata (git commit/exact command/env) this registry
  requires, since the original `mod_transformer` run predates that
  infrastructure.
- **Predicted outcome:** [pending]
- **Result:** [pending]
- **Interpretation:** [pending]
- **Next decision:** [pending]

## B3: Task B error analysis (arithmetic-difficulty features + baselines)

- **Status:** planned (depends on B2)
- **Hypothesis:** [to be written after B2 lands -- error analysis should
  follow observed failure shape, not be predicted blind]
- **Changed variable:** none -- analysis only, no training change.
- **Predicted outcome:** [pending]
- **Result:** [pending]
- **Interpretation:** [pending]
- **Next decision:** [pending]

## B4: Task B competing hypotheses (written after B2/B3)

- **Status:** planned (blocked on B2/B3 results)
- Not a run -- a written comparison of >=3 competing failure explanations
  (quotient estimation / borrow-chain / comparison-near-multiples /
  output-decoding / capacity / training-duration / dataset shortcut), each
  with supporting observations, opposing observations, a discriminating
  experiment, and a falsifying result. Explicitly gates any auxiliary-loss
  implementation for Task B until this is written.

## B5: Task B decision gate (written after B4)

- **Status:** planned (blocked on B4)
- Chooses one of: (1) move to Task C, (2) targeted auxiliary-loss ablation
  on Task B's actual failure pattern, (3) test training duration/scale,
  (4) repair the pipeline/data construction. Explicitly must NOT copy the
  Task A carry intervention onto Task B without deriving it from Task B's
  own failure pattern.

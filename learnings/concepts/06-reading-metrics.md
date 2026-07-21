# Reading competition metrics

## What stops training?

Easy manifests set `total_training_time_seconds: 60` and `max_steps: 1_000_000`.  
Our e1 runs all report `training_seconds: 60.1` with only ~260–290 steps → **the clock killed training**, not convergence and not early stop from a scheduler.

## What “train exact_accuracy” is

JSONL `type=train` rows are **logged training batches** (every `log_every=100` steps), not a full pass over the train set. So “train exact 21%” ≠ “model gets 21% of the training distribution.” It is a noisy batch signal that was **still climbing** when the wall hit.

## Loss

Same rows include `loss`. We always have loss curves if we plot them; they were just not in the first figure pack.

## OOD split

`type=evaluation`, `split=ood` comes from the **evaluator’s dataset** (alongside `test`). We did not invent it for scored runs. Local `solving/experiments/data_samples/` is only for eyeballing rows; it is unrelated to the hosted `ood` metric.

For Easy, score ≈ average of test and merged OOD (see problem page). Our e1 OOD exact was **0%** for b0/b1/b2.

## Baseline recovery

`b0_transformer` is a faithful fork of official `baseline_adamw`. There is no public Easy leaderboard number to match — Easy scores are private. Recovering the baseline means: same arch+AdamW, scored under the same e1 suite. We did that (1.00% mean).

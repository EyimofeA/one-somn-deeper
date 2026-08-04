# Competition run logging

Every local Easy, Medium, or Hard candidate uses:

```text
runs/<submission_name>/
  config.json        command provenance
  git_commit.txt     exact research/source revision
  evaluator_commit.txt pinned competition-evaluator revision
  manifest.json      evaluator configuration
  submission.py      submitted source snapshot
  train.log          evaluator-owned training and final evaluation messages
  metrics.jsonl      evaluator-owned bounded train/evaluation event history
  result.json        full final evaluator result and depth profiles
  gpu.txt / gpu.csv  external local utilization sample, when available
  checkpoints/       user-owned only; never copied into a submission
  summary.md         start/end time
```

`scripts/run_competition_logged.sh` is intentionally a process wrapper. It does
not patch the benchmark runner, modify batches, add evaluator calls, alter
validation timing, or read evaluator-owned data.

Run directories are intentionally Git-ignored: they can contain large copied
submission snapshots, telemetry, and checkpoints. Their source revision and
location belong in the corresponding committed report instead.

For a remote mirror rather than a Git checkout, set `RESEARCH_COMMIT` to the
source revision before invoking the wrapper. The wrapper then records that
immutable revision rather than guessing from the remote working directory.

## What the official runner records

- Bounded training checkpoints: step, wall time, total loss, batch exact
  (`--include-structured-metrics`; capped by the evaluator at 256 records).
- Final test/OOD loss and exact accuracy.
- Depth and OOD-N ladders, including certified Max T where configured.
- Completed steps, wall-clock training time, model-state count, and batch sizes.

## What requires external local telemetry

The public runner does not expose learning rate, gradient/parameter norms,
activations, or periodic validation. GPU utilization and throughput are sampled
outside the evaluator only for local runs. Do not add hooks to an official
submission to write these metrics.

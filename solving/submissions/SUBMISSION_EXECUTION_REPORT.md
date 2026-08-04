# Submission execution report — 2026-08-05

## Scope and rule snapshot

This is a packaging and selection pass, not a claim that the scientific
diagnostic is already a competition solver. The evaluator is pinned locally to
live commit `e32c2f9`. A submission is one standalone `submission.py`; it may
use learned recurrence, but may not inspect/augment evaluator data, run a
task-specific solver, load weights, or use a participant-owned training loop.

Budgets are 60 seconds (Easy), 600 seconds (Medium), and 3,600 seconds (Hard)
on hosted H100s; evaluation gets half the corresponding training budget. Easy
and Medium rank by mean exact accuracy. Hard ranks by the consecutive,
all-example-exact `T=1,2,4,8,16,32,64` prefix, then OOD-N prefix.

## Candidate matrix

| Tier | Candidate source | Architecture | Local evidence | Decision |
| --- | --- | --- | --- | --- |
| Easy | `submissions/fable_tcap_adamw/submission.py` | 1,595,904-state tied register with train-time random effective depth | e1: 0.67% test, 8.00% OOD, 4.33% mean; no T=1 certificate | Promoted for one hosted Easy attempt |
| Medium | `experiments/2026-08-04_deadline/hard_fable_v2/submission.py` | 2,142,474-state tied digital register | no Medium run; same source gets 0.00% mean on e1 | Packaged baseline only; no 600-second run or submission justified |
| Hard | `experiments/2026-08-04_deadline/hard_fable_v2/submission.py` | same Fable register | prior hosted H1 `602bf7f1-eab7-46c2-91e8-e4a4a010f9d7`: 0.0467% mean, no certified T=1; current e1 is 0.00% | Rejected; preserve the sole daily Hard attempt |

Both files pass `one-layer validate`. “Packaged baseline” means a self-contained
source is available at that path, not that it earns a submission attempt.

## Current recurrent VDF result and its boundary

The scientifically strongest code is
`diagnostics/train_recurrent_vdf_square_reduce.py`, artifact
`diagnostics/artifacts/somn-l40-2026-08-05/recurrent_vdf_reducer_square_trace_support/seed0/`.
It uses tied learned Square, Comparator, and LSD-first Subtractor modules.
With VDF-square-generated intermediate reduction labels in a controlled small
regime, held-out reduction reaches 95.56% and VDF T=8 reaches 89.02%.

It is **not packaged as a competition submission**. Its diagnostic training
constructs internal square/reduction traces that the public evaluator does not
supply; recreating those traces inside an official run would violate the
no-data-augmentation/no-task-solver rule. Its remaining one-step error also
cannot be hidden by a wrapper. The necessary bridge is an end-to-end,
final-label-only implementation that retains the same learned transition—not a
precomputed-square reducer.

## Logged local validations

| Run | Hardware | Train dynamics | Final result | Artifact |
| --- | --- | --- | --- | --- |
| `easy_serial_recurrent_e1_current_retry1` | L40 | loss 17.732 → 1.424; batch exact 0.2% → 14.1%; 506 updates / 60.07 s | test 0.00%, OOD 1.00%, mean 0.50% | `runs/easy_serial_recurrent_e1_current_retry1/` |
| `fable_v2_e1_current` | L40 | loss 2.974 → 0.130; batch exact 0.0% → 92.8%; 417 updates / 60.10 s | test 0.00%, OOD 0.00%, mean 0.00% | `runs/fable_v2_e1_current/` |
| `fable_tcap_adamw_e1_control` | L40 | loss 2.884 → 1.842; batch exact 0.4% → 1.6%; 941 updates / 60.02 s | test 0.67%, OOD 8.00%, mean 4.33% | `runs/fable_tcap_adamw_e1_control/` |

Fable v2 is overfit: its training exactness is 92.8% while both held-out splits
are exactly zero. The recurrent-transfer candidate also improves training fit
without gaining reliable held-out exactness. In contrast, the T-cap control
underfits its training batches but generalizes enough to justify one Easy-only
attempt; this is a selection result, not evidence of robust recurrence.

## Logging contract

All future local candidates must run through `scripts/run_competition_logged.sh`
from the evaluator checkout. The script enables the evaluator’s bounded
structured metrics and creates:

```text
runs/<name>/{config.json,git_commit.txt,evaluator_commit.txt,manifest.json,
             submission.py,train.log,metrics.jsonl,result.json,gpu.csv,
             checkpoints/,summary.md}
```

Runs are intentionally Git-ignored; reports name their paths and commits.
The wrapper changes neither evaluator batching nor scoring. It adds only an
external GPU-utilization sample and source snapshots.

## Next execution gate

Before any new hosted attempt, run a final-label-only recurrent VDF candidate
locally through this logging path. Submit only if it exceeds the applicable
historical reference on held-out exact accuracy *and* shows nonzero T=1
certification locally. Until that gate, use the L40 to improve the existing
Square→Reduce transition with legal final-label training rather than spending
Medium/Hard attempts.

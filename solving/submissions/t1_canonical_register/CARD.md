# T=1 canonical register

## Frozen Hard source

- File: [`submission.py`](submission.py)
- SHA-1: `5b622f06680600f4b346e34b635b839dde18471c`
- Evaluator source: `e32c2f9`
- Model: 1,592,064 persistent elements on public e5
- Optimizer: AdamW, peak LR 3e-3, batch 512
- Hard job: `7714d650-78a4-4d4a-8fc1-a384914d7658` (completed 2026-08-08)

The mutable LSD-first register is initialized from `x` once. Every tied cell
application then sees only the current register and immutable `N`; requested
`T` controls loop count only. Answer logits are the state logits. Embeddings
use an explicit small initialization rather than PyTorch's large default.
Active register width is derived conservatively from `ModelSpec.max_seq_len`.

No arithmetic trace, oracle, answer lookup, generated label, custom derivative,
or hard-coded numeric operation is used.

## Evidence

Exact-source hosted Easy e5:

| Job | Updates | Mean exact | seen-N T=1 | OOD-N T=1 |
|---|---:|---:|---:|---:|
| `b99d4e4b-95f3-420c-ba08-282e4060d4d0` | 1,036 | 0.7083% | 2/512 | 3/512 |
| `e0542460-87d9-49ef-aaa7-a691ac378414` | 1,189 | 0.8333% | 5/512 | 4/512 |

Local L40S e5 seed 74 was 0.4583% mean with 11/512 seen-N and 1/512
OOD-N T=1. The wider fixed-16-slot parent was nonzero across three local seeds,
but its exact hosted run had zero OOD-N T=1.

## Limitation and selection rule

This is **not** a promoted or solved transition. On full-budget public Medium
m1 (T=4/8/16 training only), the canonical parent stayed at CE 2.2930 for
9,815 updates and scored 0/192 seen-N plus 0/512 OOD-N at T=1. Final labels at
composed depths did not identify the one-step cell.

The owner explicitly requested one Hard submission after the sprint. This card
is selected as a forced first-rung lottery because it is the only exact hosted
source tested today that produced nonzero T=1 on both seen and OOD-N profiles
twice. Hard results must not be described as mechanism validation unless they
materially exceed these chance-scale counts.

Hard upload validation rechecked the source filename/size policy and exact
SHA-1 immediately before submission. The service accepted it for `h1` and
reported zero Hard attempts remaining for the UTC day.

Hard result: 0.0500% mean exact; 8/9,999 test, 2/10,002 OOD-T, and 5/10,002
OOD-N; no certified rung; 0/768 seen-N T=1 and 0/768 OOD-N T=1. The run
completed 163,274 updates in 3,600.01 seconds with final train loss 2.17846.
This refutes transfer of the hosted Easy first-rung counts.

Selection figure:
[`../../experiments/figures/t1_hard_selection_2026-08-08.png`](../../experiments/figures/t1_hard_selection_2026-08-08.png).

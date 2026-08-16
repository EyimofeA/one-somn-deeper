# Full fused T=1 at fixed N=403

## Question

Can the complete binary work-state machine learn `x squared mod N` when
modulus variation is removed?

Unlike the preceding reduction-only diagnostic, the model receives `x` bits,
not exact `x squared` bits. It receives no square label, quotient, comparison,
carry, subtraction, or execution trace. Only final residue bits supervise the
44 tied recurrent updates.

## Setup

- fixed `N=403`, the same modulus used by Easy E10 but with a new synthetic
  split;
- deterministic split of all 403 legal x values: 282 train, 60 validation,
  and 61 untouched audit;
- 128 channels, 443,777 parameters, 44 tied updates, 9% recurrent dropout;
- constant AdamW `3e-4`, batch 512, 10,000 steps / 5.12M sampled examples;
- validation exactness alone selects the checkpoint.

## Result

| Measurement | Exact accuracy |
|---|---:|
| Train at step 1,000 | 95.39% |
| Train at step 1,250 | **100.00%** |
| Train at step 10,000 | **100.00%** |
| Peak validation across the run | **1/60 = 1.67%** |
| Final validation | **0/60 = 0.00%** |
| Selected-checkpoint audit | **1/61 = 1.64%** |

Validation was already 1/60 at step 1. It returned to the same value once at
step 9,750 but never exceeded it, so strict validation selection retained step
1. The selected-checkpoint train score is consequently 0%; this does not
contradict the curve's final 100% training fit. The untouched audit is a check
of the chance checkpoint, not the final memorizing state.

Training loss fell from 0.70 to `1.39e-4`. There was no delayed validation rise
after interpolation: validation stayed at zero for nearly the entire interval
from step 500 through step 10,000.

## Decision

The fixed-N experiment solves optimization but fails function learning. The
machine can store all 282 training mappings, yet it does not infer the mapping
for unseen x values under the same modulus. Therefore:

- more training on this exact split is low-value;
- perfect training accuracy is not evidence that the architecture learned
  squaring or reduction;
- fixed N alone is too weak a curriculum because it permits a lookup shortcut;
- the next full-machine experiment must constrain shortcuts, for example by
  requiring the same x representation to satisfy multiple moduli, while
  retaining AdamW and final-label-only supervision.

This result does not refute the direct supervised squarer. It refutes the claim
that terminal modular labels on one small fixed modulus identify that squarer
inside the fused machine.

Figure:
[`../../figures/binary_workstate_fused_fixed_n403_2026-08-16.png`](../../figures/binary_workstate_fused_fixed_n403_2026-08-16.png).

Ignored verified backup:
`diagnostics/artifacts/prime-7072f85e48094888bcf3893db897ea54/binary-workstate-fused-fixed-n403-2026-08-16/`.

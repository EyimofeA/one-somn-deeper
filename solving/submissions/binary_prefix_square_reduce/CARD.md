# Learned square plus streaming reducer — hosted gate

Date: 2026-08-17

## Candidate

- Source: [`submission.py`](submission.py)
- SHA-1 before upload: `0e8975b2d795cb377b1b8f393a650fa32011dbae`
- Representation: exact evaluator-input decimal to 11-bit binary conversion
- Learned transition: a randomly initialized tied Neural-GPU squarer followed
  by a randomly initialized 22-stage prefix reducer
- Training signal: evaluator final residue labels, plus the same labels applied
  to the square head only on T=1 rows whose input bounds prove reduction is the
  identity
- Optimizer: flattened Muon at 0.006 with 250-update warmup for matrices;
  AdamW at 3e-4 for vectors

The prior version's first curriculum phase returned only the auxiliary square
loss, which gave the reducer exactly zero gradient. This candidate keeps final
residue BCE active from update one. No pretrained weights, exact square,
remainder, quotient, comparison, carry, or intermediate arithmetic trace are
provided.

## Gate order

1. Easy E5: varying N, 11-bit scale, T in 1/2/3.
2. Medium M6: fixed N=1517, 11-bit scale, T in 1/2/4.
3. Hard H1: owner explicitly authorized the unchanged file after both practice
   runs; Hard may use another recurrence, so the square-specific bias is a
   known transfer risk.

The complete prediction was registered before the Easy upload in
[`../../experiments/predictions.md`](../../experiments/predictions.md).

## Hosted results

| Tier | Job | Updates | Final train loss | Mean exact | Certified T |
|---|---|---:|---:|---:|---:|
| Easy E5 | `b08f3e88-96cd-41d1-986b-bec26b23d17a` | 398 | 0.690 | 0.00% | none |
| Medium M6 | `a4403938-7b64-40db-8f52-e350c1255a02` | 4,122 | 0.697 | 0.00% | none |
| Hard H1 | `caff39c2-ed89-4834-84d9-34997dd1eabd` | running | — | — | — |

E5 test/OOD losses were 36.982/36.839. M6 test/OOD losses were
36.583/36.275. M6 briefly reported 3.1% exact on one training batch at step
4,000, but neither final training nor either evaluation split had an exact
row. The registered prediction failed. Keeping the final residue path live did
not make the factorization identifiable, and ten times more fixed-N training
did not cause a delayed transition. The owner explicitly authorized the final
unchanged Hard upload despite these failed gates; treat it as a forced
research submission, not a promotion.

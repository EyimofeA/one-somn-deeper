# Submission selection refresh

## Answer first

The exact canonical register lost its previously nonzero public OOD-N T=1
signal on a third hosted Easy e5 replication. The exact Fable T-cap/AdamW
source reproduced its chance-scale Medium m5 score but again had zero T=1 on
both profiles. Neither source learned the operator. For the owner's scheduled
forced Hard attempt, Fable remains selected only because its family owns the
best historical hosted Hard mean and the strongest direct Easy/Medium scores.

## Exact-source evidence

| Source | Job | Tier | Mean | Seen-N T=1 | OOD-N T=1 | Decision |
|---|---|---|---:|---:|---:|---|
| Canonical `5b622f0` | `cfb0fc73` | Easy e5 | 0.5000% | 2/512 | 0/512 | dual-profile repeat refuted |
| Fable `aa75819` | `233cbce0` | Medium m5 | 0.1556% | 0/768 | 0/768 | chance baseline confirmed |

Canonical completed 1,569 updates. Fable completed 13,672 updates and scored
13/9,000 test plus 5/3,000 OOD. Neither certified a rung. Exact hosted details
come from `one-layer status <job> --json`; the prediction and outcome are in
[`../predictions.md`](../predictions.md).

## Hard selection

Selected file: [`../../submissions/fable_tcap_adamw/submission.py`](../../submissions/fable_tcap_adamw/submission.py)

- SHA-1: `aa75819a878fab6c03c6a23d979f6234560f6e3d`
- Size: 9,071 bytes
- Validation: `one-layer validate` passed on 2026-08-11.
- Static audit: no task-value modulo, `pow`, `sympy`, `gmpy`, `torch.load`, or
  custom backward; AdamW owns `model.parameters()` and model-state assertion is
  present.
- Expected result: 0.02%--0.08%, no rung, 0/768 on both T=1 profiles.

This is quota execution, not scientific promotion. The next research session
must return to legal unseen-N T=1 identifiability rather than tune this source.

## Submission

At 00:45 WAT, after confirming no active job and rechecking validation and
SHA-1, the service accepted Hard job
`05f53719-7717-4923-88d5-a3cafe373167`. It reported zero Hard attempts left
for the UTC day. The source was not changed after the Easy/Medium refresh.

# Latent-state VDF second-opinion packet

Prepared by Codex, 2026-08-05.

## Question

The current hypothesis is that recurrence is not itself refuted. Instead, the
competition model may fail because it reuses answer-aligned **token positions**
as the evolving algorithmic state. The next possible architecture would use a
genuine latent state:

`h0 = Encode(N, x)`; then `h(t+1) = F_theta(h(t), N)` exactly T times; finally
`Decode(h(T))` emits the answer. T is only an execution count, never a learned
quantity or phase-specific weight selector.

The reviewer should attack this hypothesis. In particular, distinguish a bad
token-register workspace from a bad transition cell, insufficient supervision,
ordinary optimization failure, or a broader failure of learned recurrence.

## Included files

- `vdf_final_label_submission.py`: current tied SquareCell -> ReduceCell
  final-label-only candidate. It is the current prompt-tail-register design.
- `vdf_t_curriculum_submission.py`: same candidate with a staged final-label
  T=1 -> T<=2 -> T<=3 loss experiment.
- `train_vdf_trace_ablation.py`: diagnostic-only intermediate-state-loss
  ablation; generated trace labels are not competition legal.
- `train_vdf_depth_curriculum.py`: research-only genuine final-label
  T=1 -> T<=2 -> T<=4 curriculum, with held-out-x / unseen-N probes.
- `direct_transformer_baseline.py`: direct final-output control used in the
  architecture audit.
- `reports/`: evaluator-owned per-T reports and curves, plus both diagnostic
  JSON reports.
- `RESEARCH_LOG.md`, `STATUS.md`, and `predictions.md`: source-of-truth
  chronology, current status, and preregistered predictions.

## Key results

| Condition | Final held-out result |
| --- | --- |
| Tied VDF, final labels | e1 3.33% test / 0% OOD; no certified depth rung |
| Final-label e1 curriculum | T=1 5.26%, T>=2 0%; 3.33% test / 0% OOD |
| Trace-label ablation | 24.2% in-batch fit but 0% held-out final exact T=1/2/3 |
| Genuine final-label curriculum | near-100% phase training fit, 0% held-out x at T=1/2/4 |
| Direct Transformer control | 2.00% test / 1.00% OOD; no certified depth rung |

The actual successful *diagnostic* serial reducer work is described in
`RESEARCH_LOG.md`: it used a real evolving serial GRU state and learned
comparator/subtractor primitives. It is evidence that learned recurrence can
work in a structured controlled setting, but it is not a solved competition
submission.

## Constraints

Do not use handwritten modular arithmetic, lookup, or solver shortcuts in a
competition forward pass. Research diagnostics may generate labels externally,
but must explicitly label that fact. The primary task is repeated modular
squaring: `(N, x, T) -> x^(2^T) mod N`.

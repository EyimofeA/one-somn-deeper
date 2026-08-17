# Evaluator information-flow audit

## Question

Could the T=64 leaderboard result plausibly come from an accidental legal
information channel, rather than from a learned recurrent transition?

## Static findings

1. All public Easy and Medium generation commands set
   `separate_input_output=true`.
2. The model is called with only `input_ids` and `attention_mask`. Labels and
   target positions are consumed by the evaluator only after the forward call.
3. A custom token loss can read labels during training, as documented, but its
   `auxiliary` input is produced by the model. Evaluation calls the same model
   without a loss callback and never supplies labels to it.
4. Evaluation snapshots persistent model-state versions and rejects mutation.
   Thus an evaluation-time table-building channel through buffers is closed.
5. The Hard metric recorder is evaluator-owned and explicitly forbidden as an
   attack surface.

## Decision

No honest answer-token leakage path was found. This does not prove there is no
implementation bug, but it removes the most obvious explanation. Continue the
legitimate reconstruction branch: a learned, weight-tied recurrent transition
with strong global communication and exact discrete state.

Do not implement or submit a loophole. Revisit this audit only if new public
evidence contradicts one of the five findings.

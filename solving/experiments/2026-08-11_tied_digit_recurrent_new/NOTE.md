# Tied digit recurrent new candidate

Prediction registered before execution: Easy e1 should execute without a rules/runtime failure and achieve non-random token learning, but exact-match may remain low. Hard T=1 is expected to remain below 5%, with a small chance of a nonzero profile if the local mixer learns decimal carry structure. The causal change versus prior candidates is a compact tied Transformer transition with a depthwise digit-local path and late-step deep supervision. It is intended as a genuinely new probe, not a rerun of a prior source.

First Easy attempt `8a854963-dd8e-4921-82ad-b2f4edb9a76e` reached the evaluator but failed generically. Before the second attempt, the only repair was to accept both supported attention-mask ranks and set explicit conservative batch sizes; this creates a new source SHA.

Second Easy attempt `a920aaf0-9bf5-4723-b64a-e13ca254a4aa` also failed with `EVALUATION_FAILED` before producing metrics. Under the predeclared gate, this source was not submitted to Hard. Final tested SHA-1: `ac181fd0e7d3d24e46673809866865c519691837`.

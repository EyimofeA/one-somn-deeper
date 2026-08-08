# Exact Hard-source local e5 replication

No-change replication of the exact source uploaded as Hard job
`9e7404cb-b0c9-480a-aa64-8d90cc853d67`.

- Source: `solving/submissions/exact_match_optimizer/submission.py`
- SHA-1: `8c796bf39f3b0d2f90043b08430be26c23f0f180`
- Public manifest: `benchmark/manifests/h100_easy_e5.json`
- Device: live Prime Intellect NVIDIA L40
- Hidden Hard inputs are not available locally and are not inspected.

Prediction is registered in [`../predictions.md`](../predictions.md). Raw
runner output is preserved under the ignored
`diagnostics/artifacts/hard_exact_source_e5_replication_2026-08-08/` tree.

## Result

- 1,192 updates in 60.03 seconds; 1,600,513 model-state elements.
- Test: 7/1,200 = 0.5833%; OOD: 4/600 = 0.6667%; mean: **0.6250%**.
- Seen-N T=1: 4/512; OOD-N T=1: 1/512.
- No seen-N or OOD-N rung certified.

The prediction is confirmed: this is within 0.1667 points of the prior local
0.4583% run and retains nonzero T=1 signal on both profiles. It verifies the
uploaded source and local evaluator path, but does not establish a reliable
one-step transition.

## Lineage

This is a direct descendant of `fable_tcap_adamw`: the recurrent `StepBlock`,
prompt-derived T, training T-cap 16, evaluation ceiling 64, straight-through
token state, and AdamW wall-clock schedule come from that card. The GPT-5 Pro
`exact_match_optimizer` child added opposite-orientation output heads, the
sequence/worst-digit/margin/agreement loss, evaluator-owned two-pass SAM, and
high-loss batch reuse. The Hard attempt then changed only normalized 8x T=1
row weighting. It is not derived from `t1_assassin`, the diffusion refiner, or
the later research-only pair-fold tape.

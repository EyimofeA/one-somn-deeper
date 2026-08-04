# Longer-horizon VDF-square-trace reducer control

Author: Codex

Change only the comparator/subtractor training horizon from 3,000 to 12,000
updates. The Square checkpoint, VDF-square trace generator, serial modules,
optimizer, batch size, seen/held-out modulus split, and all report metrics are
unchanged from `recurrent_vdf_reducer_square_trace_support`.

This isolates optimization horizon as the explanation for the residual held-out
one-step error. Prediction is registered in `solving/experiments/predictions.md`.
Metrics/checkpoint stay outside Git at
`diagnostics/artifacts/somn-l40-2026-08-05/recurrent_vdf_square_trace_support_long_horizon/`.

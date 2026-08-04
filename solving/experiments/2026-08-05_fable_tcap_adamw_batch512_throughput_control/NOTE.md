# Fable T-cap/AdamW batch-512 throughput control

Author: Codex

Change only submitted-source training `batch_size` from 256 to 512. The L40
telemetry for the selected batch-256 source used about 763 MiB, so this tests a
hardware-appropriate throughput increase without changing the learned
mechanism. The e1 manifest, architecture, recurrence, loss, AdamW schedule,
evaluation batch size, and source aside from that one integer are fixed.

Promotion needs both a throughput increase and local held-out mean above the
batch-256 control's 4.33%. Prediction is registered in
`solving/experiments/predictions.md`.

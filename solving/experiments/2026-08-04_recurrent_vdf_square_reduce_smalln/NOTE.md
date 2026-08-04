# Recurrent VDF square→reduce diagnostic

Complete two-digit-semiprime square→reduce composition gate. A q=0 target-label
bug was recovered from the first remote artifact and repaired before training;
the architecture was otherwise unchanged. Held-out-N square is 100% exact,
but reduction after that exact square is 46.96% and T=8 is 33.88%. See the
append-only research log for component and quotient-bucket metrics.

Source: `diagnostics/train_recurrent_vdf_square_reduce.py`.

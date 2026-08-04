# Recurrent VDF square→reduce diagnostic

Complete two-digit-semiprime square→reduce composition gate. A q=0 target-label
bug was recovered from the first remote artifact and repaired before training;
the architecture was otherwise unchanged. Held-out-N square is 100% exact,
but reduction after that exact square is 46.96% and T=8 is 33.88%. See the
append-only research log for component and quotient-bucket metrics.

Source: `diagnostics/train_recurrent_vdf_square_reduce.py`.

Follow-up card `recurrent_vdf_reducer_square_trace_support` changes only reducer
training rows to true `s²` reduction traces while retaining the square
checkpoint. It restores held-out reduction to 95.56% and T=8 to 89.02%; see the
append-only log for the distinct artifact and comparison.

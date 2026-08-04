# Final-label-only recurrent VDF submission candidate

Author: Codex

This is the legal transfer of today’s diagnostic mechanism. Each outer VDF
iteration applies two distinct learned modules with tied weights over time:

```text
register -- learned SquareCell --> latent square -- learned ReduceCell --> register
```

Both cells use bidirectional prompt interaction followed by an LSD-to-MSD GRU
scan. The state is a learned discrete token register at the answer-aligned tail
of the prompt. Only final evaluator labels train it. In particular, it does
not construct `x²`, quotient, comparison, subtraction, residue, or diagnostic
trace labels in the submitted forward/training path.

This card changes architecture from the Fable control to the explicit VDF
decomposition. Prediction is registered in `predictions.md`; local artifacts
belong in `runs/vdf_square_reduce_final_label_e1/`.

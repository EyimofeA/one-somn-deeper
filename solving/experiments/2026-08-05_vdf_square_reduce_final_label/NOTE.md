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

## First local result

The literal first version is not a useful speed baseline: despite Easy inputs
having only T=1/2/3, it executed 64 masked outer cells per forward. It completed
only 64 updates in 60 seconds and reached 2.67% test / 0% OOD (1.33% mean).
The next card changes only that execution loop to exactly the prompt-selected
maximum T, keeping all learned computation unchanged.

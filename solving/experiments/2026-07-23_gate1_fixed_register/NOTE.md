# gate1_fixed_register

**Author:** Codex
**Implementation:** Boyle (`implement_gate1_square`)

**CHANGE:** pack each row's valid prompt tokens into the right edge of a fixed
`spec.max_seq_len` internal register before the unchanged Transformer and RoPE.
Generator, data, splits, optimizer, blocks, parameters, and runtime
hyperparameters are frozen from `gate1_aligned_products`.

At width 9, prompt lengths 1–4 digits occupy internal slots 3–8, 2–8, 1–8,
and 0–8 respectively. In a padded mixed batch, the attention mask determines
each row's valid tokens; tensor width does not. Fixed-register PAD slots are
masked as attention keys. Valid logits are gathered back into their original
row positions, and padded output positions are zero, preserving the evaluator
shape. This adds no parameters; the model remains at 51,136 state elements.

**PREDICTION:** agent-authored under the human's explicit supervision override;
recorded in [`../predictions.md`](../predictions.md).

The model ran for 1,000 fixed optimizer steps on the A6000 in 44.6 seconds.
The final train batch reached 99.22% exact match; held-out same-length reached
99.50%; length-4 OOD reached 6.70%. OOD loss rose to 2.73 while the same-length
loss fell to 0.00285.

**RESULT:** refuted, classified by Codex under the human's override.

**Interpretation (Codex):** Stabilizing the global prompt/output slot grid did
not improve the frozen baseline's 7.3% length-4 result. Batch-dependent slot
geometry is not the causal bottleneck in this diagnostic.

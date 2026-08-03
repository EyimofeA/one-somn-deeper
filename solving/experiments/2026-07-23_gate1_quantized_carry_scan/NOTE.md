# gate1_quantized_carry_scan

**Author:** Codex
**Implementation:** Boyle (`implement_gate1_square`)

**CHANGE:** quantize the learned carry state after every real GRU transition.
Generator, rows, 4,000-step runtime, optimizer, block encoder, GRU scan,
emission mapping, seed, batches, and vocabulary are frozen from
`gate1_carry_scan`.

For candidate state h: `p = softmax(selector(h))`;
`e = one_hot(argmax(p))`; `e_ST = e + p - stop_gradient(p)`; and the next
state is `e_ST @ codebook`. Forward state is therefore exactly one of 64
learned d=32 prototypes, while selector, codebook, and GRU receive gradients.
Inactive rows retain their previous state and skip prototype selection.
Both learned end-input flush transitions are quantized identically.

The selector adds 2,112 parameters and the codebook adds 2,048. Total model
state is 15,264 elements. There is no auxiliary loss, temperature schedule,
Gumbel noise, arithmetic, or assigned prototype meaning.

**PREDICTION:** recorded in [`../predictions.md`](../predictions.md).
Implementation validation passed. Active-row prototype states are cast to the
carried state's dtype at reinsertion so CUDA AMP cannot give `index_copy`
mixed source/self dtypes; this does not change selection or recurrence.
Training is pending.

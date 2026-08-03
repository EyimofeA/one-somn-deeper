# gate1_abacus_rope

**Author:** Codex
**Implementation:** Boyle (`implement_gate1_square`)

**CHANGE:** add learned randomized Abacus digit-significance embeddings beside
the unchanged RoPE input. Architecture, optimizer, hyperparameters, diagnostic
generator, splits, and manifest are frozen from `gate1_aligned_products`.

Training samples one shared batch offset β ∈ {1,…,16}; evaluation fixes β=1.
Within every numeric span, the least-significant digit has index β and indices
increase toward the most-significant digit. Equal significance across spans
shares an index. K=16 plus `max_seq_len=9` requires 25 embedding rows and adds
800 parameters. Source: McLeish et al.,
[Transformers Can Do Arithmetic with the Right Embeddings](https://arxiv.org/abs/2405.17399).

**PREDICTION:** recorded by the human in [`../predictions.md`](../predictions.md).

The model ran for 1,000 fixed optimizer steps on the A6000 in 46.1 seconds.
Same-length test exact match peaked at 99.33% and ended at 98.67%. Length-4 OOD
exact match peaked at 10.0% at step 1 and ended at 4.4%; OOD loss rose from
1.77 at step 300 to 2.99 at step 1,000.

**RESULT:** confirmed, classified by Codex at the human's request.

**Interpretation (Codex):** Randomized Abacus embeddings beside RoPE did not
repair positional reuse in this small exact diagnostic. Final length-4 exact
match was below the frozen RoPE baseline's 7.3%.

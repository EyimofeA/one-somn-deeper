# gate1_position_coupling

**Author:** Codex
**Implementation:** Boyle (`implement_gate1_square`)

**CHANGE:** replace physical sequential RoPE indices with deterministic
task-structured coupled indices. Data, generator, splits, architecture,
optimizer, RoPE rotation, and all hyperparameters are frozen from
`gate1_aligned_products`.

For `max_seq_len=9`, numeric digits use LSD=1, tens=2, hundreds=3, and so on
inside every span. PAD=0; BOS=10; N=11; X=12; T=13; ANS=14; EOS=15. Thus
`N 372 X 4 T 1` uses `[11,3,2,1,12,1,13,1]`. This adds no parameters:
the model remains at 51,136 state elements. Source: Cho et al.,
[Position Coupling: Improving Length Generalization of Arithmetic Transformers Using Task Structure](https://arxiv.org/abs/2405.20671).

This diagnostic has no explicit output tokens. Logits remain gathered from
tail-aligned prompt positions, whose coupled coordinates describe input tokens
rather than output-product significance. That mismatch is a known risk and is
not changed in this card.

**PREDICTION:** recorded by the human in [`../predictions.md`](../predictions.md).

The model ran for 1,000 fixed optimizer steps on the A6000 in 44.9 seconds.
The final train batch reached only 48.05% exact match; held-out same-length
reached 19.50%; length-4 OOD reached 1.50%. Train loss flattened around
0.32–0.48 after step 700. The card never fit the trained positions, unlike the
frozen physical-RoPE baseline.

**RESULT:** confirmed, classified by Codex at the human's request.

**Interpretation (Codex):** The tail-aligned/no-output-token risk materialized.
Coupling input-token coordinates erased distinctions needed to decode product
digits from the prompt slots. This card does not isolate length generalization
because it breaks the in-domain mechanism first.

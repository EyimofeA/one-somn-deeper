# gate1_length4_priming

**Author:** Codex
**Implementation:** Boyle (`implement_gate1_square`)

**CHANGE:** move exactly three complete length-4 x groups from the original OOD
split into training. Submission, model, optimizer, RoPE, all original
length-1–3 train rows, same-length test rows, and runtime hyperparameters are
frozen from `gate1_aligned_products`.

The selected x values are 5106, 1970, and 3486. Their ten multiplier rows each
add 30 priming rows: 1.00% of the original 3,000-row train split and 0.99% of
the resulting 3,030 rows. The other 970 length-4 rows remain OOD, with no
duplicate prompt or split leakage. Source: Jelassi et al.,
[Length Generalization in Arithmetic Transformers](https://arxiv.org/abs/2306.15400).

**PREDICTION:** recorded by the human in [`../predictions.md`](../predictions.md).

The model ran for 1,000 fixed optimizer steps on the A6000 in 44.6 seconds.
Train and held-out same-length exact match reached 100%. On the remaining 970
length-4 OOD rows, exact match peaked at 10.21% at step 700 and ended at 9.28%.
The frozen unprimed baseline peaked at 10.0% and ended at 7.3%.

**RESULT:** confirmed, classified by Codex at the human's request.

**Interpretation (Codex):** Thirty target-length rows produced only a marginal
final lift and no high-accuracy reuse. The priming rate is below the threshold
needed to change the learned mechanism in this diagnostic.

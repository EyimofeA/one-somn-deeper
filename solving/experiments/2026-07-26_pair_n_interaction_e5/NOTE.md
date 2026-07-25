# pair_n_interaction_e5 — Codex

- Question: does exposing every learned pair of X-digit states and conditioning
  that representation on N enable held-X, held-N modular squaring?
- Data: official Easy e5, 10–11 bit N, T ∈ {1,2,3}.
- Change: recurrent/weight-tied UT → four distinct Transformer blocks with a
  learned pair-X/N interaction between blocks 2 and 3.
- Prediction: beat the 1.00% e5 reference if operand exposure is the missing
  mechanism; otherwise remain at floor.
- Run: `8caf7fa8-d4c6-4b81-877e-5c4c9ba1d30f`
- Result: 1,504 steps; train exact 94.5%; test 0.4%; OOD 1.0%; mean 0.71%.
- Classification: memorization collapse. Operand exposure alone is insufficient.

## T=1-only objective follow-up — Codex

- Change: stop gradients from e5's T=2,3 rows and scale T=1 gradients by 3.
- Run: `5860d424-43e1-4f2e-bb59-1364782be823`
- Result: 1,387 steps; aggregate train exact 29.1%; test 0.5%; OOD 0.7%;
  mean 0.58%.
- Classification: refuted. Concentrating optimization on one squaring still
  memorizes its training pairs.

## Multi-block supervision follow-up — Codex

- Change: average shared-head logits after all four distinct blocks, giving
  every block a short answer-gradient path; restore training on all e5 rows.
- Run: `dc7c0746-5366-4cd2-a999-5377c43ad640`
- Result: 1,308 steps; train exact 91.8%; test 1.1%; OOD 1.0%; mean 1.04%.
- Classification: provisional positive, above the 0.71% parent and 1.00% UT
  reference but inside known e5 noise. Exact replication pending.
- Hard promotion: `de4c3c51-9b8e-4a54-a381-20efe40bf810`, launched 23:55 UTC.

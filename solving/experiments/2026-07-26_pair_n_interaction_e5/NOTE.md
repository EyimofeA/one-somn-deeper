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


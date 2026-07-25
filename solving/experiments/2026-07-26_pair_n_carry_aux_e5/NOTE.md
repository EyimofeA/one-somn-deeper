# pair_n_carry_aux_e5 — Codex

- Change: add per-column first-square carry-in/out supervision to the committed
  pair/N model; all architecture and optimizer settings held fixed.
- Data: official Easy e5.
- Prediction: exceed 0.71% and the 1.00% reference if carry-state
  identification transfers to held u,N.
- Run: `faaa00f3-2a5e-4eec-85a7-87418b347d7f`
- Result: 1,180 steps; train exact 92.6%; test 0.4%; OOD 0.7%; mean 0.54%.
- Classification: refuted at Easy budget; extra arithmetic supervision reduced
  throughput and did not improve held-out answers.


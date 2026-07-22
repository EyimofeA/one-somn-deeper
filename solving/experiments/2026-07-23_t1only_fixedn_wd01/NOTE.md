# t1only_fixedn_wd01 — P2 ladder rung 1 (fixed single N, unseen x, T=1)

**CHANGE:** dataset only — `split_group=x` with a fixed semiprime (all units mod N,
split 80/20 by x, T=1, `separate_input_output=true`). Model is the unmodified
`claude_std_rope_e1` anchor (wd=0.1); N=323 run uses `claude_std_rope_e1_b115.py`
(batch_size 256→115 so drop_last leaves ≥1 batch on 230 train rows — no other change).
Two datasets: N=323 (φ=288: 230 train / 58 test) and N=1073 (φ=1008: 806 / 202).
900s each on the L40S, monitored eval every 500 steps.

**RESULT (rung 1 of the P2 grokking ladder from `claude code fable/FULL_TRANSCRIPT.md`):**

| N | steps | train EM | test EM final | test EM peak | test loss final |
|---|---|---|---|---|---|
| 323 | 86,500 | 100% (loss 3e-8) | **5.17%** (3/58) | **6.90%** @ step 59k | 7.45 |
| 1073 | 23,000 | 100% (loss 3e-8) | **0.00%** | 0.99% @ step 9.5k | 9.44 |

**Interpretation (marked as interpretation):** the first non-floor held-out number in
the project, but tiny. N=323 shows a slow grokking-shaped climb (test EM drifts up
over tens of thousands of steps after full memorization) that was still at 5-7% when
the LR annealed out. N=1073, with 3.5x the data, shows nothing. Even the easiest rung
of the ladder — one fixed modulus, no composition, no N-transfer — barely moves off
floor for this architecture. Rung 2 (multi-N seen at train) was already at floor
(0.5-0.75%, see `2026-07-22_t1only_probe_*` and the split-labeling correction in its
NOTE). The gate ("rung 3 ≥ ~5%") is nowhere in sight: the bottleneck is upstream of
cross-N transfer — the one-step map itself is barely learnable even per-modulus.

**Gotcha for reproducers:** generator default is `data_format=causal_lm`, which puts
the answer digits inside `input_ids` — a bidirectional model scores 100% instantly by
copying. First launch hit exactly this (test loss 1.8e-7 by step 2500) and was
discarded. Always generate ladder data with `--separate_input_output true`.

Follow-ups run afterwards: wd=1.0 round (worse — see `2026-07-23_t1only_fixedn_wd1/`),
1800s wd=0.1 reruns with final checkpoints for D1 (results in RESEARCH_LOG when done).

Metrics: `../metrics/rung1_n{323,1073}_monitor.jsonl`. Runs on L40S box, datasets at
`data/generated/squaring_mod_t1only_fixed_n_{323,1073}_xsplit` (box only).

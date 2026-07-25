# Competition run scoreboard

Auto-built 2026-07-24 from `solving/experiments/metrics/*.jsonl` + NOTE/concept recoveries.
Means are exact-accuracy ×100 (%). **e1 is not a valid ranking signal** (answer-space collapse). Prefer e5 / Medium / Hard.

**Scoring regime note (upstream `79f0a09`, 2026-07-24):** Easy/Medium still rank by mean exact %. Hard leaderboard now ranks by certified **Max T** then **OOD N Max T** on ladder T=1,2,4,8,16,32,64 (consecutive 100%-exact prefix). Historical Hard rows below still show mean% because that is what we logged; they are **not** the current Hard rank key. Re-submit after this pin to populate Max T fields.

Live filtered view: Cursor canvas `experiment-scoreboard`.

| # | date | card | ds | mean% | test% | ood% | steps | src | one-change / note |
|---|------|------|----|------:|------:|-----:|------:|-----|-------------------|
| 1 | 2026-07-21 | `b0_transformer` | e1 | 1.00 | 2.00 | 0.00 | 261 | JSONL | baseline TF |
| 2 | 2026-07-21 | `b0_transformer_max` | e1 | 1.30 | 2.70 | 0.00 | 557 | JSONL | d=64 + sched + bs |
| 3 | 2026-07-21 | `b1_mlp` | e1 | 0.30 | 0.70 | 0.00 | 287 | JSONL | MLP bag |
| 4 | 2026-07-21 | `b1_mlp_max` | e1 | 1.00 | 2.00 | 0.00 | 585 | JSONL | max recipe |
| 5 | 2026-07-21 | `b2_rnn` | e1 | 0.70 | 1.30 | 0.00 | 258 | JSONL | BiGRU |
| 6 | 2026-07-21 | `b2_rnn_max` | e1 | 1.00 | 2.00 | 0.00 | 555 | JSONL | max recipe |
| 7 | 2026-07-21 | `claude_evalk4_zeroinit` | e1 | 2.33 | 2.70 | 2.00 | — | RECOVERED |  |
| 8 | 2026-07-21 | `claude_hard_h1` | h1 | 0.03 | 0.00 | — | 190017 | RECOVERED | d=2048 K=4 Hard |
| 9 | 2026-07-21 | `claude_pv_ansplace` | e1 | 2.00 | 2.00 | 2.00 | — | RECOVERED |  |
| 10 | 2026-07-21 | `claude_pv_d128` | e1 | 2.00 | 2.00 | 2.00 | 394 | RECOVERED |  |
| 11 | 2026-07-21 | `claude_pv_evalk4` | e1 | 2.17 | 3.30 | 1.00 | — | RECOVERED |  |
| 12 | 2026-07-21 | `claude_pv_k4_ut` | e1 | 5.83 | 2.70 | 9.00 | — | RECOVERED | place-value emb |
| 13 | 2026-07-21 | `claude_pv_k4_ut` | e3 | 1.31 | 1.30 | 1.40 | 1505 | RECOVERED | place-value emb |
| 14 | 2026-07-21 | `claude_pv_noabspos` | e1 | 3.83 | 0.70 | 7.00 | — | RECOVERED |  |
| 15 | 2026-07-21 | `claude_pv_tadapt` | e1 | 2.00 | — | — | — | RECOVERED |  |
| 16 | 2026-07-21 | `claude_pv_tcoupled` | e1 | 2.00 | 2.00 | 2.00 | — | RECOVERED |  |
| 17 | 2026-07-21 | `depth_d32_act` | e1 | 3.80 | 2.70 | 5.00 | 397 | JSONL | soft ACT |
| 18 | 2026-07-21 | `depth_d32_act` | e5 | 0.80 | 0.70 | 0.80 | 1798 | JSONL | soft ACT |
| 19 | 2026-07-21 | `depth_d32_k2` | e1 | 6.20 | 3.30 | 9.00 | 383 | JSONL | K=2 at d32 |
| 20 | 2026-07-21 | `depth_d32_k2` | e5 | 0.50 | 0.30 | 0.70 | 2503 | JSONL | K=2 at d32 |
| 21 | 2026-07-21 | `depth_d32_k2_ut` | e1 | 6.50 | 4.00 | 9.00 | 393 | JSONL | UT depth emb K=2 |
| 22 | 2026-07-21 | `depth_d32_k2_ut` | e5 | 0.70 | 0.80 | 0.50 | 2359 | JSONL | UT depth emb K=2 |
| 23 | 2026-07-21 | `depth_d32_k2_ut_evalk4` | e1 | 6.80 | 4.70 | 9.00 | 407 | JSONL | train K2 / eval K4 |
| 24 | 2026-07-21 | `depth_d32_k2_ut_evalk4` | e1 | 6.80 | 4.70 | 9.00 | 409 | JSONL | train K2 / eval K4 |
| 25 | 2026-07-21 | `depth_d32_k2_ut_evalk4` | e1 | 6.80 | 4.70 | 9.00 | 609 | JSONL | train K2 / eval K4 |
| 26 | 2026-07-21 | `depth_d32_k2_ut_evalk4` | e5 | 0.40 | 0.70 | 0.20 | 3583 | JSONL | train K2 / eval K4 |
| 27 | 2026-07-21 | `depth_d32_k2_ut_evalk4` | m5 | 0.10 | 0.10 | 0.10 | 62770 | JSONL | train K2 / eval K4 |
| 28 | 2026-07-21 | `depth_d32_k3` | e1 | 5.00 | 2.00 | 8.00 | 407 | JSONL | K=3 at d32 |
| 29 | 2026-07-21 | `depth_d32_k3` | e5 | 0.40 | 0.70 | 0.00 | 2341 | JSONL | K=3 at d32 |
| 30 | 2026-07-21 | `depth_d32_k4` | e1 | 5.50 | 2.00 | 9.00 | 471 | JSONL | d32×K4 combo |
| 31 | 2026-07-21 | `depth_d32_k4` | e5 | 0.80 | 1.10 | 0.50 | 2527 | JSONL | d32×K4 combo |
| 32 | 2026-07-21 | `depth_d32_k4_ncond` | e1 | 5.80 | 2.70 | 9.00 | 407 | JSONL | N-cond FiLM |
| 33 | 2026-07-21 | `depth_d32_k4_ncond` | e5 | 0.30 | 0.30 | 0.30 | 2215 | JSONL | N-cond FiLM |
| 34 | 2026-07-21 | `depth_d32_k4_ut` | e1 | 4.70 | 1.30 | 8.00 | 413 | JSONL | UT depth emb K=4 |
| 35 | 2026-07-21 | `depth_d32_k4_ut` | e5 | 1.00 | 0.80 | 1.20 | 2275 | JSONL | UT depth emb K=4 |
| 36 | 2026-07-21 | `depth_d32_k4_ut` | m1 | 0.10 | 0.10 | 0.00 | 44993 | JSONL | UT depth emb K=4 |
| 37 | 2026-07-21 | `depth_d32_k4_ut` | m5 | 0.10 | 0.10 | 0.10 | 51049 | JSONL | UT depth emb K=4 |
| 38 | 2026-07-21 | `depth_d32_k4_ut_optsched` | m5 | 0.20 | 0.10 | 0.20 | 70007 | JSONL | clamped cosine schedule |
| 39 | 2026-07-21 | `depth_d32_k6` | e1 | 4.50 | 2.00 | 7.00 | 411 | JSONL | K=6 at d32 |
| 40 | 2026-07-21 | `depth_d32_k8` | e1 | 2.70 | 3.30 | 2.00 | 413 | JSONL | K=8 at d32 |
| 41 | 2026-07-21 | `depth_d32_midloop_k4` | e1 | 0.80 | 0.70 | 1.00 | 567 | JSONL | midloop only |
| 42 | 2026-07-21 | `depth_d32_midloop_k4` | e5 | 0.80 | 0.90 | 0.70 | 2817 | JSONL | midloop only |
| 43 | 2026-07-21 | `depth_looped_k4` | e1 | 1.80 | 0.70 | 3.00 | 489 | JSONL | tied loops K=4 |
| 44 | 2026-07-21 | `depth_looped_k8` | e1 | 1.70 | 1.30 | 2.00 | 491 | JSONL | tied loops K=8 |
| 45 | 2026-07-21 | `scale_tf_d128` | e1 | 1.80 | 2.70 | 1.00 | 503 | JSONL | width d=128 |
| 46 | 2026-07-21 | `scale_tf_d32` | e1 | 2.70 | 1.30 | 4.00 | 539 | JSONL | width d=32 |
| 47 | 2026-07-21 | `scale_tf_d96` | e1 | 1.50 | 2.00 | 1.00 | 541 | JSONL | width d=96 |
| 48 | 2026-07-22 | `claude_abacus_e1` | e1 | 3.67 | 1.33 | 6.00 | 1381 | RECOVERED |  |
| 49 | 2026-07-22 | `claude_fire_e1` | e1 | 1.83 | 0.67 | 3.00 | 1251 | RECOVERED |  |
| 50 | 2026-07-22 | `claude_fireabacus_e1` | e1 | 1.33 | 0.67 | 2.00 | 1225 | RECOVERED |  |
| 51 | 2026-07-22 | `claude_std_rope_e1` | e1 | 4.83 | 2.67 | 7.00 | 1353 | RECOVERED | mean=avg test/ood |
| 52 | 2026-07-22 | `fable_hard_h1_muon` | e5 | — | 0.42 | 2.00 | 3616 | RECOVERED | replaced fable_hard_h1's flat-lr=3e-4 WarmupSchedule with Muon (hidden |
| 53 | 2026-07-22 | `fable_hard_h1_muon` | h1 | — | — | — | — | UNRECOVERED | replaced fable_hard_h1's flat-lr=3e-4 WarmupSchedule with Muon (hidden |
| 54 | 2026-07-22 | `fable_hard_h1_muon` | m5 | — | 0.12 | 0.07 | 27759 | RECOVERED | replaced fable_hard_h1's flat-lr=3e-4 WarmupSchedule with Muon (hidden |
| 55 | 2026-07-24 | `depth_d32_k4_ut_ste` | e5 | 0.50 | 0.70 | 0.30 | 2521 | JSONL | After each tied UT block except the last, decode→argmax/STE→re-embed t |
| 56 | 2026-07-24 | `depth_d32_k4_ut_ste` | m5 | 0.20 | 0.10 | 0.20 | 58525 | JSONL | After each tied UT block except the last, decode→argmax/STE→re-embed t |
| 57 | — | `submission_v2` | h1 | 0.05 | 0.10 | 0.00 | — | RECOVERED | Fable v2 confidence-gated Hard |

## Champions (honest tiers)

| Tier | Card | Metric | Note |
|------|------|--------|------|
| Easy e5 | `depth_d32_k4_ut` | mean **1.00%** | Prefer over e1 |
| Medium m5 | `depth_d32_k4_ut_optsched` | mean **0.20%** | Schedule-safe |
| Hard H1 (legacy mean%) | `submission_v2` | mean **0.05%** | Pre-`79f0a09` metric; LB now Max T / OOD N Max T |
| Easy e1 (invalid) | `depth_d32_k2_ut_evalk4` | mean 6.80% | Do not rank on this |

Regenerate by asking the agent to rebuild from metrics + NOTE.md.

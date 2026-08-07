# Deadline priority table — 2026-08-04

Goal: improve the official leaderboard today. Scores below are prior hosted
evidence, not claims about the current evaluator.

| Rank | Candidate | Expected leaderboard gain | Confidence | Implement | Train / eval | Risk | Legal | Run now? |
|---:|---|---|---|---|---|---|---|---|
| 1 | Pair/N interaction + multi-block supervision on Easy e5 | 0 to +0.04 pp vs 1.00% UT e5 champion | medium-low; prior 1.04% is inside e5 noise | 0 min | hosted 60s | noise / no Medium evidence | validated | yes |
| 2 | Fable timeout-safe AdamW on Medium m5 | about +0.05 pp vs 0.20% UT control | medium; prior 0.25% | 0 min | hosted 600s | low absolute score; old metric did not expose current depth profile | validated | yes |
| 3 | Fable v2 confidence-gated recurrent register on Hard | best historical mean (0.05%) and learned T-conditioned recurrence | low-medium; old profile metric only | 5 min provenance/validation | hosted 3600s | may certify no T; current Hard rank changed | validate after packaging | yes |
| 4 | Pair/N multi-block on Hard | possible higher next-rung T=1 than Fable | low; Easy-only 1.04%, old Hard result unavailable | 0 min | hosted 3600s | one Hard slot; no current-rule Hard evidence | validated | no—reserve behind v2 |
| 5 | UT K4 e5 / optsched m5 controls | 0 pp; known 1.00% / 0.20% | high | 0 min | hosted | dominated by rows 1–2 | validated historically | fallback only |
| 6 | New local curriculum or architecture change | unknown | low before a new run | 30–60 min | local + hosted | consumes deadline with no current evidence | potentially legal | no |

Decision rule: submit rows 1–3 first. Use remaining Easy quota only for a
replication or a tiny, isolated inference-time setting that beats row 1 locally.

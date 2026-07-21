# Status (living)

Last updated: 2026-07-21 end of day.

## Hard leaderboard

We are **#11 at 0.03%** (`mof` / Claude Hard run). Top is **0.40%** (az). Ranks ~3–16 are a near-tie at 0.02–0.05%. Nobody has solved the task.

Day synthesis: [`learnings/sessions/2026-07-21.md`](../learnings/sessions/2026-07-21.md).  
What’s next (design): [`learnings/concepts/17-recurrence-generalisation.md`](../learnings/concepts/17-recurrence-generalisation.md).

## Best scored cards (learned line)

| Axis | Card | Score | Note |
|------|------|-------|------|
| Easy e1 | `depth_d32_k2_ut_evalk4` | **6.80%** (n=3, σ≈0) | **Invalid ranking signal** — see day synthesis |
| Easy e5 | `depth_d32_k4_ut` | **1.00%** | Variable N; prefer over e1 for ranking |
| Medium m5 | `depth_d32_k4_ut_optsched` | **~0.20%** | Fixed cosine sawtooth |
| Hard H1 | `claude_hard_h1` | **0.03%** | Train 100% / eval 0% — **grokked memorization** |

## Active submissions (use these)

| Path | Role |
|------|------|
| `depth_d32_k4_ut_optsched/` | Medium/Hard schedule-safe UT K4 |
| `depth_d32_k2_ut_evalk4/` | Easy e1 peak (train2/eval4) |
| `claude_pv_k4_ut/` | Place-value UT (Claude; +1.16 pp on e1 vs anchor) |
| `claude_hard_h1/` | Hard run artifact (do not scale width further) |

Everything else under `solving/submissions/` is archive / ablation — see [`submissions/README.md`](submissions/README.md).

## Next (ordered)

1. Local grokking probes (no quota) — find memorise→generalise knobs.
2. Weight decay ↑ (0.1 → 1–3) one-change cards.
3. Re-quantised recurrence (concept 17) — force exact iterative step.
4. Gate on **e5 / Medium**, not e1 alone.
5. Default LR: inv-sqrt / clamped cosine — never short `T_max` cosine on Medium+.

## Ops

Quota, noise protocol, Medium split: [`experiments/OPS.md`](experiments/OPS.md).  
Append-only facts: [`RESEARCH_LOG.md`](RESEARCH_LOG.md).

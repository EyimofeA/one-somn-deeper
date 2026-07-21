# Ops

## Daily quotas

| Tier | / UTC day |
|------|-----------|
| Easy | 60 |
| Medium | 6 |
| Hard | 1 |

When two agents share a day: split Medium (e.g. 3+3). Hard = principal only. Update `left` from CLI after submits.

## Noise

Easy/Medium manifests use **one seed (74)**. Estimate σ by resubmitting the **same** file n times.

| Goal | n |
|------|---|
| Rough σ | 3 |
| Promote tiny Δ (~0.3 pp) | 5, or demand larger Δ / dual gate |

Promote only if beat champ by ≳2σ **or** clear win on **e5 + Medium** (not e1 alone).

## Schedule rule

Never `CosineAnnealingLR` with small `T_max ≈ c×seconds` on Medium/Hard. Prefer inv-sqrt/Noam or clamped cosine. See `learnings/concepts/15-lr-schedules-wallclock.md`.

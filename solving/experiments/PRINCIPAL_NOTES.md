# Principal notes — 2026-07-21 (session)

Standing instructions from you this turn:

1. **Stay on Karpathy’s recipe** — one change at a time; don’t jump to fancy losses before the baseline is solid; visualize everything we get back.
2. **Reserve 10 Easy accepts today for Claude Code** — it will also critique. Parent/Cursor must not burn the day below that reserve.
3. **Clarify “scaling laws”** = curves when we **increase model size** (width / params / capacity under the clock), not ponder-τ curves. Plot those when we scale up.
4. **Later: try diffusion** (e.g. diffusion on the output / iterative refine of answer tokens) — backlog only for now.
5. Intermediate **scored** step outputs are not available; don’t confuse that with internal loops or aux training tricks.

## Quota math (this UTC day)

| | |
|--|--|
| Last CLI `left` | **33** (after UT e1+e5 batch) |
| Reserved for Claude Code | **10** |
| Parent budget remaining today | **≤ 23** Easy accepts |

Re-check `left` on next submit; if Claude burns some, parent budget = `left − 10` (floor at 0).

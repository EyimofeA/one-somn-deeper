# Daily Easy quota

Competition: **60 Easy / UTC day**, 6 Medium, 1 Hard. CLI prints `left: N today` on submit.

## 2026-07-21 (UTC)

| When | left | Note |
|------|------|------|
| After K-sweep e1 + e5 gates | **41** | CLI after K=3 e5 submit |
| After N-cond e1 | **40** | job `f752d166…` score 5.83% |
| After N-cond e5 | **39** | job `7cecdbb0…` score 0.29% |
| After ACT e1 | **38** | job `ef972089…` score 3.83% |
| After ACT e5 | **37** | job `4af90448…` score 0.79% |
| After UT k2/k4 e1+e5 | **33** | best e5 now UT K4 **1.00%** |
| After midloop e1+e5 | **31** | midloop rejected (e1 0.83%, e5 0.79%) |
| After UT K2→eval4 e1+e5 | **29** | new e1 best **6.83%**; e5 0.42% |

Used today ≈ 31 Easy accepts.

See `PRINCIPAL_NOTES.md`. Noise / repeat protocol: `NOISE.md`.

## Protocol

After each `one-layer submit`: set `left` from CLI here + experiment card. Never spend into the Claude reserve without principal override.

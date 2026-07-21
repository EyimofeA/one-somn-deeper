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
| After noise + Medium batch | Easy **5**; Medium **left after optsched** | see RESEARCH_LOG |

## Medium results (parent)

| Arch | dataset | mean |
|------|---------|------|
| UT K4 (broken sched) | m5 | 0.09% |
| UT K4 (broken sched) | m1 | 0.08% |
| UT K2→eval4 | m5 | 0.14% |
| **UT K4 optsched** | m5 | **~0.20%** |

Parent Medium slots used. Hard candidate: `depth_d32_k4_ut_optsched`.

## Medium split (2026-07-21)

Daily Medium = **6**. Parent and Claude Code each take **3**. Do not burn the other’s slots.  
Hard = **1**/day — principal picks best or combo after both Medium batches; no auto Hard.

Parent Medium plan: `depth_d32_k4_ut` (e5 champ) → m5, m1; third = `depth_d32_k2_ut_evalk4` → m5.

## Protocol

After each `one-layer submit`: set `left` from CLI here + experiment card. Never spend into the Claude reserve without principal override.

# depth_d32_k4_ut_ste

**CHANGE:** After each tied UT block except the last, decode→argmax/STE→re-embed through tied token embedding (+ re-add absolute position). Parent: `depth_d32_k4_ut_optsched`.

**RESULT:** refuted vs parent on Easy e5; Medium m5 ≈ optsched floor.

**Detail:** PR #1 (`chatgpt/ste-token-bottleneck`). `one-layer validate` green; L40S smoke OK.

| run | job | mean | test | ood | steps |
|-----|-----|------|------|-----|-------|
| e5 | `92e064ea…` | **0.50%** | 0.70% | 0.30% | 2521 |
| m5 | `fac54972…` | **0.17%** | 0.10% | 0.20% | 58525 |

Parent refs: UT K4 e5 **1.00%**; optsched m5 **0.17–0.20%**. STE bottleneck did not help e5 and did not lift Medium.

Metrics: `solving/experiments/metrics/depth_d32_k4_ut_ste_{e5,m5}.jsonl`.

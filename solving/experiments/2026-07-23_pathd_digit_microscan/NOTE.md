# pathd_digit_microscan

**Author:** Codex

**CHANGE:** step operator only — replace each outer Path-D step's parallel register mutation with a weight-tied LSD-to-MSD digit micro-scan carrying one learned hard categorical state.

**RATIONALE:** the serial discrete channel supplies an explicit local dependency for carry/borrow-like behavior while leaving prompt conditioning, outer recurrence, digit quantization, optimizer, scheduler, and readout unchanged.

**RESULT:** confirmed (human classification).

**Facts:** CPU contract smoke passed source validation, variable/padded row geometry, bf16 forward, one backward/optimizer step, deterministic pure eval, and T=0/T=16 edge forwards.

Local A6000, fixed N=1073, T=1, held-out x, 900-second budget: train EM reached 100% by step 2,000. Held-out EM peaked at 1.49% at step 2,000 and fell to 0.99% at step 2,500 while held-out loss rose to 5.41. Throughput was about 3.35 steps/s. The N=323 launch stopped before step 1 because the unchanged submission batch size 512 exceeds that tiny split under `drop_last=True`.

Hosted Easy e5 `c22bc015-e380-46cb-a36d-1425e5cf4524`: 613 steps in 60.1 seconds; final logged train loss 2.115 / train EM 1.0%; test 0.50%, held-out-T OOD 0.50%, mean 0.50%.

Metrics: `metrics/n1073_monitor.jsonl`, `metrics/n1073_stdout.log`, `metrics/e5_c22bc015_metrics.jsonl`.

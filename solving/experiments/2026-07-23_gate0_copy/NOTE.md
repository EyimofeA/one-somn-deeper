# gate0_copy

**Author:** Avicenna

**CHANGE:** diagnostic target only: modular-squaring answer → exact copy of X; model and optimizer are byte-identical to `claude_std_rope_e1`.

**RESULT:** refuted (human classification).

Fixed N=10403 and T=1; x=1..999 is deterministically split into 800 train and 199 test rows, with 1,000 unique four-digit OOD x rows.

The A6000 run was stopped deliberately at step 2,000. Train, held-out-x test,
and four-digit OOD exact match were all 100% from step 500 through step 2,000.
At step 500, test loss was 0.000057 and OOD loss was 0.000064. The predicted
failure on the new leading OOD digit did not occur.

Metrics: `metrics/monitor.jsonl`, `metrics/stdout.log`.

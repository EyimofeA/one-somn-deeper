# Neural GPU square/reduce hosted transfer

Frozen legal source: [`../../submissions/neural_gpu_square_reduce/submission.py`](../../submissions/neural_gpu_square_reduce/submission.py).

| Dataset | N regime | Train steps | Final train | Test | OOD | Mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| E1 | fixed 323 | 651 | 75.8% | 39.3% | 42.0% | 40.67% |
| E2 | fixed 899 | 676 | 39.1% | 22.5% | 10.0% | 16.25% |
| E3 | varying | 1,476 | 3.9% | 0.4% | 0.9% | 0.62% |
| E4 | varying | 1,504 | 0.0% | 0.4% | 0.7% | 0.59% |
| E5 | varying | 1,681 | 32.0% | 0.6% | 0.5% | 0.54% |
| E6 | fixed 247 | 690 | 53.1% | 37.8% | 28.3% | 33.09% |
| E7 | fixed 287 | 737 | 77.3% | 37.1% | 41.2% | 39.15% |
| E8 | fixed 287 | 709 | 84.4% | 31.1% | 21.2% | 26.12% |
| E9 | fixed 299 | 712 | 80.5% | 34.7% | 22.2% | 28.44% |
| E10 | fixed 403 | 931 | 71.1% | 35.9% | 46.4% | **41.13%** |
| M6 | fixed 1,517 | 15,573 | 0.0% | 0.3% | 0.4% | 0.33% |

M6 initially learned: train exact peaked at 28.1% at step 2,100. It then
collapsed toward zero by step 6,000 and loss exceeded 20 after step 12,000.
Thus the final M6 result is confounded by optimizer instability and does not
isolate arithmetic scale or model width.

The full Easy sweep cleanly separates fixed N (16.25%--41.13%) from varying N
(0.54%--0.62%). The owner explicitly authorized Hard escalation. Exact SHA-1
`ff3381c9be98884f0409a3a63fa467cf6be47ab9` was accepted as Hard H1 job
`37cedcd2-a172-4fb3-b289-24260777c83b`, but failed before scoring with
`EVALUATION_FAILED`.

## Muon correction

The standalone winner held Muon at 0.02 through step 1,000, cosine-decayed only
Muon to 0.002 through step 5,000, then clamped it at 0.002; scalar AdamW stayed
at 0.0003. The submission incorrectly stretched warmdown across wall time.
Easy ended around 650--1,700 updates, but M6 kept an excessive Muon rate long
enough to destroy its early solution. The matched correction is a scheduler
counter implementing the original 1,000/5,000 boundaries, not a new optimizer.

Raw hosted metrics are in [`metrics/`](metrics/).

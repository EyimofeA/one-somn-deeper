# Neural GPU square/reduce hosted transfer

Frozen legal source: [`../../submissions/neural_gpu_square_reduce/submission.py`](../../submissions/neural_gpu_square_reduce/submission.py).

| Dataset | Job | Train steps | Final train | Test | OOD | Mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| E6 | `6b5ef912-e92f-42a2-8675-cf47937bc50b` | 690 | 53.1% | 37.8% | 28.3% | 33.09% |
| E7 | `fb58a13a-7a2d-4694-b76a-5618710c81ae` | 737 | 77.3% | 37.1% | 41.2% | 39.15% |
| M6 | `006cfcc8-c40e-4083-8e6d-0874f758ef16` | 15,573 | 0.0% | 0.3% | 0.4% | 0.33% |

M6 initially learned: train exact peaked at 28.1% at step 2,100. It then
collapsed toward zero by step 6,000 and loss exceeded 20 after step 12,000.
Thus the final M6 result is confounded by optimizer instability and does not
isolate arithmetic scale or model width.

The owner explicitly authorized Hard escalation. Exact SHA-1
`ff3381c9be98884f0409a3a63fa467cf6be47ab9` was queued as Hard H1 job
`37cedcd2-a172-4fb3-b289-24260777c83b`.

Raw hosted metrics are in [`metrics/`](metrics/).

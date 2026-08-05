# Competition-style run report

- Run: `vdf_square_reduce_dynamic_depth_e1`
- Completed training steps: 434
- Training time: 60.0s
- Mean final exact: 1.6667%

## Final splits

| Split | Exact | Loss |
| --- | ---: | ---: |
| test | 3.3000% | 3.010 |
| ood | 0.0000% | 5.715 |

## Seen-N depth ladder

| T | Exact | Correct / total | Certification |
| ---: | ---: | ---: | --- |
| 1 | 0.0000% | 0 / 38 | failed |
| 2 | 0.0000% | 0 / 38 | failed |
| 4 | 0.0000% | 0 / 38 | failed |
| 8 | 0.0000% | 0 / 38 | failed |
| 16 | 0.0000% | 0 / 38 | failed |
| 32 | 0.0000% | 0 / 38 | failed |
| 64 | 0.0000% | 0 / 38 | failed |

## OOD-N depth ladder

| T | Exact | Correct / total | Certification |
| ---: | ---: | ---: | --- |
| 1 | 0.9766% | 5 / 512 | failed |
| 2 | 0.3906% | 2 / 512 | failed |
| 4 | 0.0000% | 0 / 512 | failed |
| 8 | 0.1953% | 1 / 512 | failed |
| 16 | 0.1953% | 1 / 512 | failed |
| 32 | 0.1953% | 1 / 512 | failed |
| 64 | 0.1953% | 1 / 512 | failed |

## Training dynamics

The evaluator retained 6 bounded training observations. See `training_curve.svg` and `depth_profile.svg`.

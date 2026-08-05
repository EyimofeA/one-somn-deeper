# Competition-style run report

- Run: `vdf_final_label_t_curriculum_e1`
- Completed training steps: 461
- Training time: 60.1s
- Mean final exact: 1.6667%

## Final splits

| Split | Exact | Loss |
| --- | ---: | ---: |
| test | 3.3000% | 2.611 |
| ood | 0.0000% | 4.839 |

## Seen-N depth ladder

| T | Exact | Correct / total | Certification |
| ---: | ---: | ---: | --- |
| 1 | 5.2632% | 2 / 38 | failed |
| 2 | 0.0000% | 0 / 38 | failed |
| 4 | 0.0000% | 0 / 38 | failed |
| 8 | 0.0000% | 0 / 38 | failed |
| 16 | 0.0000% | 0 / 38 | failed |
| 32 | 0.0000% | 0 / 38 | failed |
| 64 | 0.0000% | 0 / 38 | failed |

## OOD-N depth ladder

| T | Exact | Correct / total | Certification |
| ---: | ---: | ---: | --- |
| 1 | 0.3906% | 2 / 512 | failed |
| 2 | 0.3906% | 2 / 512 | failed |
| 4 | 0.0000% | 0 / 512 | failed |
| 8 | 0.3906% | 2 / 512 | failed |
| 16 | 0.1953% | 1 / 512 | failed |
| 32 | 0.0000% | 0 / 512 | failed |
| 64 | 0.1953% | 1 / 512 | failed |

## Training dynamics

The evaluator retained 6 bounded training observations. See `training_curve.svg` and `depth_profile.svg`.

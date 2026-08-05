# Competition-style run report

- Run: `vdf_architecture_audit_direct_transformer_e1`
- Completed training steps: 697
- Training time: 60.1s
- Mean final exact: 1.5000%

## Final splits

| Split | Exact | Loss |
| --- | ---: | ---: |
| test | 2.0000% | 3.762 |
| ood | 1.0000% | 8.911 |

## Seen-N depth ladder

| T | Exact | Correct / total | Certification |
| ---: | ---: | ---: | --- |
| 1 | 2.6316% | 1 / 38 | failed |
| 2 | 2.6316% | 1 / 38 | failed |
| 4 | 0.0000% | 0 / 38 | failed |
| 8 | 0.0000% | 0 / 38 | failed |
| 16 | 0.0000% | 0 / 38 | failed |
| 32 | 10.5263% | 4 / 38 | failed |
| 64 | 0.0000% | 0 / 38 | failed |

## OOD-N depth ladder

| T | Exact | Correct / total | Certification |
| ---: | ---: | ---: | --- |
| 1 | 0.1953% | 1 / 512 | failed |
| 2 | 0.9766% | 5 / 512 | failed |
| 4 | 0.5859% | 3 / 512 | failed |
| 8 | 0.3906% | 2 / 512 | failed |
| 16 | 0.1953% | 1 / 512 | failed |
| 32 | 1.7578% | 9 / 512 | failed |
| 64 | 0.7812% | 4 / 512 | failed |

## Training dynamics

The evaluator retained 8 bounded training observations. See `training_curve.svg` and `depth_profile.svg`.

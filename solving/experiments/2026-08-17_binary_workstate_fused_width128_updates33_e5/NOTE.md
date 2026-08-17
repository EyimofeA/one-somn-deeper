# Official local E5 transfer: width 128, 33 updates

This changes only width from the failed width-256/33 official local E5 arm.
The deterministic T=1 diagnostic predicts much better learning per wall-clock.

Promotion requires more than 200 updates, mean exact above 0.25%, and nonzero
T=1 accuracy on both seen and unseen N.

## Result

Rejected for Easy transfer.

- Completed updates: 277
- Final train loss: 0.629739
- Test exact: 0.250% (3/1,200)
- OOD exact: 0.000% (0/600)
- Seen-N T=1: 0/512
- OOD-N T=1: 0/512
- Mean exact: 0.125%

The update-count gate passed, but score regressed below the width-256/33 arm
and both T=1 profiles remained zero. Continue only with the preregistered
batch-256 throughput control.

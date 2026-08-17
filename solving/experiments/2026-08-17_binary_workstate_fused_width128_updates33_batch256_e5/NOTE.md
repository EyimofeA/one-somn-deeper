# Official local E5: width 128, 33 updates, batch 256

This changes only training batch size from the failed batch-512 arm. It tests
whether more optimizer updates within the evaluator clock outweigh fewer
examples per update.

Promotion requires more than 450 updates, mean exact above 0.125%, and nonzero
T=1 accuracy on both seen and unseen N.

## Result

Partially passed; not promoted.

- Completed updates: 505
- Final train loss: 0.635214
- Test exact: 0.5833% (7/1,200)
- OOD exact: 0.0000% (0/600)
- Seen-N T=1: 0/512
- OOD-N T=1: 1/512
- Mean exact: 0.2917%

The update and mean-score gates passed, and OOD-N T=1 became nonzero. Seen-N
T=1 remained zero, so the all-gates promotion rule fails. Test batch 128 once
as the last pure batch-size control.

# Official local E5: width 128, 33 updates, batch 128

This is the final pure batch-size control. It changes only training batch from
256 to 128 and tests whether additional updates outweigh reduced examples per
update.

Promotion requires more than 850 updates, mean exact above 0.2917%, and
nonzero T=1 accuracy on both seen and unseen N.

## Result

Rejected for the primary score, retained as a T=1 signal.

- Completed updates: 528
- Final train loss: 0.634668
- Test exact: 0.2500% (3/1,200)
- OOD exact: 0.0000% (0/600)
- Seen-N T=1: 1/512
- OOD-N T=1: 1/512
- Mean exact: 0.1250%

Batch 128 did not materially improve update count and regressed mean exact, so
batch 256 remains the throughput anchor. However, this is the first arm tonight
with nonzero T=1 on both profiles. That motivates the isolated full-window
T=1-gradient test at batch 256.

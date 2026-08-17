# Official local E5 transfer: width 256, 33 updates

This ports the locally promoted 33-update transition into the unchanged
competition interface. The official public E5 evaluator tests parsing, mixed
T=1/2/3 recurrence, output conversion, compilation, and the 60-second training
clock. No online quota is used.

Promotion requires a successful run, more than 200 updates, and nonzero T=1
accuracy on both seen and unseen N.

## Result

Rejected for Easy transfer.

- Completed updates: 129
- Final train loss: 0.667566
- Test exact: 0.500% (6/1,200)
- OOD exact: 0.000% (0/600)
- Seen-N T=1: 0/512
- OOD-N T=1: 0/512
- Mean exact: 0.250%

The run completed successfully but failed both the >200-update throughput gate
and both T=1 gates. The 60-second practice budget starves this width/depth pair.
This does not refute a one-hour Hard run, but it prohibits promotion from Easy.

## 2026-08-17 — width-128 / 33-update T=1 control

- **One change:** Reduce channels from 256 to 128 in the promoted 33-update,
  3x3 ConvGRU. Keep data, seed, loss, dropout, Muon, and 5.12M examples fixed.
- **Prediction:** Validation exact will exceed 16%, both unseen-N audits will
  exceed 13%, and wall time will stay below 800 seconds. Promote only if all
  three accuracy gates pass.

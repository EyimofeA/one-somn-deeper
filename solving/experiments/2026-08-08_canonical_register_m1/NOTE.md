# Canonical register Medium m1 downward-transfer gate

Run the exact canonical-register candidate on public Medium m1 for the full
600-second budget. M1 training labels are T=4/8/16, so the conditional T=1
curriculum is inactive and the model must identify a reusable one-step cell
only through composed final labels.

Primary readout is the evaluator-owned seen-N and OOD-N T=1 profile. This is a
Hard-selection gate, not permission to inspect any hidden Hard data.

Prediction is registered in [`../predictions.md`](../predictions.md). Raw runner
output remains Git-ignored and will be backed up from the GPU.

## Result

Refuted after 9,815 updates / 600.02 seconds. Final train CE was 2.2930,
test was 0/3,000, and OOD was 2/3,000. The seen-N T=1 profile was 0/192 and
OOD-N T=1 was 0/512. The loss stayed at the decimal digit-frequency floor for
the entire run. A canonical state interface does not make the one-step root
identifiable from T=4/8/16 final labels.

# Square/reduce interface-noise ablation

Status: refuted at the seed-0 gate; no additional seeds run.

This is the owner's selected Option 1 and a one-variable follow-up to
[`../2026-08-09_t1_factored_e5_support/NOTE.md`](../2026-08-09_t1_factored_e5_support/NOTE.md).
The data, 443,594-parameter factored model, optimizer, seed, final-label-only
loss, and 180-second clock remain fixed. During training only, Gaussian noise
with standard deviation 0.1 is added once after the fourth square step and
before the first reduction step. Evaluation is deterministic and noise-free.

Mechanism: a continuous interface can encode brittle example-specific details
that the reducer decodes while memorizing training rows. Perturbing only that
interface should penalize such hidden codes and favor representations with a
margin. It does not provide intermediate arithmetic labels.

Promotion requires at least 95% train exact, 5% seen-N T=1, and 2% OOD-N T=1.
Kill if train exact falls below 95% or OOD-N remains at or below 1%.

Remote run root:
`/home/ubuntu/somn-taskb/runs/t1_factored_e5_interface_noise/seed0`.

No checkpoint from this diagnostic may be loaded into a competition
submission.

## Result

The L40 completed 16,677 updates in 180.01 seconds (92.65 updates/s):

| Profile | Exact | Token accuracy |
|---|---:|---:|
| Training rows | 1,599/1,600 (99.9375%) | 99.9844% |
| Public seen-N T=1 | 2/512 (0.3906%) | 30.4199% |
| Public OOD-N T=1 | 0/512 (0%) | 14.4043% |

The OOD kill threshold fired. Relative to the deterministic anchor, seen-N
fell from 7/512 to 2/512 and OOD-N fell from 1/512 to zero. Seeds 1 and 2 are
therefore not authorized by the registered gate.

## Interpretation

Small isotropic noise prevents perfectly brittle fitting during training, but
does not create the reusable square representation that matters. The model
still memorizes essentially every training row and generalizes worse. This
closes simple smooth robustness regularization at the phase boundary; the next
credit-assignment proposal needs a genuinely discrete or cross-example
constraint rather than another noise magnitude sweep.

The incremental GPU backup verified 15 files and 143,757 bytes at the local
artifact root for Prime pod `0c1aba701be94af3bb8494f88e962a53`.

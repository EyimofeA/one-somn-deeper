# Public-E5 support ablation for the factored T=1 tape

Status: refuted at the registered seed-0 gate; no additional seeds run.

This is a one-variable follow-up to
[`../2026-08-08_t1_phase_square_reduce/NOTE.md`](../2026-08-08_t1_phase_square_reduce/NOTE.md).
The factored model, optimizer, depth, seed, final-label-only objective, and
180-second clock remain fixed. Only the data support changes from 18 tiny
two-digit training moduli to the public Easy e5 T=1 rows. Evaluation uses the
public e5 seen-N and OOD-N T=1 profiles.

Remote lifecycle:

- Prime pod: `0c1aba701be94af3bb8494f88e962a53`
- SSH alias: `oneL40`
- Remote run root: `/home/ubuntu/somn-taskb/runs/t1_factored_e5_support`
- Local backup root: `diagnostics/artifacts/prime-0c1aba701be94af3bb8494f88e962a53/`

No checkpoint from this diagnostic may be loaded into a competition
submission.

## Result

The L40 completed 18,019 optimizer updates in 180.01 seconds (100.10
updates/s). The unchanged 443,594-parameter tape fit all 1,600 public T=1
training rows exactly, but generalization collapsed:

| Profile | Exact | Token accuracy |
|---|---:|---:|
| Training rows | 1,600/1,600 (100%) | 100% |
| Public seen-N T=1 | 7/512 (1.3672%) | 29.9805% |
| Public OOD-N T=1 | 1/512 (0.1953%) | 13.3301% |

Both preregistered kill thresholds fired. The OOD result is far below the
prior tiny-support experiment's 17.29% median, so running seeds 1 and 2 would
violate the gate.

## Interpretation

Increasing modulus support does not repair the final-label identifiability
problem. The model has enough capacity and optimization time to memorize the
public T=1 mapping perfectly while failing almost every new modulus. This
rules out "not enough modulus diversity" as the primary explanation for the
earlier factored tape's failure. The next experiment must alter legal credit
assignment or enforce a robust reusable interface; another topology or
training-support expansion is not justified.

Verified raw source, stdout, and structured metrics are preserved under the
ignored local backup root named above. Backup verification matched 9 files and
89,710 bytes.

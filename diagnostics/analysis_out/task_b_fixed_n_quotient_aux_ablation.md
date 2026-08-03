# Task B Phase 5–6: fixed-N error diagnosis and quotient auxiliary

## Fixed-N=1349 final baseline diagnosis

The final standard-Transformer baseline averages **33.65±2.95%** held-out-u exact across seeds 0–2 (36.60, 30.70, 33.65); token accuracy is 61.32±0.96%.

| Observation | Evidence |
|---|---|
| Generalization weakens with quotient scale | q-digit length: 94.4% (1 digit), 39.2% (2), 38.0% (3), 24.6% (4), in seed-0 final diagnostic. |
| It is not a copy/interpolation shortcut | u unchanged=4.0% EM; u mod 10/100/1000=1.5/0.4/2.7%; nearest-train-u=0%; model agreement with each is ≤8.2%. |
| Remainder geometry matters | nearest-multiple-distance quartiles: 54.9%, 60.6%, 29.3%, 9.6%; q≥10 is 33.9%. |
| Digit errors propagate | digit positions MSB→LSB: 77.1%, 58.4%, 64.2%, 46.8%; 82.6% of wrong examples have one contiguous error run. |
| A trivial heuristic is competitive but not what the model copies | nearest-multiple heuristic=39.7% EM; model agrees with it only 11.0%. |

Full data: `analysis_out/task_b_final/fixed_n_1349_v2/task_b_analysis.json`.

## Competing hypotheses before intervention

1. **Quotient estimation failure.** Supported by the q-length decline and lower performance at high u/N. Against: remainder geometry and contiguous digit error also matter. Falsifier: direct quotient labels do not improve held-out-u against a semantic control.
2. **N-routing/interference failure.** Supported by the one-N→two-N collapse. Against: N-broadcast and counterfactual analysis show models already alter predictions for N but remain wrong. Falsifier: no correct-broadcast advantage over shuffled broadcast (observed).
3. **Serial remainder/borrow-state failure.** Supported by contiguous output runs and position-specific error; against: not directly tested in this parallel Transformer branch. Discriminating test requires a learned serial reduction workspace, outside the current no-recurrence branch.

## Controlled intervention

All runs retain N=1349, 8k/2k disjoint-u data, 4L d128 Transformer, AdamW 3e-4/wd .01, batch 512, budget/early-stop schedule, and seeds 0–2. Both auxiliary variants add the same 1,290-parameter 10-way auxiliary head at the four output states and use λ=.25. Both passed quotient-label/direct-autograd and 32-row 100% smoke tests.

| Final condition | Params | Held-out-u EM per seed | Mean ± sd | Token mean ± sd |
|---|---:|---:|---:|---:|
| baseline | 799,498 | 36.60, 30.70, 33.65 | **33.65 ± 2.95** | 61.32 ± 0.96 |
| quotient-digit auxiliary | 800,788 | 33.50, 21.40, 33.25 | **29.38 ± 6.91** | 59.35 ± 3.65 |
| u-copy auxiliary control | 800,788 | 22.50, 21.90, 22.65 | **22.35 ± 0.40** | 57.31 ± 0.43 |

## Decision

Quotient supervision is **refuted at λ=.25 under this standard-Transformer branch**: it does not improve the baseline and is only better than a deliberately non-semantic auxiliary. This does not prove quotient information is useless in every mechanism; it shows that adding a parallel output-side quotient head does not teach a reusable reduction algorithm.

The current parallel-standard-Transformer branch is now falsified for Task B: it fails fixed-N unseen-u reduction, collapses at two N, does not benefit from paired-u counterfactual data, direct N broadcast, or quotient auxiliary supervision. The next justified experiment—**not launched here**—is a learned serial remainder/subtraction workspace with a capacity-matched non-serial control, because that is the remaining untested serial-state hypothesis.

Artifacts: `diagnostics/runs/{mod_fixed_n_1349,task_b_fixed_n_baseline,task_b_fixed_n_quotient_aux,task_b_fixed_n_u_aux}/`.

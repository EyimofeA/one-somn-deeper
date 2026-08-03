# Task B canonical matrix — `(N, u) -> u mod N`

**Frozen evaluation:** 2026-07-30 on `oneL40` (NVIDIA L40S), using the current
`diagnostics/data/generated/mod/` manifests. Exact/token metrics below are from
`runs/*/{train_,}eval_report.json`. Error reports are in `task_b/<run>/<split>/task_b_analysis.json`.

## Common varying-N data

- Train N: 90 10–11-bit semiprimes; evaluation `val_iid` and `heldout_u` reuse those N.
- Held-out N: 30 disjoint 10–11-bit semiprimes (`heldout_modulus`); held-out-factor is a stricter disjoint-factor split.
- Every N-conditioned validation split has u disjoint from train u. `u` is stratified over `[0, (N-1)^2]`: `u<N`, near multiples, high quotient, remainder near 0, and remainder near `N-1`. Thus `q=floor(u/N)` ranges from 0 to roughly `N-2`.
- Fixed-N data uses N=1349 (an 11-bit modulus from the varying-N train pool), 8k train and two disjoint 2k evaluation splits. The fixed domain cannot support 100k distinct rows while retaining the varying-N sampler's tiny-u strata; its wrapper retains stratified samples then fills with uniform unseen u.

| Experiment / status | Script/config | Architecture | N condition and eval split | Train EM / token | Val EM / token (frozen) | Seed | Checkpoint / logs | Completed? / reproducible? |
|---|---|---|---|---:|---:|---:|---|---|
| `mod_transformer` | `train.py`; `runs/mod_transformer/config_used.yaml` | standard Transformer, 4L d128 h4 FF512; AdamW 3e-4, wd .01, bs64, 20 epochs | varying N; `val_iid` seen N / unseen u | 25.77 / 53.96 | 23.65 / 51.67 | 0 | `runs/mod_transformer/peak.pt`, `final.pt`, `metrics.jsonl` | **Yes**: log reaches 31,240 of ~31,260 expected updates. Re-evaluable on current data; prior saved report was 27.64%, so the original manifest/report pairing was not preserved. |
| `mod_50k/seed0` | `train.py`; `runs/mod_50k/seed0/config_used.yaml` | same Transformer; AdamW 3e-4, wd .01, bs512, planned 50k | varying N; `val_iid` seen N / unseen u | 25.75 / 60.79 | 18.98 / 51.10 | 0 | `final.pt`; no original metrics | Final checkpoint evaluates deterministically; **completion unverified** (no training log, early stopping enabled). |
| `mod_50k/seed1` | same, `seed1/config_used.yaml` | same | varying N; `val_iid` seen N / unseen u | 29.37 / 62.71 | 18.66 / 49.26 | 1 | `final.pt`; no original metrics | Final checkpoint evaluates deterministically; **completion unverified**. |
| `mod_50k/seed2` | same, `seed2/config_used.yaml` | same | varying N; `val_iid` seen N / unseen u | 26.92 / 61.05 | 19.46 / 50.86 | 2 | `final.pt`; no original metrics | Final checkpoint evaluates deterministically; **completion unverified**. |
| `mod_fixed_n_1349` | `train.py`; `configs/mod_fixed_n_1349.yaml` | same Transformer and optimizer as `mod_50k`; bs512, planned 50k | **fixed N=1349**; `heldout_u` disjoint u | 91.79 / 97.28 (peak) | 36.30 / 60.96 (`heldout_u`); selection-val 53.45 / 75.98 | 0 | `runs/mod_fixed_n_1349/{peak,final}.pt`, `metrics.jsonl`, `eval_report.json` | **Yes**, early-stopped at 16k after no new selection-val peak for 10 evaluations. Re-evaluable. |

### Same checkpoints, N-generalization cells

| Run | heldout_u EM / token (seen N, unseen u) | heldout_modulus EM / token (unseen N) | heldout_factor EM / token | length12 EM | length13 EM |
|---|---:|---:|---:|---:|---:|
| `mod_transformer` peak | 22.97 / 50.75 | 27.31 / 54.61 | 27.06 / 53.95 | 21.32 | 15.34 |
| `seed0` final | 18.24 / 50.56 | 21.95 / 53.16 | 21.97 / 52.46 | 14.75 | 7.65 |
| `seed1` final | 18.49 / 49.55 | 21.67 / 50.84 | 22.14 / 50.28 | 11.72 | 2.49 |
| `seed2` final | 19.07 / 50.61 | 22.27 / 53.26 | 22.33 / 52.40 | 13.94 | 6.64 |

## Two-modulus control (the dynamic-conditioning boundary)

`configs/mod_two_n_1349_1357.yaml` changes only one-N training to two 11-bit train-pool moduli (1349 and 1357), retaining the same standard Transformer, optimizer, total 8k train rows, 2k evaluation rows, and 50k-step/early-stop policy. Six labels and a balanced 32-row smoke batch were checked before the full run; smoke reached 100% at step 200.

| Checkpoint | Train EM / token | Selection-val EM / token | Independent held-out-u EM / token | Per-N held-out-u EM |
|---|---:|---:|---:|---:|
| selection peak, step 1k | 21.81 / 49.80 | 15.65 / 49.18 | 15.10 / 50.16 | N=1349: 15.10; N=1357: 15.10 |
| final, step 11k | 96.94 / 97.93 | 13.55 / 44.67 | **11.95 / 43.33** | not separately evaluated; aggregate is sufficient |

The two-N model memorizes both training mappings but loses nearly all held-out-u generalization. This removes the 90-N data-volume confound as the primary explanation: the transition from one to two moduli is already enough to produce the failure.

### Paired-u intervention

`configs/mod_two_n_paired_u.yaml` makes the two-N train and evaluation u values counterfactual: each u occurs once under each modulus. It retains 8k/2k row counts, moduli, model, optimizer, and budget. It **does not** help: selection peak reaches 95.30% train EM / 14.25% selection-val EM / **10.75% independent held-out-u EM**; final is 96.64% / 13.40% / **10.45%**. The standard Transformer therefore does not fail merely because unpaired u allows it to ignore N; explicit paired evidence still does not yield a reusable N-conditioned reduction procedure.

## Completed non-canonical reduction mechanisms

These were actually run, but are **not** canonical Task B controls: they use a learned long-division recurrent cell at fixed `N=323`, not the standard Transformer or the 10–11-bit diagnostic data. Their raw metrics/checkpoints/manifests are absent locally, so only the recorded `predictions.md` results are reproducible as prose, not as artifacts.

| Experiment / script | Fixed/varying N; u distribution | Recorded result | Seed / logs / reproducibility |
|---|---|---|---|
| `pure_reduction_cell.py` + `generate_pure_reduction.py` | fixed N=323; 8k/2k held-out P; uniform P in `[0,99,999,999]`, q up to ~309,597 | train about 100%; held-out peak **0.60%** EM | generator seed 45; model seed, raw metrics, checkpoint, exact command missing |
| `pure_reduction_cell_v2.py` + `generate_pure_reduction_v2.py` | fixed N=323; 8k/2k held-out P; reciprocal/log-uniform P, same numerical range | peak **78.45%** EM; stable 69–84% at 78–80k; P>=323 75.5% at step 12k | generator seed 45; recorded as 80k-step run, but raw metrics/checkpoint missing |
| `learned_reduction_cell.py` + `generate_reduction_cell_fixed_n.py` | fixed N=323; input is x and intermediate x², not direct u; 230/58 held-out x | peak **5.17%** then **1.72%** | generator seed 45; raw metrics/checkpoint missing; this is Task C-composed, not Task B |

## Evidence from the completed standard-Transformer runs

- The model **does not fit** the varying-N training set: train EM is only 25.75–29.37%.
- It nearly solves the identity subcase `u<N` (99.4–100.0% EM), but q>=10 is only 2.9–9.9% on `val_iid`, `heldout_u`, and `heldout_modulus`; see the per-split JSON error reports.
- With fixed N, it fits (91.79% peak-checkpoint train EM) and holds **36.30%** on a never-selected, disjoint-u test. q=0 is 100%; q>=10 is 33.4%. The 53.45% selection-val score is not the test estimate because that split drove early stopping.
- **Classification: 2 — failure to condition dynamically on varying N.** Held-out N is not the specific failure: it is no worse than seen N in the varying-N runs. Fixed-N performance is imperfect, but far above the varying-N result; its smaller dataset is an explicit remaining confound.

Excluded deliberately: Task A carry/diagonal auxiliary-loss experiments and all Task C `(N,x)->x² mod N` experiments.

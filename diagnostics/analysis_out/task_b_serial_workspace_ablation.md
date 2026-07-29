# Task B: serial workspace versus depth control

## Setup

Fixed N=1349, existing 8k train / 2k independent held-out-u data, AdamW 3e-4/wd .01, batch 512, same early-stop budget, seeds 0–2. All new models passed direct forward/backward and 32-row 100% smoke checks.

| Condition | Parameters | Mechanism |
|---|---:|---|
| Existing baseline | 799,498 | 4-layer d128 standard Transformer |
| Deep non-recurrent control | 877,810 | 5-layer d120 standard Transformer |
| Recurrent workspace K=8 | 845,434 | frozen N/u context; 8 learned workspace tokens; one tied self/cross-attention/FFN transition applied 8 times; final-only output decode |

## Held-out-u exact match

| Condition | Peak per seed | Peak mean ± sd | Final per seed | Final mean ± sd |
|---|---:|---:|---:|---:|
| baseline | 36.30, 35.90, 32.05 | 34.75 ± 2.35 | 36.60, 30.70, 33.65 | 33.65 ± 2.95 |
| deep control | 32.90, 41.60, 40.20 | **38.23 ± 4.67** | 29.35, 39.85, 37.75 | **35.65 ± 5.56** |
| recurrent K=8 | 22.55, 31.80, 33.35 | 29.23 ± 5.84 | 16.20, 30.35, 23.85 | 23.47 ± 7.08 |

All final checkpoints fit train (baseline 98.03%, deep 98.06%, recurrent 97.20% mean exact), so recurrent failure is not inability to memorize the data.

## Final held-out-u arithmetic breakdown, mean across seeds

| Condition | q=1 digit | q=2 digits | q=3 digits | q=4 digits | q>=10 | Output positions MSB→LSB |
|---|---:|---:|---:|---:|---:|---|
| baseline | 94.0 | 35.0 | 34.5 | 22.5 | 30.8 | 77.5, 58.0, 65.8, 43.9 |
| deep control | 96.3 | 35.9 | 36.4 | 24.9 | 32.8 | 78.6, 59.8, 67.2, 45.5 |
| recurrent K=8 | 93.6 | 20.3 | 22.6 | 15.3 | 20.2 | 75.1, 56.5, 63.2, 31.1 |

Contiguous wrong-run fraction: baseline 81.6%, deep 81.2%, recurrent 84.2%.

## Recurrent depth diagnostic

At each recurrent seed's selected checkpoint, evaluating with fewer shared transitions rises monotonically toward K=8. Example seed 1 held-out-u EM by K: 0.0, 0.1, 4.3, 15.9, 23.5, 27.6, 30.1, 31.7%. More iterations are used, but K=8 does not beat the non-recurrent control.

## Conclusion

Recurrence is **refuted for this workspace formulation**. The data do not support a K sweep: the gate required recurrence to beat both controls, and it does not. The model has enough capacity to fit train and benefits from extra transitions, but its fixed learned workspace / tied-transition representation fails to transfer high-quotient reduction. Late decay exists, but selected recurrence checkpoints are already below baseline, so optimization decay is not the sole explanation.

**Next recommendation, not launched:** keep K=8 and the shared transition fixed; replace only the fixed learned workspace initialization with a learned projection of the current input context (N/u), compared to an equal-parameter shuffled-context initialization control. This directly tests workspace state representation without adding auxiliary labels or a solver.

Artifacts: `diagnostics/runs/task_b_fixed_n_{deep_control,recurrent_k8}/`; detailed final error JSONs: `diagnostics/analysis_out/task_b_serial/`.

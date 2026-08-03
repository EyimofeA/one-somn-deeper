# Task B Phase 1–3: N-broadcast ablation

**Question:** does a direct learned route from the four N digits to each output slot repair two-modulus held-out-u reduction?

## Fixed setup

- Data: existing unpaired two-N data, N={1349,1357}, 8k train / 2k selection-val / 2k independent held-out-u.
- Seeds: 0, 1, 2. Baseline seed 0 is the completed `mod_two_n_1349_1357` run; seeds 1–2 were replicated.
- Backbone: 4-layer d128, 4-head, FF512 Transformer; AdamW 3e-4/wd .01, bs512, same early-stop rule.
- Correct broadcast: mean-pool the input hidden states of N's four digits; apply a learned 128→128 projection unique to each layer; add it to all four output slots after that layer.
- Shuffled semantic control: identical model and parameters, but the broadcast receives another batch example's N representation, mismatched whenever possible. Input N tokens to the main Transformer remain correct.
- All new conditions: direct forward/backward assertion, wrong-N permutation assertion, and 32-row 100% memorization smoke passed. An earlier contextual-state shuffled control failed smoke because it also shuffled another example's U context; it is retained as `smoke32` and excluded. The valid input-N control is `smoke32_v2`.

## Final checkpoints (independent held-out-u)

| Condition | Params | Train EM, mean ± sd | Held-out-u EM, mean ± sd | Held-out-u token, mean ± sd | Mean throughput |
|---|---:|---:|---:|---:|---:|
| baseline | 799,498 | 96.78 ± 0.14 | **11.27 ± 0.63** | 42.08 ± 1.42 | 112.0 steps/s |
| correct N broadcast | 865,546 | 96.78 ± 0.07 | **11.55 ± 0.30** | 42.01 ± 1.07 | 95.5 steps/s |
| shuffled N broadcast | 865,546 | 91.82 ± 0.85 | **11.35 ± 0.39** | 41.25 ± 0.61 | 94.3 steps/s |

Selection-peak held-out-u EM was baseline 13.58 ± 1.64, correct 11.95 ± 0.73, shuffled 11.75 ± 0.76. Broadcast therefore gives no material, reproducible gain over either control.

## Counterfactual N sensitivity (final checkpoints)

A shared 2,000-u evaluation set was sampled below `(1349-1)^2`, excluding all training u under either modulus. Each u was evaluated under both N. True remainders differed in 100% of pairs.

| Condition | Prediction changes with N | Prediction unchanged despite target change | Correct both modulus-specific answers | Responds but wrong under both |
|---|---:|---:|---:|---:|
| baseline | 92.55 ± 1.65% | 7.45 ± 1.65% | 0.00% | 92.38 ± 1.70% |
| correct broadcast | 95.10 ± 0.90% | 4.90 ± 0.90% | 0.00% | 94.83 ± 0.85% |
| shuffled broadcast | 95.02 ± 1.38% | 4.98 ± 1.38% | 0.00% | 94.78 ± 1.37% |

## Classification

**Case C:** neither meaningful N broadcast nor its matched shuffled control improves two-N held-out-u generalization. The main Transformer already responds to N on most counterfactual pairs; it responds incorrectly. This falsifies the narrow claim that a short N→output route is the primary bottleneck.

Artifacts: raw run directories `diagnostics/runs/{mod_two_n_1349_1357,task_b_two_n_baseline,task_b_two_n_n_broadcast,task_b_two_n_shuffled_n_broadcast}/`; counterfactual JSON/JSONL/logs in `diagnostics/analysis_out/task_b_counterfactual/`.

# Independent critique request — Task B N-routing intervention

We are diagnosing local Task B: input decimal tokens `(N,u)`, output decimal tokens `u mod N`. This is research-only, not a competition submission.

## Fixed baseline

All standard-Transformer runs use 4 layers, d=128, 4 heads, FF=512, AdamW lr=3e-4/wd=.01, batch 512, 50k planned steps and early stopping. Outputs are 4 decimal digits.

| Data | Checkpoint | Train EM | Independent held-out-u EM |
|---|---|---:|---:|
| 90 varying 10–11-bit semiprime N, ~100k rows | best existing | 25.8–29.4% | 18.2–23.0% seen-N; 21.7–27.3% held-out-N |
| one N=1349, 8k rows | final | 98.36% | 36.60% |
| two N={1349,1357}, 8k rows (u samples independent by N) | final | 96.94% | 11.95% |
| same two N, paired u under both N | final | 96.64% | 10.45% |

All test u are disjoint from train u. The model gets `u<N` at ~100%, while actual reduction (`q=floor(u/N)>=10`) is poor. Every condition passed a 32-example 100%-memorization smoke test and manual label checks.

## Current reading

The one-to-two-N transition destroys unseen-u generalization despite high train fit. Pairing each u with both N does not repair it. We classify the bottleneck as failure to form a reusable N-conditioned reduction computation, not an N-heldout failure. The strongest remaining confound is that fixed/two-N data uses 8k/2k rows (the fixed domain cannot provide the varying generator's 100k distinct stratified rows).

## Proposed next run: learned N broadcast

Keep the **unpaired two-N dataset**, optimizer, 50k budget, and all Transformer widths/depth unchanged. Replace the backbone only with a model that, after each Transformer layer:

1. mean-pools the four hidden states corresponding to N's decimal digits;
2. maps that pooled state through a learned d→d linear projection;
3. adds it to each of the four output-slot states.

Each layer has its own projection. This adds no hard-coded arithmetic, no quotient/remainder labels, no recurrence, and no auxiliary loss. It is intended to test whether the ordinary Transformer lacks a reliable learned route from N tokens to output slots.

## Critique requested

1. What is the weakest inference in the current evidence?
2. Does the proposed N broadcast actually test routing, or is it likely to add only capacity / an N-ID lookup?
3. What one simpler falsifier or control is required before interpreting an improvement as reusable N-conditioned reduction?
4. Should we proceed unchanged, modify it, or abandon it?

Evidence artifacts: `diagnostics/analysis_out/task_b_canonical_matrix.md`, per-split reports under `diagnostics/analysis_out/task_b/`.

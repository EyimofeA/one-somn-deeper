# Noise / significance (Easy)

## What the evaluator fixes

Easy manifests set `runtime.seeds: [74]` only — we **cannot** multi-seed inside one submit. Score = mean exact over test+ood for that one train seed.

So a single card is **one train trajectory** (plus GPU/bf16/wall-clock jitter). Tiny deltas are not “wins.”

## Observed effect sizes we have been calling wins

| Comparison | Δ mean exact |
|------------|--------------|
| UT K2 → train2/eval4 (e1) | +0.33 pp (6.50 → 6.83) |
| plain K4 → UT K4 (e5) | +0.20 pp (0.80 → 1.00) |
| midloop vs UT (e1) | **−3.9 pp** (clear) |

At ~5–7% absolute, **±0.3 pp** is in the “maybe noise” zone until measured.

## How many runs to get a band

Same `submission.py`, same dataset, resubmit **n** times (seed still 74; variance = clock cutoff + GPU nondeterminism).

| Goal | n (repeats of *one* config) | Cost |
|------|-----------------------------|------|
| Rough σ (order of magnitude) | **3** | 3 Easy |
| Usable mean ± band for promotions | **5** | 5 Easy |
| Claim A > B when |Δ| ≈ 0.3 pp | **5 each** (A and B) or demand larger |Δ| | 10 Easy |

**Practical protocol (Karpathy + tired-proof):**

1. Pick one champ (e.g. `depth_d32_k2_ut_evalk4` on e1).
2. Burn **3–5** identical e1 submits → record mean ± sample sd of `mean_exact_accuracy`.
3. Only promote a new arch if it beats champ mean by **≳ 2σ** *or* wins on **both** e1 and e5 by a clear margin (≳1 pp), not a 0.3 pp e1-only bump.
4. e5 noise: same idea but absolute % is ~1%, so demand **≳ 0.3–0.5 pp** or dual-gate before caring.

All current learned scores are still **bad** in absolute terms (≪ solver 100%, ≪ “solved”). Noise bands decide *ranking among weak models*, not whether the task is cracked.

## Budget sketch

With ~20 Easy “for us”: **5** for e1 noise on champ + **3** for e5 noise on UT K4 = **8**, leave **12** for real one-change cards that must clear the band.

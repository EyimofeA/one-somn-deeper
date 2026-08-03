# 2026-07-29: Research Critique — Current Direction vs "Back to Basics"

## Current direction: tune STE + aux + wd on Easy e5

### Weakest assumption
**Easy e5 (60s, ~500 steps) is informative about what works on Hard (3600s, 190k steps).**

The Hard grokking transition happens at ~64,000 steps. Easy gives us ~500. We are optimizing in a regime that *by definition* cannot show grokking, generalization, or composition. We're tuning hyperparameters on a proxy that measures pre-grok memorization capacity. Every Easy card could be a local optimum of the wrong thing.

Evidence: Hard H2 trained to 100% exact / 0% eval. More of the same architecture with more steps = more memorization, not more generalization.

### Under what conditions does this fail completely?
If the true solution requires (a) a phase transition at ~64k steps AND (b) an architectural inductive bias that only matters *after* the transition. Then every Easy test is blind — we're optimizing pre-transition behavior and hoping it correlates with post-transition behavior. It probably doesn't.

### What simpler experiment would falsify faster?
Locally, on the GPU box, run the current best architecture (UT K4, wd=0.1) for 200k steps on a fixed-N squaring task. Does it grok? If yes → Easy results are pre-grok noise and we should tune on the GPU box. If no → the architecture can't grok and we need a different inductive bias entirely.

### Prior work that contradicts
The grokking literature (Power et al. 2022, Nanda et al. 2023) shows that weight decay, data size, and optimizer choice determine *whether* grokking happens, but the architecture determines *what* the model groks to. A memorization-friendly architecture will grok to memorization. The model needs an architectural reason to prefer the algorithmic solution.

---

## User's idea: "Back to basics — minimal ground-up path requiring GPUs"

### What this means (my interpretation)
Start from first principles: what is the simplest architecture that can learn *one step* of the recurrence, then compose it? Test on the GPU box (not Easy) where we can run long enough to see grokking. Build the minimal thing that works, then scale.

### Weakest assumption
**The primitives are learnable in isolation, therefore composition is just an engineering problem.**

The pure squaring diagnostic (18.65% exact, stable) and pure reduction (78.45% exact) both work in isolation. But the composed cell failed (5.17% → 1.72%). The gap isn't "we haven't tried enough architectures" — it's that composition itself is the hard problem. Going "back to basics" might just rediscover that the primitives work and composition doesn't.

### What this idea gets right
- Easy is a broken signal. Local GPU runs with full grokking budget are the right measurement.
- Simpler architectures → faster iteration → more experiments per GPU hour.
- The one-step squaring + reduction isolation experiments are the right diagnostic toolkit.

### What this idea misses
- "Basics" is underspecified. Which basics? A single transformer block? A weight-tied RNN? A NALU cell?
- The current architecture (UT K4, d=32) IS already a minimal recurrent block. It's 39K params. "Simpler" from here means stripping to a pure RNN, which was tried (BiGRU baseline, 0.67% e1).
- The gap isn't architecture complexity — it's that no architecture has learned to compose the learned step.

### Blind spot (what neither direction addresses)
**The model might need to SEE composition during training.** If T is always 1-3 in training, the model never sees the composition it needs to generalize to. The T=1 bootstrap idea (train on T=1 until perfect, then T=2, then T=3...) was mentioned but never run. This is the simplest "back to basics" experiment that directly targets the composition gap.

## Synthesis

| | Current direction | Back to basics |
|---|---|---|
| **Agreement** | Easy is pre-grok, not informative | Easy is broken, use GPU box |
| **Disagreement** | Tune existing arch on short clock | Start over with simpler arch on long clock |
| **Blind spot** | Neither tests curriculum learning (T=1 → T=2 → T=3) | |

## Decision

**Modified: curriculum on GPU box.** Don't start over from scratch. Don't keep tuning on Easy. Instead: take the working UT K4 baseline, run it locally on the L40S with a curriculum (train on T=1 only until 95%+, then add T=2, then T=3). Let it run for 200k steps with wd=0.1. Measure whether it learns a reusable step that composes. This directly tests the composition hypothesis using the architecture we know works, on hardware that gives us the grokking budget. One experiment, one clear answer.

Cost: ~30 minutes on L40S. Zero quota.
# 13 — Decisions glossary (FiLM, ACT, ponder, Easy sets)

We do **not** know which internal algorithm the net will learn. Arch choices are bets about *inductive bias + clock*, not claims that the model “does modular squaring.” Log every bet so final writeups can reconstruct why.

## What we log (yes)

| Artifact | Role |
|----------|------|
| `solving/RESEARCH_LOG.md` | Append-only hypothesis → setup → scored facts → next |
| `solving/experiments/metrics/*.jsonl` | Raw train/eval/summary from `one-layer metrics` |
| `solving/experiments/EXPERIMENT_LOOP.md` | Phase tables + funnel |
| `solving/experiments/figures/` | Script + PNG per comparison |
| `learnings/concepts/11-ideas-backlog.md` | Yours / ours / researched / rejected |

Missing from early cards: plain-language definitions of gadgets (this file).

## FiLM (Feature-wise Linear Modulation)

[SOURCED] Perez et al., AAAI 2018 — https://arxiv.org/abs/1709.07871

Given a conditioning vector `c` (for us: pooled N-digit embeddings) and hidden features `h`:

- Predict per-channel scale and shift: `(γ, β) = Linear(c)`
- Apply: `h′ = γ ⊙ h + β` (we used `h′ = (1+γ) ⊙ h + β` so init ≈ identity)

**Why people use it:** cheap way to let one signal (question, class, modulus) turn channels up/down without rewriting the whole block.

**Our bet:** force every loop to “see” N. **Measured:** e1 ≈ flat, e5 worse → reject that recipe, not the whole idea of conditioning.

## ACT + ponder loss

[SOURCED] Graves, Adaptive Computation Time (2016) — https://arxiv.org/abs/1603.08983  
[SOURCED] Dehghani et al., Universal Transformers (2018) — https://arxiv.org/abs/1807.03819

**ACT:** after each reuse of a shared block, a halt head outputs a probability; computation can stop early (or soft-mix intermediate states).

**Ponder cost / ponder loss:** an extra term that **penalizes using many steps**, so the model does not always run to `K_max`.

```text
L = L_task + τ · (mean ponder steps)
```

`τ` (tau) is a knob: larger τ → prefer fewer loops; smaller τ → prefer accuracy over thrift.

**Ponder “scaling curves”** usually means: plot task metric vs `τ` (or vs average steps used) — *not* the same as our width/steps scaling plots in `figures/fig_scaling_*.png`.

**Our soft-ACT card** mixed states but **had no ponder term** and no halt-bias init trick from later UT practice. That is an incomplete ACT, not a full Graves/UT recipe.

## Easy e1–e5 — what actually varies

From competition `service/tiers.py` labels + manifest `data_root` names:

| ID | Varies mostly | Label |
|----|---------------|-------|
| **e1** | Fixed modulus **N=323**, T∈{1,2,3} | easiest fixed-N depth slice |
| **e2** | Fixed **N=899**, T∈{1,2,4} | another fixed N; T set includes 4 |
| **e3** | **Many N** (10–11 bit), **fixed T=2** | modulus variety, depth fixed |
| **e4** | Larger N (11–12 bit), **fixed T=2** | harder moduli, depth fixed |
| **e5** | Many N (10–11 bit), T∈{1,2,3} | modulus + mild depth |

Medium m1–m5 push larger N and/or larger T (up to 16) with 600s train.

## Literature default we under-implemented

Universal Transformer = **weight-tied block + depth/timestep embedding each loop** (± ACT+ponder).  
We shipped tied loops **without** depth embeddings, then a soft-ACT without ponder. Next literature-faithful card: fixed-K loops **with** depth emb, same d=32, gate e1+e5 vs `depth_d32_k2` / `depth_d32_k4`.

## Hard reality check (leaderboard snapshot)

Hosted Hard best ≈ **0.19%** exact (CLI `one-layer leaderboard`, 2026-07-21). Exact-100% on Hard with tiny Easy-tuned models is not what the public board shows anyone doing.
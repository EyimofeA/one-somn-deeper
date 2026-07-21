# 12 — Current architecture and naming

## Name map (read this once)

| Symbol in our folder names | Means | Not to be confused with |
|----------------------------|-------|-------------------------|
| **d** / `d_model` | **Width** — embedding / residual channel size | Serial squaring depth T; number of stacked layers |
| **K** / `num_loops` | How many times one **shared** block runs in `forward` | Problem T (number of modular squarings); LLM “depth” as untied layers |
| **T** (data) | Number of modular squarings in the label | Our loop count K |
| **heads** | Attention heads inside the block | — |

We kept **K** in paths for now (renaming to something like `L` or `loops` later would churn metrics filenames). Mentally: **d = width, K = loop count**.

Transformer **width** is exactly `d_model`. Expanding width grows QKV/FFN matrices ≈ O(d²). Expanding **K** reuses the same matrices — more compute, same parameter count.

## What we are running now (reference)

**`depth_d32_k4`** — Pre-LN Transformer block, width 32, 4 heads, weight-tied, applied **K=4** times.

```text
 tokens (BOS N <digits> X <digits> T <digits> ANS …)
        │
        ▼
   Emb(d=32) + Pos
        │
        ▼
   ┌─────────────────────────┐
   │  Shared Block           │  ← same weights
   │  RMSNorm → Attn → +     │
   │  RMSNorm → FFN(4×) → +  │
   └───────────┬─────────────┘
        × K=4 (loop)
        │
        ▼
   RMSNorm → tied LM head → logits
```

**Params @ smoke vocab:** ≈ **16K** (≈ **0.003%** of the 500M element ceiling). Wider cousins (d=128) are still only ~0.04% of the ceiling.

## Why so small? Will we underfit?

Two different ceilings:

1. **Parameter ceiling (500M)** — we are nowhere near it. Raw capacity is not the Easy bottleneck.
2. **Wall-clock ceiling (60s Easy)** — each extra flop in `forward` burns steps. Width sweeps already showed **wider can lose** on Easy because steps drop and the model barely fits.

So “underfitting” here is mostly **algorithmic**: the network must learn a composition that depends on N and on T. Tiny tied loops help OOD T on fixed-N e1; they still fail when N varies (e5). Dumping parameters without a better bias often **wastes** the Easy clock. Medium (600s) / Hard (3600s) are where larger state may start to pay — after an Easy funnel picks winners.

## What N-conditioning was (tried, rejected)

Prompt tokens look like: `N` then digits of the modulus, then `X` ….  
**N-cond** mean-pooled the embeddings of those modulus digits into one vector `n`, then each loop did FiLM: `h ← (1+γ(n)) ⊙ h + β(n)` before the shared block.

Intent: force every loop to “know which ring ℤ/Nℤ it is in.”  
Result: e1 ≈ unchanged; e5 got worse — so that particular FiLM recipe is out.

## Adaptive loops (ACT) — tried next

Same tied block, but up to **K_max=8** passes. After each pass a halt head outputs a probability; the final hidden state is a **soft weighted mix** of the intermediate states (remainder mass forced on the last step). Idea: harder examples (larger T / messier N) keep looping; easy ones halt early.

Scored: e1 mean 3.83% (below fixed K=4); e5 mean 0.79% (≈ fixed K=4).

## Funnel (your protocol)

1. **Easy** — many arch/ablation cards (e1 + e5 gate).
2. Promote **best ~5–10** Easy configs → **Medium** (6/day).
3. Best Medium that day → **one Hard** (your approval).

LR-schedule day is separate; current recipe stays warmup + cosine AdamW until that day.
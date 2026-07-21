# Expected loss vs what we see

## Theory (if loss were mean token CE)

Vocab size for squaring_mod markers+digits is **V = 17**. Uniform random:

```text
H = ln(17) ≈ 2.83 nats per token
```

A strong model on easy digits might land near **0.1–1.0** mean CE. Exact-match can still be low even when CE looks “fine.”

## What the evaluator actually logs

Our Easy e1 JSONL starts at **loss ≈ 14–84** at step 1 and ends near **1.2–1.9** after 60s. That scale is **much larger than ln(17)**, so the logged value is almost certainly **not** a simple mean-over-tokens CE (likely summed over positions and/or batch, or another reduction). Treat it as a **relative** curve, not a calibrated cross-entropy.

### Empirical (Easy e1, logged batch loss)

| Model | ver | L_start | L_end | steps |
|-------|-----|---------|-------|-------|
| Transformer | v1 | 83.8 | 1.63 | 261 |
| Transformer | max | 43.5 | 1.64 | 557 |
| MLP | v1 | 80.4 | 1.87 | 287 |
| MLP | max | 44.7 | 1.85 | 585 |
| BiGRU | v1 | 21.6 | 1.21 | 258 |
| BiGRU | max | 13.9 | 1.29 | 555 |

**Expect:** start ≫ ln(17) on this logger; end ≈ **1–2** after 60s for these tiny models; further drops need more time (Medium) or better bias (loops), not just “wait for CE → 0.”

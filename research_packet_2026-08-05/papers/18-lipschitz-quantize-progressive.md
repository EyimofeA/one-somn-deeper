# 18 — Lipschitz drift, quantization margin, progressive loss

Evidence-backed pieces cite metrics. Mechanism lecture (full proofs/paths): [`../readings/one-layer-deeper-notes.md`](../readings/one-layer-deeper-notes.md) Parts 4–8. Hypotheses until we run Part 9: [`../../HYPOTHESES.md`](../../HYPOTHESES.md).

Unicode math only.

## Error recurrence (analog)

Block B with one-step error ε and Lipschitz constant L:

eₖ ≤ L · eₖ₋₁ + ε

Solve: L ≠ 1 → eₖ ≤ ε · (Lᵏ − 1)/(L − 1); L = 1 → eₖ ≤ k·ε.

| L | Behaviour |
|---|-----------|
| L > 1 | eₖ ~ ε Lᵏ — explosion |
| L = 1 | linear growth — still fails exact-match at large T |
| L < 1 | bounded error — but states collapse; lose the value of x |

**Hypothesis H1:** no fixed L yields exact recovery under continuous state. Notes call this Result 3.

**Evidence of confident wrongness after drift:** `claude_hard_h1` eval loss ~15.8 vs ln(10)≈2.3 (`solving/RESEARCH_LOG.md`, Hard H1). That matches “stretched into the wrong basin,” not mild noise.

## Quantized step

D = valid digit-strings. δ = min distance between distinct members of D.  
S(z) = q(B(z)) with q = snap to nearest d ∈ D (STE argmax in train).

**Claim (Result 4 in notes):** if ‖B(d) − f(d)‖ < δ/2 for every d ∈ D, then S(d) = f(d) exactly, hence Sᵀ = fᵀ for every T.

Training usually only sees clean starts within K steps — not the full reachable set. That is why long T fails even when short T looks fine (**Result 5 / H3**).

## Progressive loss (mechanics)

```python
n = random.randint(0, T - 1)
m = T - n
with torch.no_grad():
    z = embed(x0, N)
    for _ in range(n):
        z = quantize(block(z, N))
z = z.detach().requires_grad_()
for _ in range(m):
    z = quantize(block(z, N))
# loss vs true answer at T; consistency aux optional
```

Detached loop: no activation memory; cheap exposure to self-generated states.

## Path D checklist (build order)

1. Quantize between every B apply (STE).  
2. Re-inject N (and x) every loop.  
3. Progressive loss as above.  
4. Entropy aux on pre-quant digit distributions via `auxiliary`.

Measure with frozen `scripts/extrapolation_curve.py` (train T∈{1,2,3}; eval T=4,5,8,16,…). See notes Part 9.

## Related

- Day-1 proposal stub: [`17-recurrence-generalisation.md`](17-recurrence-generalisation.md)  
- Decisions: [`../../RESEARCH_PROTOCOL.md`](../../RESEARCH_PROTOCOL.md)  
- Layout: [`../../solving/experiments/LAYOUT.md`](../../solving/experiments/LAYOUT.md)

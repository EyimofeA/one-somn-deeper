# Playground — position encodings (learning only)

Not competition cards. Read these, run the demos, then decide whether Abacus (or FIRE)
belongs in a one-change Path D experiment.

## Abacus ([McLeish et al. 2024](https://arxiv.org/abs/2405.17399))

**Problem:** transformers lose *which place* a digit is in (ones vs tens vs …).
That hurts column alignment for addition / modular arithmetic.

**Idea:** extra learned embedding keyed by **place within the current number**, not
absolute sequence index. Digits of equal significance (across operands) share the
same place id → they “line up.”

**Train trick:** start place indices at a random offset β ∼ U[1, k] (default k≈100)
so short training numbers still touch high embedding rows → length generalization.
At test, β = 1.

**Paper setup:** LSD-first (digits reversed). Code: [`abacus.py`](abacus.py)
(from [mcleish7/arithmetic](https://github.com/mcleish7/arithmetic)).

```bash
python learnings/playground/demo_abacus.py
```

**For One Layer Deeper:** our prompts are `N … X … T …` with MSD-first decimals.
Abacus still applies if you detect digit spans and assign place ids (from LSD or
MSD — pick one and stick to it). Orthogonal to Path D (quantize/progressive), but
helps sub-problem A (represent one step) when N has many digits.

## FIRE ([Li et al. 2023](https://arxiv.org/abs/2310.04418))

**Problem:** absolute / RoPE-style positions often fail when test length ≫ train length.

**Idea:** relative attention bias from an MLP:

  b(i, j) = f_θ( ψ(i−j) / ψ(max(L, i)) ) ,  ψ(x) = log(c·x + 1)

Normalizing by query position keeps MLP inputs in [0, 1] at any length
(“progressive interpolation”). Abacus paper uses FIRE as a strong *sequence*
positional baseline for addition; Abacus is the *within-number* place signal.

```bash
python learnings/playground/fire.py
```

Code: [`fire.py`](fire.py) — tiny causal MHA + FIRE bias.

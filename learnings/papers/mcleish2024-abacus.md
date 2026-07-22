# McLeish et al. 2024 — Abacus Embeddings

Paper: [arXiv:2405.17399](https://arxiv.org/abs/2405.17399) · Code: [mcleish7/arithmetic](https://github.com/mcleish7/arithmetic) (`abacus.py`)

## Claim

Length generalization on addition fails largely because models cannot track **digit place**
inside long numbers. Abacus adds a learned embedding of place-within-number (same id for
equal significance across operands). With LSD-first formatting + random place offset at
train time, models trained on ≤20-digit addends reach high exact-match on 100-digit addends.

Also stacks with: input injection, looped (weight-tied) transformers, progressive loss —
same themes as our Path D notes.

## Relevance here

- Helps **represent** multi-digit ops (carry / align), not T-composition by itself.
- FIRE is their strong *sequence* RPE baseline; Abacus is *intra-number* place.
- Our competition format is MSD-first + markers `N/X/T` — adapt span detection; do not
  copy their causal LM addition harness wholesale.

Playground: [`../playground/`](../playground/).

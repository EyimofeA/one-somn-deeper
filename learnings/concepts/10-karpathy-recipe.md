# Guiding principle — Karpathy recipe

Source: [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) (2019).

## How we map it here

| Karpathy stage | Our move |
|----------------|----------|
| 1. Become one with the data | `data_samples/`, concepts/01–02 |
| 2. Skeleton + dumb baselines | b0/b1/b2 → `*_max` on Easy e1 |
| 3. Overfit / simple→complex | next: depth loops on Transformer_max; then Medium |
| Visualize everything | `figures/` dashboard from JSONL; no silent complexity |
| One change at a time | experiment cards; ablate K or width, not both |

## Non-negotiables for us

- Hypothesis before submit
- Plot every returned scalar we can
- Prefer Easy ablations over unverified Medium fancy
- No checkpoint fine-tune (API does not return weights) — “iterate” = new `submission.py`

Leaks: we cannot “overfit one batch” on the hosted runner the way Karpathy means; approximate with smoke_cpu + watching train-batch exact rise under 60s.

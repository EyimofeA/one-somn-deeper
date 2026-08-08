# Canonical transition and T=1 curriculum ablation

Test whether the exact-match Hard failure is primarily caused by a noncanonical
state interface: the old cell sees original `x` on every recurrence and can
learn prompt/depth shortcuts instead of a Markov transition.

Public Easy e5 arms:

1. `prompt_reinject_curriculum`: prior T-hidden card that still re-injects
   original `x` at every recurrence.
2. `canonical_plain`: mutable LSD-first state initialized from `x` once;
   immutable N context; T is loop count only; plain CE/AdamW.
3. `canonical_t1_curriculum`: arm 2 plus T=1-only first 50% and 4x T=1
   late weight.

Arms 2 and 3 differ only in the two registered curriculum constants. Arm 1 is
an architectural baseline, not a parameter-matched control. No arithmetic
intermediate or extra label is used.

Prediction is registered in [`../predictions.md`](../predictions.md). Result is
complete. Raw runner outputs remain Git-ignored.

## Result

| Arm | e5 mean exact | seen-N T=1 | OOD-N T=1 |
|---|---:|---:|---:|
| prompt reinjection + curriculum | 0.7083% | 1/512 | 1/512 |
| canonical + plain | 0.5833% | 1/512 | 1/512 |
| canonical + curriculum | 0.6667% | **5/512** | **2/512** |

The canonical curriculum improves the first-rung counts but fails the
registered 5%/5% and margin gates. It is not a solved transition. It is the
strongest current Hard T=1 lottery candidate because the exact previous Hard
source produced 4/512 seen and 1/512 OOD-N on the same public e5 profile.

Figure: [`../figures/t1_canonical_ablation_2026-08-08.png`](../figures/t1_canonical_ablation_2026-08-08.png).
The plot script consumes the preserved ignored runner logs.

## Hard selection extension

Deriving active slots from `ModelSpec.max_seq_len` produced the frozen compact
source `5b622f06680600f4b346e34b635b839dde18471c`. Its two exact hosted e5 jobs
were 2/512 seen + 3/512 OOD-N T=1 and 5/512 + 4/512. Width 128, batch 256,
and LR 6e-3 controls each zeroed one of the two profiles and were rejected.

Full rationale and source lineage: [`../../submissions/t1_canonical_register/CARD.md`](../../submissions/t1_canonical_register/CARD.md).
Selection plot: [`../figures/t1_hard_selection_2026-08-08.png`](../figures/t1_hard_selection_2026-08-08.png).

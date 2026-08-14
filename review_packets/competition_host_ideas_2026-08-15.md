# Competition-host ideas to investigate

Source: user-provided remarks attributed to the One Layer Deeper competition
organizers on 2026-08-15. These are community/organizer observations, not
internal scores and not evidence that any technique improves our model.

## Reported submission patterns

- Looped Transformers are common, consistent with the rules' pressure toward
  learned recurrence.
- Muon is reportedly the most common optimizer.
- Competitors have explored sparse recurrent memory and sparse access to past
  states.
- Other explored families include learned microprograms, polynomial
  populations, Mamba/GRU architectures, algebraic position-aware positional
  representations, and specialized optimizer combinations.

## Organizer diagnosis

Many submissions reportedly invest heavily in deep recurrence before learning
one exact modular transition. The newly released E6--E10 and M6--M10 datasets
were designed to provide a more granular progression through that failure.

## Old-school training suggestions

The host specifically suggested early stopping and dropout as changes that can
help some strong submissions immediately. We will treat these as hypotheses and
test them with matched controls rather than assuming transfer to our Neural GPU.

## Research implications

1. Keep T=1 exactness as the first competition promotion gate.
2. Record peak validation checkpoints, not only final checkpoints.
3. Test recurrent dropout as a one-variable Neural GPU card.
4. Defer Muon until architecture-level transport controls are complete.
5. Treat sparse memory, learned microprograms, polynomial populations, Mamba,
   and algebraic positions as separate future architecture families rather than
   mixing them into the current Neural GPU ablation ladder.

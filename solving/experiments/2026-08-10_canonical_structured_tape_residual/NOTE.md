# Canonical structured LSD-tape residual

Status: completed; gate refuted with an optimization confound.

This is the smallest competition-safe transplant from the structured latent
tape that beat global/register controls in the small-N diagnostic.  Each
canonical macrostep regenerates a width-128 tape from the current discrete
state, mixes left/self/right neighbors, pairs each slot with same-position
immutable `N`, performs one shared GRU update, and adds the projection to the
canonical mutable features before global attention.

The tape is not decoded and is not a continuous bypass across macrosteps.
Training remains evaluator final-label CE with the canonical T=1 curriculum.
Gate: over 1% mean and at least 10/512 on both T=1 profiles.  Kill if either is
zero or both are below 8/512.  No hosted submission without a fresh decision.

The card completed only 1,362 updates and ended at train loss 1.8876.  Mean
exact was 0.7917%; T=1 was 3/512 seen-N and 2/512 OOD-N.  It fails the gate,
but the retained canonical attention plus tape slowed and underfit enough that
this is not a clean falsification of the direct structured state topology.

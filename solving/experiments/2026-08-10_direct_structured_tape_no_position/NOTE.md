# Translation-equivariant direct structured tape

Status: completed; refuted.

This exact direct-tape control removes learned absolute place embeddings from
both input features and the shared decoder.  Slot ordering and zero boundaries
remain, but every interior position must use the same learned digit operation.
That directly tests the support-audit hypothesis that absolute/identity
channels let 27 repeatedly observed training moduli become a lookup key.

Gate: at least 5/512 on both T=1 profiles and over 0.75% mean exact.  Kill at
OOD-N at or below 2/512 or no mean improvement over the 0.4167% parent.

The card completed 1,864 updates and scored 0.5000% mean exact.  T=1 was
3/512 seen-N and 1/512 OOD-N.  Removing absolute position slightly improves
aggregate exactness over the 0.4167% parent but does not improve unseen-N T=1;
the registered OOD kill fired.

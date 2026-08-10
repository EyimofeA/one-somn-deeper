# Easy e5 multi-lane Neural GPU

Status: complete; local and hosted e5 both failed promotion.

This is the competition-legal transfer of the generic Neural GPU mechanism.
Relative to the translation-equivariant direct structured tape, it replaces
the single state lane/update with six generic 64-wide scratch lanes and one
tied local GRU update repeated four times per requested recurrence step. It
preserves the parser, T=1 curriculum, final-label-only loss, optimizer,
wall-clock schedule, batch sizes, and canonical digit recurrence.

The source contains no carry label, arithmetic trace, modulus operation,
task solver, or hand-authored phase schedule. Scratch lanes are recreated from
the canonical digit state each macrostep. Left/right weights are learned and
asymmetric, so the grid can discover directional propagation without a lane or
direction being assigned arithmetic semantics.

Predict local e5 above the parent's 0.5000% mean exact, with at least 5/512 on
both T=1 profiles and >0.75% mean exact as the promotion gate. Kill at OOD-N
T=1 <=2/512 or mean exact <=0.5000%. The owner requested an Easy attempt; a
hosted e5 run may still be made after source validation even if the local gate
fails, but it will be labeled forced/non-promoted.

## Local e5 result

The exact source validated at 10,693 bytes and completed 1,636 updates in the
60-second local evaluator mirror. Test exact was 12/1,200 (1.0000%), OOD exact
was 2/600 (0.3333%), and mean exact was **0.6667%**. T=1 was **1/512 seen-N**
and **2/512 OOD-N**; neither profile certified a rung.

The card improves the parent's 0.5000% mean but fails the registered OOD-N
kill and both first-rung profiles remain chance-scale. It is not promoted.
The owner explicitly requested an Easy run, so the exact frozen source will be
submitted once to hosted e5 as a forced measurement. Predict hosted mean in
the 0.3%--1.0% range, no certified rung, and no more than 4/512 on either T=1
profile. A hosted fluctuation does not reopen the architecture.

## Hosted e5 result

Exact source SHA-1 `c436691686c76e406445484b64849ac06eac5cac` was submitted
as Easy e5 job `ff081248-f600-40c6-a133-045783f76c68`. It succeeded at
**0.3333% mean exact**: test 0.5000%, OOD 0.2000%, with no certified seen-N
or OOD-N rung. The bounded hosted metrics do not expose first-rung example
counts. Training completed only 532 updates, ended at 8.0% batch exact and
loss 1.565, versus 1,636 local updates and roughly 28% final batch exact.

The hosted prediction was confirmed, and the architecture is closed. Its
generic multi-lane mechanism does not solve T=1 under final-label supervision,
and its nested Python recurrence is especially inefficient in the hosted
wall-clock environment. Do not spend Medium or Hard quota on this source.

The local run's four remote files (source, config, failed first log, completed
rerun log) were backed up from Prime pod
`a6eb7c97e54d4174a9b265674758a383`; the lifecycle helper verified 18,236
bytes on each host. Hosted metrics were downloaded separately into the same
ignored local artifact directory after that verification. The pod remains
active and idle.

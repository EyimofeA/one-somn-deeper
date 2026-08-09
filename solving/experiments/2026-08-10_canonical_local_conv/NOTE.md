# Canonical register plus local ConvGLU

Status: refuted on local and hosted Easy e5.

## One change

Relative to exact canonical Hard SHA-1
`5b622f06680600f4b346e34b635b839dde18471c`, add one translation-shared
depthwise-convolution/GLU residual over adjacent LSD-first mutable digits after
the unchanged global attention step. Parsing, canonical state, T1 curriculum,
optimizer, batch sizes, recurrent loop, output head, and final-label-only loss
are unchanged.

This is the narrow legal translation of the shifted-reducer evidence: local
neighbor propagation is learnable, but no multiplication, comparison,
subtraction, quotient, shift traversal, or arithmetic trace is prescribed.

## Gate

Run public Easy e5 first. Hard promotion requires hosted mean exact above 1%
and at least 10/512 exact on both seen-N and OOD-N T=1 profiles. Either zero
profile or failure of any source/validation check is an immediate kill.

## Parallel review

Terra independently recommended a generic local ConvGLU grid and rejected the
explicit shifted schedule as Rule-7 unsafe. The implemented card is narrower:
it preserves the canonical global transition and adds only a local residual.

## Local Easy e5

The first launch failed before training because of a Conv1d initialization
shape assumption; replacing explicit dimension indices with `math.prod` was a
smoke fix and changed no mechanism. The valid run completed 1,695 updates:

- test 14/1,200, OOD 1/600, mean exact **0.6667%**;
- seen-N T=1 **5/512**;
- OOD-N T=1 **1/512**;
- no certified rung.

This fails every Hard-promotion threshold. Raw runner output:
[`runs/local_e5/train.log`](runs/local_e5/train.log). Hosted Easy job
`d53c55a8-1abd-4b21-af31-dec9071ce42b` tests the exact SHA-1
`64639a3c3c51aa0ee6ab23f5cc286e2dc0c1a05a` on the H100 service.

## Hosted Easy e5

Hosted job `d53c55a8-1abd-4b21-af31-dec9071ce42b` completed 1,513 updates:

- test 3/1,200, OOD 2/600, mean exact **0.2917%**;
- seen-N T=1 **3/512**;
- OOD-N T=1 **1/512**;
- no certified rung.

The card is refuted. Post-run source audit found that `scale=0.1` was passed
through sigmoid, giving an actual starting residual multiplier of 0.525 rather
than the preregistered 0.1. This result remains valid for the strong-residual
card; a separate one-variable scale control tests the intended multiplier.

## Forced Hard selection

The exact source is selected only because the owner explicitly requested a
Hard attempt and it is the only newly tested local-convolution source with
both hosted T=1 profiles nonzero. It did not clear the research promotion
gate. No source change follows hosted Easy SHA-1
`64639a3c3c51aa0ee6ab23f5cc286e2dc0c1a05a`.

Hard job: `f79ebe42-b146-4cce-92e5-1e980c27d55e` (running at upload handoff).

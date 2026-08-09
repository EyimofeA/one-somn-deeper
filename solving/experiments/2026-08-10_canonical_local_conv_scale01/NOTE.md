# Canonical local ConvGLU, exact 0.1 residual

Status: refuted; hosted OOD-N T=1 was zero.

This is the one-variable scale control for the failed strong-residual card.
The local ConvGLU multiplier is a directly learned scalar initialized at 0.1,
instead of `sigmoid(0.1)=0.525`. Every other source line is fixed.

Promotion requires both T=1 profiles nonzero and an improvement in total T=1
hits or mean exact over the strong-residual hosted result (3 seen + 1 OOD-N,
0.2917%). A zero profile kills the source before Hard.

## Local result

- 1,597 updates; test 4/1,200 and OOD 7/600; mean **0.7500%**.
- seen-N T=1 **3/512**; OOD-N T=1 **1/512**; no rung.
- This improves mean exact over the strong-residual local result while keeping
  both profiles nonzero, so it advances to hosted Easy e5.

Exact SHA-1: `c50a5d6cdfd4a4ba7caf69dfffa5c4ddc123fd22`. Hosted job:
`00fdf63b-4428-4e92-89e1-b213222b13ab`. Raw local output:
[`runs/local_e5/train.log`](runs/local_e5/train.log).

## Hosted result

Job `00fdf63b-4428-4e92-89e1-b213222b13ab` completed 1,369 updates with
test 6/1,200, OOD 2/600, and **0.4167%** mean exact. T=1 was **3/512**
seen-N and **0/512** OOD-N. The registered zero-profile kill fired; this
source is not eligible for Hard.

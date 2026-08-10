# Direct MLP x2 mod N

Status: completed; architecture rejected as a reusable arithmetic learner.

This card trains a roughly two-million-parameter plain GELU MLP directly from
fixed-width decimal digits `(x, N)` to the digits of `x2 mod N`. It receives no
intermediate square, quotient, comparison, subtraction, or remainder labels.

The frozen width-three split contains 185 train, 39 validation, and 41 test
semiprime moduli. Each train N contributes 64 train x and up to 64 disjoint
evaluation x; unseen-N test evaluation exhausts every x in `[0, N)`. Seeds
0, 1, and 2 use identical data, AdamW, and 12,000 updates.

Promotion requires at least 90% unseen-N exact in all three seeds. The expected
failure signature is near-perfect training fit with less than 2% unseen-N
exact, which would identify memorization rather than the shared arithmetic
function.

| seed | train exact | seen-N / unseen-x | unseen-N test | unseen-N digit |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 100.00% | 4.53% | 3.87% | 19.55% |
| 1 | 100.00% | 4.45% | 3.92% | 19.65% |
| 2 | 100.00% | 4.46% | 3.79% | 19.82% |

The 2,192,414-parameter MLP memorized all 11,840 training rows in every seed.
Unseen-N cross-entropy nevertheless reached 11.64--11.93, and exactness varied
by only 0.13 percentage points across seeds. The preregistered `<2%` numerical
prediction was refuted. A post-run audit shows the result is not explained by
trivial constants: identity `y=x` reaches 0.77%, constant zero 0.19%, and an
optimistic test-label digit-mode vector 0.23% exact. The MLP learns some
transferable correlations, but roughly 96% of unseen-N outputs remain wrong;
it did not learn modular squaring.

Evidence: ignored artifacts under
`diagnostics/artifacts/prime-a6eb7c97e54d4174a9b265674758a383/runs/2026-08-10_x2modn_direct_mlp/`.

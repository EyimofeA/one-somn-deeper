# Matched binary work-state reduction versus fused T=1

The same 443,777-parameter binary ConvGRU processor was trained for 5.12M
examples in two arms. The only changed input was exact `x squared` bits versus
zero-padded `x` bits. Both used immutable source and N lanes, 44 tied updates,
9% recurrent dropout, Muon warmdown, final residue-bit BCE, and validation-only
checkpoint selection.

| Arm | Train | Unseen x, seen N | Seen x, unseen N | Unseen x, unseen N |
| --- | ---: | ---: | ---: | ---: |
| Exact square input | 4.04% | **5.92%** | 3.56% | **6.80%** |
| Fused x input | 2.41% | **2.04%** | 2.06% | **1.74%** |

Both preregistered gates failed. Exact square input helped, proving that missing
multiplication information contributes, but the privileged arm still failed to
fit training reduction. Both curves peaked by step 1,500 and collapsed during
the same high-Muon phase. The supported conclusion is that this processor and
optimizer are independently inadequate for reduction; the fused result cannot
be interpreted only as square/reducer credit assignment.

Compilation raised the 44-update training smoke from about 3,423 to 5,425
examples/second after its one-time compile. Full arms took 798 and 682 seconds.

Evidence (ignored verified backup):
`diagnostics/artifacts/prime-7072f85e48094888bcf3893db897ea54/binary-workstate-matched-2026-08-16/`.
Figure: [`../../figures/binary_workstate_matched_2026-08-16.png`](../../figures/binary_workstate_matched_2026-08-16.png).

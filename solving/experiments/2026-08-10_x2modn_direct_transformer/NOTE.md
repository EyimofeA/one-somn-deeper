# Direct Transformer x2 mod N

Status: completed; architecture rejected as a reusable arithmetic learner.

This card changes only the direct model family relative to the completed MLP
card. A standard four-layer, eight-head, pre-norm Transformer encoder consumes
LSD-first x/N digit tokens plus learned output-query tokens and predicts the
three final remainder digits. It receives no arithmetic traces.

Data, split seed, three model seeds, 12,000 updates, batch size, AdamW, learning
rate, weight decay, dropout zero, and exact evaluation sets match the MLP card.
Hidden width 192 keeps the parameter count in the same rough two-million range.

Promotion requires at least 90% unseen-N exact in all three seeds. The prediction
is that attention improves shared digit interactions slightly but still fits
examples rather than discovering division, leaving unseen-N exact below 10%.

| seed | train exact | seen-N / unseen-x | unseen-N test | unseen-N digit |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 100.00% | 4.99% | 4.26% | 19.65% |
| 1 | 100.00% | 4.73% | 4.06% | 19.66% |
| 2 | 100.00% | 4.95% | 4.11% | 19.62% |

The 1,786,570-parameter Transformer memorized all 11,840 training rows in every
seed. It improved unseen-N exactness by only 0.17--0.47 percentage points over
the MLP, while unseen-N cross-entropy still rose to 11.03--11.36. Its learning
curves show training exactness reaching 100% near 2,500 updates while held-out
exactness remains flat around 4% and held-out loss worsens.

The preregistered prediction is confirmed. Generic attention makes a small
statistical improvement but does not learn modular squaring from sparse final
labels. Evidence: ignored artifacts under
`diagnostics/artifacts/prime-a6eb7c97e54d4174a9b265674758a383/runs/2026-08-10_x2modn_direct_transformer/`.

# Learned x2 mod N sanity reproduction

Status: completed environment and mechanism sanity check.

The exact existing learned Square -> Comparator/Subtractor diagnostic was run
on the new L40. The split used 18 train moduli and eight disjoint test moduli;
all states of every selected two-digit modulus were evaluated. Synthetic
arithmetic appears in labels only. The forward uses learned digit-serial cells.

| metric | seen N | unseen N |
| --- | ---: | ---: |
| raw square exact | 100.00% | 100.00% |
| full T=1 exact | 100.00% | 95.79% |
| comparator accuracy | 99.98% | 99.93% |
| T=8 rollout exact | 100.00% | 92.52% |

The weakest unseen-N quotient bucket was q=2..3 at 85.71% exact. The complete
report and checkpoint are preserved under
`diagnostics/artifacts/prime-a6eb7c97e54d4174a9b265674758a383/runs/2026-08-10_x2modn_sanity_seed0/`;
all three remote/local SHA-256 hashes matched.

Interpretation: digit-serial learned squaring can generalize perfectly in this
small regime, and a trace-supported reducer nearly composes. This does not
solve the competition regime: the reducer executes one learned subtraction per
quotient unit, so its cost is O(floor(x2/N)) and can approach O(N). The next
mechanism question is efficient learned reduction on disjoint N, not another
ordinary end-to-end optimizer sweep.

Protocol note: this was started as an environment sanity check before its
prediction entry was written. It is therefore a reproduction, not a valid
preregistered decision card. No promotion decision is based on it.

# Public Easy-e5 support audit

Read-only audit of the exact generated public dataset used by the local runner.
No model was trained and no submission reads these statistics.

## Support geometry

- Train: 4,800 rows, balanced across T=1/2/3 (1,600 each), but only **27
  distinct N**.  The same 27 N appear in test, OOD-T, and the seen-N depth
  profiles.
- Train-N range: 527--1,891; six are three decimal digits and 21 are four.
  Each N occurs 97--431 times across all training depths.
- Unseen-N T=1: 512 rows from **71 entirely disjoint N**, all four decimal
  digits, range 2,173--7,747 (12--13 bits).
- The seen-N and unseen-N T=1 profiles have zero `(N,x)` overlap with any
  training row.  Test T=1 has 83/400 `(N,x)` pairs seen at another training T,
  but none seen at training T=1.
- Per train N, T=1 exposes only 28--146 unique x values (median 41), covering
  1.64%--25.05% of its residue domain (median 3.09%).

## Interpretation

The effective one-step learning problem is not “1,600 diverse modular-square
examples.”  It is 27 sparse, repeatedly observed modulus-specific functions,
followed by evaluation on 71 larger unseen functions.  A high-capacity learned
embedding can lower training loss by identifying N and fitting a sparse map;
full-time T=1 training confirms this shortcut by driving train loss near zero
while unseen-N exact falls to zero.

Therefore the decisive architectural property is cross-modulus parameter
sharing at the digit-operation level.  More T=1 weight, worst-digit pressure,
coarse decimal-length balancing, or generic capacity does not create that
sharing.  The next legal test should remove absolute-position/identity channels
from a local state machine before adding depth or optimizer complexity.

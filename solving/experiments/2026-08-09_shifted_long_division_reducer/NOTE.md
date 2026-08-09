# Shifted long-division reducer

Status: first gate complete; registered promotion threshold failed.

## Question

Can the qualified learned decimal comparator/subtractor reuse the same local
operation on `N * 10^p`, making modular reduction linear in digit width rather
than in the quotient?

## Intervention

Initialize from the qualified width-14 `q<=100` unit reducer. Fine-tune both
learned modules on states `k * (N * 10^p) + r`, where `k` is a single quotient
digit in 0..9, `p` is 0..8, and `r < N * 10^p`. Evaluation uses 16 unseen
four-digit semiprimes.

The autonomous diagnostic sweeps shifts high to low and schedules nine learned
compare/subtract opportunities at each shift. Python constructs labels and
scores outputs, but performs no arithmetic in either learned module's forward.

## Promotion gate

- At least 99.9% unseen-N comparator accuracy and subtraction exactness at every shift.
- At least 99% exact autonomous remainder through quotient 999,999.
- Kill if any shift is below 99% or quotient 999,999 is below 95%.

## Legality

Research-only. The fixed decimal-shift traversal encodes a long-division
schedule and requires a Rule-7 audit before it can inform submission code.

## First result

Fine-tuning made subtraction **100% exact at every shift** on 23,040 unseen-N
examples, and comparator accuracy was 99.9609% at the weakest shift. Rollouts
were 100% exact for q=1/10/100/1000, 98.83% for q=99, and 96.19% for q=999,
but only 19.34% for q=9,999 and 37.40% for q=999,999. The long-range gate is
therefore refuted. Extra selected subtractions implicate false comparator fires
on leading zero quotient digits; a separately preregistered boundary repair
freezes the solved subtractor and tests that diagnosis.

## Boundary diagnosis and consolidation

Boundary-only comparator fine-tuning confirmed the diagnosis: all 11 quotient
scales through 99,999,999 became 100% exact, but uniform one-step comparator
accuracy fell as low as 99.4922%. The preregistered mixed-support consolidation
started again from the first shifted checkpoint, froze the subtractor, and
rehearsed both uniform and boundary cases.

The mixed model passed the full gate. Subtraction was 100% exact at every shift
on unseen moduli; comparator accuracy was 99.9609%--100%. Autonomous reduction
was 100% exact at 10 of 11 scales and 1023/1024 (99.9023%) at q=100. At all
fully exact scales, the number of selected subtractions exactly matched the sum
of the quotient's decimal digits. Raw configs, metrics, and reports are in
[`runs/`](runs/); [`autonomous_exactness.png`](autonomous_exactness.png)
compares the three rollout profiles.

This solves the *directly supervised research primitive*, not T=1. The next
question is whether a legal generic recurrent grid can discover the same
shift/reuse behavior from final labels without an explicit long-division
schedule.

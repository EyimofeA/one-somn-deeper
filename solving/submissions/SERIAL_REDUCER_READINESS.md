# Serial reducer submission-readiness audit

Date: 2026-08-04.  Scope: public competition source and manifests only; no
hidden data was opened.

## Candidate 1 — serial unit reducer

**Status: not an end-to-end submission candidate. Do not create a
`submission.py` wrapper.**

The validated diagnostic consists of a learned 136,330-parameter serial
subtractor and a learned 135,169-parameter serial comparator. It processes
least-significant decimal digits first and applies

`F(u, N) = u - g(u, N)N`

through learned categorical digit outputs plus an identity residual. It is
trained and evaluated on an already supplied raw nonnegative integer `u=qN+r`.
It does not receive official prompts `(N, x, T)` nor form `x²`.

An honest official model would first need a learned squaring module that emits
the raw state. Computing that state with Python integer multiplication, or
using the reducer's target construction in `submission.py`, would be a
handwritten arithmetic shortcut and violates the competition rules. The
diagnostic's identity-residual composition is also explicitly documented as a
diagnostic-only implementation, not cleared submission source.

Even if a legal learned squarer existed, the current loop is infeasible. Public
inputs are units `1<=x<N`; one square has
`floor(x²/N) <= N-2`. A W=14 state is wide enough for the largest public Medium
raw square, but the learned reducer executes one 14-position GRU transition per
unit of this quotient. Public ranges include:

| Tier | Public N range | Worst reductions per modular square | Assessment |
| --- | --- | ---: | --- |
| Easy e1/e2 | fixed 323 / 899 | 321 / 897 | Width fits; no learned squarer; O(q) loop remains impractical across batches and T. |
| Easy e3/e5 | 10–11 bits | up to 2,046 | Same interface gap; no runtime evidence. |
| Easy e4 | 11–12 bits | up to 4,094 | Same interface gap; no runtime evidence. |
| Medium m1/m2 | fixed 10,403 / 38,021 | 10,401 / 38,019 | Not feasible as a unit loop. |
| Medium m3/m5 | up to 15 / 16 bits | 32,766 / 65,534 | Not feasible as a unit loop. |
| Medium m4 | 14 / 18 / 22 bits | 4,194,302 | Decisively infeasible. |

The diagnostic has 271,499 trainable arithmetic parameters in its two learned
modules (before any controller), comfortably under the model-state ceiling.
Parameter count is not the blocker; missing learned squaring and O(q) execution
are.

## Candidate 2 — hybrid square model plus serial reducer

**Status: not a legal or validated candidate.**

The current strongest standalone sources are sequence models that predict the
official output directly. No existing one exports an exact learned raw-square
state compatible with the serial reducer. Bridging them by calculating `x²`,
extracting a quotient, or otherwise supplying the reducer its raw state would
put hard-coded arithmetic into `submission.py`. Therefore no hybrid wrapper is
created.

## Current legal sources

`solving/submissions/fable_tcap_adamw/submission.py` is the current existing
standalone candidate. It passes the local source validator (9,071 bytes;
validator output: `valid`) and is independent
of this reducer research. Its completed Hard result is 0.0467% mean exact, so
it has low expected leaderboard value; it is not evidence that the serial
mechanism is submission-ready.

## Decision

Do not submit a reducer wrapper and do not fabricate an Easy or Medium
candidate. The earliest honest integration gate is a learned, legal raw-square
producer followed by a reducer whose *actual transition* reduces state in
sublinear learned steps. The latest per-position controller only reduces the
number of controller decisions; it still used 102.05 learned unit transitions
at q=100.

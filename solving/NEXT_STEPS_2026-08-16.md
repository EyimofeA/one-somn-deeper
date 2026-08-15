# Next steps after the square/reduce identifiability audit

This note separates verified evidence from the independent Pi reviews and the
resulting research program. The exact Hard result is pending under submission
`43d215f5-1d4a-47c7-9cc0-5a547d1a3736`.

## What is actually blocking us

1. **The standalone squarer has sufficient capacity.** The audited 11-bit
   Neural GPU squarer reached 99.55% untouched audit and 100% on the E5 T=1
   values used by the exact-modulo control.
2. **Final modular labels do not identify the square representation.** Random
   final-only training through a perfect differentiable modulo layer reached
   only 3.75% literal-square and 5% modular exact on held-out E5. The hidden
   state retained x but did not organize x-squared.
3. **The reducer independently has a terminal-credit problem.** Patching exact
   square bits into the final-only reducer yielded 0/400 correct examples at
   every checkpoint.
4. **Fixed-N scores overstate operator learning.** The public generator gives
   heavy x overlap across T cohorts. The 16--41% fixed-N Easy scores coexist
   with chance varying-N scores and near-zero unseen-x T=1 profiles.
5. **The prior hosted square phase was not N-blind.** N occupied a scratch lane
   before the square cell. The Hard artifact removes this shortcut and injects
   immutable N only immediately before reduction.
6. **The recurrence curriculum breaks the learned one-step regime.** Hosted M6
   improved until the 50%-time T=1 phase ended; loss jumped at the exact phase
   boundary and accuracy decayed through the remaining recurrent training.
7. **Throughput is below the successful squarer recipe.** Hosted Easy completed
   355 E10 and 825 E5 updates. The supervised squarer needed roughly
   10,000--14,000 updates. Sparse no-wrap supervision also scales as about
   1/sqrt(N), so it cannot be the Hard solution.
8. **Hard needs exactness, not a better average.** Certification requires every
   example in a rung to be correct. A 40% overlap-inflated Easy score does not
   imply a T=1 Hard rung.

## Independent review: Pi Kimi K3

Kimi ranked objective identifiability first, optimization trajectory second,
and architecture/interface third. Its most useful additions were:

- treat the reducer as an equally important unsolved component;
- replicate the strong-weight-decay/grokking hypothesis rather than assuming
  short-budget Muon will discover modular structure;
- test modular/Fourier positional representations as an explicit prior;
- regard one-seed Neural GPU results as promising or unclear, not confirmed;
- expect immutable N conditioning and the repaired update-count schedule to
  help Medium stability, but not to create the missing algorithm by themselves.

Kimi did not recommend GPT-5 Pro yet because the next decisions are empirical.

## Independent review: Pi Opus 5

Opus agreed with the two-sided square/reducer failure and added four concrete
paths:

1. **In-batch chain supervision from evaluator labels.** If rows with the same
   (N,x) and different T occur in one evaluator batch, a provided lower-T label
   can supervise the matching intermediate step of a higher-T row. No trace is
   generated. This needs organizer review because it is legal-looking but
   loophole-adjacent, and matches may be too sparse per batch.
2. **CRT-identified algebraic factorization.** Make S(x) strictly N-blind and
   constrain a learned quotient/multiply/subtract path so multiple moduli pin
   one bounded square tape. This has the strongest identifiability argument,
   but a learned quotient can become a compensating memorization channel.
3. **Bit-serial fused modular cell.** Avoid a free 2W-bit square altogether.
   Feed one x bit per tied recurrent step while N remains present, keep a small
   residue workspace, and learn one generic update. This trades easier
   identifiability for deeper credit assignment and launch-bound inference.
4. **Redundant residue plus final canonicalizer.** Permit noncanonical internal
   residues and enforce canonicalization once at the end instead of after every
   macrostep. This removes repeated discontinuities but has a weaker proof that
   the desired function is selected.

Opus also rejected more no-wrap tuning: the fraction vanishes with N. It
recommended GPT-5 Pro only for a later legality red-team or after a surprising
positive signal.

## Combined research program

### 1. Repair the curriculum transition

Replace the abrupt 50% switch with a loss-masked ladder:

- 0--70% of wall time: train only provided T=1 rows;
- 70--85%: admit T up to 2;
- 85--95%: admit T up to 4;
- 95--100%: admit the full training cap.

Rows above the current cap must be excluded from the loss, not run for fewer
steps against a later target. First gate: hosted M6. Falsify if the loss still
jumps at a phase boundary or final held-out exact does not exceed 0.87%.

### 2. Test CRT identifiability without asking the reducer to learn too

Research diagnostic: N-blind randomly initialized squarer, final modular labels,
and a frozen/supervised reduction path. Unlike square pretraining, the squarer
still receives no square target. Measure held-out literal-square accuracy.
Falsify the CRT route if square exact remains below 50% even when the downstream
path is solved. This is the highest-value mechanistic experiment.

### 3. Build the bit-serial fused residue cell

Use one generic tied cell, a binary residue tape, immutable N, and one scheduled
input bit per microstep. Do not hard-code double/add/subtract actions. Train from
final labels. Gate in order:

- T=1 held-out (N,x) exact >=90%;
- train T<=3, evaluate T=8 >=90%;
- hosted varying-N E5 >2%, then Medium.

This is the best long-term exact-capable architecture if the legality review
accepts the fixed input-bit schedule as ordinary recurrence.

### 4. Reproduce the reducer/grokking claim

On the existing final-label reducer, compare weight decay 1e-5 versus 1.0 after
train interpolation and keep all other variables fixed. A delayed validation
rise counts as grokking only after train exact remains near 100%. Do not use
audit for selection. If the large-decay claim does not replicate, stop citing
it as evidence.

### 5. Ask the organizers one narrow legality question

Ask whether either of these is allowed:

- matching evaluator-provided labels across rows in the same batch to supervise
  exposed recurrent steps;
- scheduling one input bit per generic tied recurrent update without coding the
  arithmetic action.

Do not ask for a broad architecture ruling. If both are rejected, focus on the
fused redundant-residue cell with ordinary final CE.

## Stop conditions

- Stop optimizing fixed-N Easy means unless unseen-x T=1 improves too.
- Stop widening the current squarer; capacity is not the supported bottleneck.
- Stop sharpening/digitizing an ungrounded latent representation.
- Stop no-wrap curriculum work after the current throughput measurement; it
  cannot scale to Hard.
- Do not spend another Hard quota unless varying-N T=1 materially exceeds
  chance or a new mechanism has a direct identifiability argument.

## GPT-5 Pro decision

Do **not** consult GPT-5 Pro yet. Opus and Kimi independently converge on the
same empirical fork: CRT-grounded factorization versus a fused bit-serial
residue cell. Use Pro after one of those produces a nontrivial varying-N result,
or for a focused legality/adversarial review of the exact candidate.

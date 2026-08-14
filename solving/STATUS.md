# Status (living)

**Fixed-width Transformer multiplication factorial (2026-08-14):** under a
commutativity-safe 80/20 split of all `0..99 x 0..99` pairs, four matched
MSD/LSD x padded/natural arms all failed the 90% exact gate. MSD-natural led at
31.66% held-out exact (85.30% train), versus 26.92% for LSD-natural; padding
improved length accuracy to 99.85--100% but reduced numerical exactness.
MSD-natural fell from 85.44% at zero carry columns to 4.48% at three, localizing
the failure to cross-product/carry composition rather than formatting. Card:
[`experiments/2026-08-14_transformer_2x2_factorial/NOTE.md`](experiments/2026-08-14_transformer_2x2_factorial/NOTE.md).

**Reduction-only depth diagnostic (2026-08-14):** supplying the correct
`x squared` digits let an eight-update local recurrent tape reach 100% training
exact in 8.75 seconds, but only 11.34% held-out-x and 18.69% unseen-N exact.
The same checkpoint fell to 17.99%/16.36% unseen-N at 16/32 updates, refuting
insufficient evaluation depth for this model and localizing the failure to
stable final-label reduction-rule generalization. The Prime L40 pod
`0e64a3962e874632adeb435a3b192ef4` remains active at the owner's request; its
completed run is verified locally. Card:
[`experiments/2026-08-14_reduction_only_depth_sweep/NOTE.md`](experiments/2026-08-14_reduction_only_depth_sweep/NOTE.md).
The matched 60-second grokking control also stayed on the same plateau: unseen-N
peaked at 19.16% and finished at 17.76% after training interpolation occurred
by 5.11 seconds. This refutes a rapid delayed-generalization transition.

**Owner learning mode (2026-08-12):** after the already-submitted Hard job is
recorded, pause new GPU rental, model sweeps, and competition submissions until
the owner completes the relevant mastery gates in
[`../learnings/researcher-track/README.md`](../learnings/researcher-track/README.md).
CPU learning exercises and read-only artifact audits remain in scope. The owner
may explicitly override this pause by naming the new evidence or deadline.

**Queued causal-message Hard (2026-08-12):** owner-requested exploratory job
`580f78bc-de32-4495-a1ca-c34726331d3a` uploads new SHA-1
`2b1d03547e064639cc914c9cbe6f529c8aec24a2`. It changes the runnable gated
arithmetic tape's generic local convolution into explicit learned left/right
messages consumed by the next recurrent update. This implements the attached
research verdict's Rank-1 mechanism, but bypasses its diagnostic promotion gate;
the preregistered expectation is no certified rung. Metrics are pending.

Last updated: 2026-08-11. **Rule audit:** the local competition checkout is now
the live upstream `e32c2f9`. It adds bounded evaluator-owned structured
metrics plus multiple evaluator-owned backward passes/same-batch reuse, while
explicitly banning participant-started derivative or nested-training calls.
Hard scoring is unchanged: certified Max T → OOD-N Max T → first-uncertified
rung accuracy → time. The source audit found no scoring or data-distribution
change and no newly prohibited call in the archived candidates.

**Completed scheduled Hard (2026-08-11):** exact validated Fable T-cap/AdamW
SHA-1 `aa75819a878fab6c03c6a23d979f6234560f6e3d` completed as job
`05f53719-7717-4923-88d5-a3cafe373167` at **0.0300%**, with no certified rung
and **0/768** on both T=1 profiles. It ran 45,376 updates and ended at train
loss 2.16981. This confirms the preregistered chance-baseline prediction.
Immediately before it, exact-source canonical Easy e5 job `cfb0fc73` lost its
OOD-N T=1 signal (2/512 seen, 0/512 OOD-N; 0.5000% mean), while exact-source
Fable Medium m5 job `233cbce0` reproduced 0.1556% mean but had 0/768 at T=1 on
both profiles. Evidence and loss plot:
[`experiments/2026-08-11_submission_selection_refresh/NOTE.md`](experiments/2026-08-11_submission_selection_refresh/NOTE.md).

**Retired x2 mod N GPU session (terminated 2026-08-10 14:11 UTC):** Prime pod
`a6eb7c97e54d4174a9b265674758a383` was an identity-verified L40 at $0.86/hour,
running research SHA `29feda0` against competition SHA `e32c2f9`. Its final
backup matched 906 files and 79,094,220 bytes; Prime then reported zero active
compute pods. The first
environment sanity reproduction reached 100.00% unseen-N raw-square exact,
95.79% unseen-N full T=1 exact, and 92.52% T=8 on the complete two-digit
regime. This is learned but O(q) repeated reduction, so it is mechanism evidence,
not competition-scale or submission evidence. Card:
[`experiments/2026-08-10_x2modn_sanity_seed0/NOTE.md`](experiments/2026-08-10_x2modn_sanity_seed0/NOTE.md).
The matched direct 2.19M-parameter MLP control then fit every training row in
all three seeds but reached only 3.79%--3.92% unseen-N exact; its unseen loss
rose above 11.6. This closes plain fixed-width MLP memorization as the mechanism.
Card: [`experiments/2026-08-10_x2modn_direct_mlp/NOTE.md`](experiments/2026-08-10_x2modn_direct_mlp/NOTE.md).
The matched 1.79M-parameter four-layer Transformer also fit train 100% in three
seeds and reached only 4.06%--4.26% unseen-N exact. Attention improves the MLP
by less than half a point without changing the memorization mechanism. Card:
[`experiments/2026-08-10_x2modn_direct_transformer/NOTE.md`](experiments/2026-08-10_x2modn_direct_transformer/NOTE.md).
The generic 89,902-parameter six-lane Neural GPU then failed an easier raw-square
gate: 12.9125% train exact and 4.0000% on 2,000 unseen x after 12,000 updates.
Its 75.6125% held-out digit accuracy shows shared correlations, but no exact
algorithm or grokking transition. Both registered kill conditions fired; do
not add reduction or tune this card. Card and curve:
[`experiments/2026-08-10_multilane_neural_gpu_square/NOTE.md`](experiments/2026-08-10_multilane_neural_gpu_square/NOTE.md).
The established carry-supervision fix does **not** transfer by attaching a
terminal carry head to this grid. At 12k it scored 4.55%; a registered matched
50k pair ended at 3.85% answer-only versus 6.25% carry-supervised, with middle
digits only 29.95%/21.80%. Carry modestly regularizes but does not become a
causal internal algorithm. Stop duration and final-state auxiliary tuning;
future work must expose carry during the recurrent computation. Card and plot:
[`experiments/2026-08-10_multilane_neural_gpu_square_carry_50k/NOTE.md`](experiments/2026-08-10_multilane_neural_gpu_square_carry_50k/NOTE.md).
The final-label-only legal transfer of the six-lane grid also failed. Local e5
scored 0.6667% with T=1 at 1/512 seen-N and 2/512 OOD-N. Owner-requested hosted
e5 job `ff081248-f600-40c6-a133-045783f76c68`, exact SHA-1
`c436691686c76e406445484b64849ac06eac5cac`, scored **0.3333%** (0.5000%
test, 0.2000% OOD) and certified no rung after only 532 updates. The mechanism
fails both program discovery and hosted wall-clock efficiency; do not send it
to Medium or Hard. Card:
[`experiments/2026-08-10_easy_multilane_neural_gpu/NOTE.md`](experiments/2026-08-10_easy_multilane_neural_gpu/NOTE.md).

**Completed forced Hard (2026-08-10):** job
`f79ebe42-b146-4cce-92e5-1e980c27d55e` ran exact SHA-1
`64639a3c3c51aa0ee6ab23f5cc286e2dc0c1a05a`. It adds one generic local
ConvGLU residual to the canonical register; no explicit shifted-reduction
algorithm is included. Hosted Easy e5 was only **0.2917%**, with T=1
**3/512 seen-N and 1/512 OOD-N**, so this is owner-requested and explicitly
non-promoted. The exact-0.1 scale control was also refuted (0.4167%, 3/512 +
0/512) and was not uploaded to Hard. The Hard run scored **0.02333%**
(3/9,999 test, 1/10,002 OOD-T, 3/10,002 OOD-N), certified no rung, and had
**0/768** on both T=1 profiles after 148,084 updates. Local ConvGLU mixing is
closed as a Hard transfer.

**GPU retired (2026-08-10 00:42 UTC):** Prime pod
`0c1aba701be94af3bb8494f88e962a53` was backed up with exact helper verification
(71 files, 4,897,911 bytes), its two remote project/artifact directories were
removed, and provider termination succeeded. Prime now reports zero pods and
SSH is unreachable. The hosted Hard evaluator is separate and remains running.

**Post-submit T=1 sprint (2026-08-10):** seven preregistered local public-e5
controls all failed unseen-N promotion. Scale telemetry proves the learned
local branch is actively amplified (0.10→1.77) rather than ignored. Smooth
worst-digit loss reached 8/512 seen but 1/512 OOD-N; full-time T=1 training
drove train loss to 0.0516 but produced 4/512 + 0/512; modulus-length group DRO
gave 2/512 + 1/512; four local microsteps gave 3/512 + 2/512. Structured tape
residual/direct/no-position ports were all chance-scale, and a 300-second
direct-tape control worsened to 2/512 + 0/512. Stop optimizer, shallow-local,
and simple-tape micro-tuning. The next real gate is a generic multi-lane local
machine: research-only trace capability first, then an unchanged legal shape
trained from final labels only. Design and kill gates:
[`experiments/2026-08-10_multilane_local_grid_design/NOTE.md`](experiments/2026-08-10_multilane_local_grid_design/NOTE.md).

**Public-e5 support audit (2026-08-10):** the 4,800 training rows contain only
27 distinct N, with median T=1 coverage of 41 x per N (3.09% of the residue
domain). The unseen-N T=1 profile uses 71 disjoint larger N, and neither depth
profile overlaps a training `(N,x)` pair. This makes modulus-identity shortcuts
far cheaper than program discovery. Details:
[`experiments/2026-08-10_e5_support_audit/NOTE.md`](experiments/2026-08-10_e5_support_audit/NOTE.md).

**Sublinear learned reduction (2026-08-10):** a width-14 learned decimal
comparator/subtractor was trained on aligned divisors `N*10^p`, then its solved
subtractor was frozen while the comparator rehearsed uniform and
long-division-boundary states. On 16 unseen four-digit semiprimes, subtraction
is 100% exact at all nine shifts, comparator accuracy is 99.9609%--100%, and
autonomous remainder exactness is at least 99.9023% for every tested quotient
from 0 through 99,999,999 (1,024 examples per scale). This closes the O(q)
research-mechanism gap, but it is **not submission-legal evidence**: arithmetic
transitions were directly supervised and the fixed high-to-low shift schedule
requires Rule-7 rejection. The live question is now a legal generic learned
scheduler plus final-label credit assignment. Evidence and plot:
[`experiments/2026-08-09_shifted_long_division_reducer/NOTE.md`](experiments/2026-08-09_shifted_long_division_reducer/NOTE.md).

**Current Hard candidate (2026-08-08 evening):** compact canonical register
SHA-1 `5b622f06680600f4b346e34b635b839dde18471c` is frozen for the owner's
explicit forced Hard attempt. Two exact hosted e5 runs produced 2/512 seen-N +
3/512 OOD-N T=1 and 5/512 + 4/512, with 0.7083% and 0.8333% mean exact. This is
the only tested source today with both hosted T=1 profiles nonzero twice.
However, full-budget m1 was 0/192 seen and 0/512 OOD-N T=1 after 9,815 updates;
the mechanism is not promoted. Selection is a first-rung lottery, not a claim
that T=1 or downward transition identification is solved. See
[`submissions/t1_canonical_register/CARD.md`](submissions/t1_canonical_register/CARD.md).
The exact frozen source completed as Hard h1 job
`7714d650-78a4-4d4a-8fc1-a384914d7658`: **0.0500%** mean exact, no certified
rung, and **0/768** at T=1 on both seen-N and OOD-N profiles. It completed
163,274 updates in 3,600 seconds and finished at train loss 2.17846. The hosted
Easy first-rung hits did not transfer to Hard.

**Current T=1 finding (2026-08-08):** three-seed final-label controls close
two more architecture-only branches. Hiding `N` during a generic square phase
ties the entangled control at 17.29% median unseen-N exact (11.76% versus
13.03% held-out-x). Replacing that phase with learned pair-to-column routing,
a shared fold, and an LSD carry scan is worse at 16.36% unseen-N / 10.50%
held-out-x. Every run fits train 100%. The blocker is upstream credit
assignment/identifiability from modular final labels, not another choice of
state topology, representation, or square-phase depth. Do not transfer either
card to a competition submission.

**Public-support control (2026-08-09):** replacing the generic factored tape's
18 tiny training moduli with all 1,600 public Easy T=1 training rows makes the
negative result substantially stronger. Seed 0 fit train exactly after 18,019
updates but achieved only **7/512 seen-N** and **1/512 OOD-N** T=1 exact.
Both preregistered kill thresholds fired, so no more seeds were run. Training
support is not the missing ingredient; the next card must change legal credit
assignment or enforce a robust square-to-reduce interface. See
[`experiments/2026-08-09_t1_factored_e5_support/NOTE.md`](experiments/2026-08-09_t1_factored_e5_support/NOTE.md).

**Interface-noise control (2026-08-09):** adding only training-time Gaussian
noise at the factored tape's square/reduce boundary retained 1,599/1,600 train
exact but worsened first-rung transfer to **2/512 seen-N** and **0/512 OOD-N**.
The registered kill threshold fired. Do not sweep noise magnitude; simple
smooth robustness does not repair final-label credit assignment. See
[`experiments/2026-08-09_t1_factored_e5_interface_noise/NOTE.md`](experiments/2026-08-09_t1_factored_e5_interface_noise/NOTE.md).

**Public Easy/Medium completion (2026-08-09):** exact Fable T-cap/AdamW SHA-1
`aa75819a878fab6c03c6a23d979f6234560f6e3d` now has hosted results on all ten
public datasets. Today's missing six scored e2 **1.2083%**, e3 **0.5000%**, e4
**0.2708%**, m2 **0.1500%**, m3 **0.2667%**, and m4 **0.0778%** mean exact;
none certified a rung. Fixed N, fixed T=2, and ten-times-longer optimization
each fail to uncover a transferable operator. See the result table and plot in
[`experiments/2026-08-09_easy_medium_completion_suite/NOTE.md`](experiments/2026-08-09_easy_medium_completion_suite/NOTE.md).

**Identity-initialized reducer (2026-08-09):** a learned reduction residual
opened from 0.01 to **0.7592** while fitting all 1,600 public T=1 rows, but
finished at exactly the deterministic anchor's **7/512 seen-N** and **1/512
OOD-N**. This refutes random-reducer credit blockage as the primary cause.
Stop smooth gate/initialization work; the next branch must make sublinear
reduction computationally plausible. See
[`experiments/2026-08-09_t1_factored_e5_identity_reducer/NOTE.md`](experiments/2026-08-09_t1_factored_e5_identity_reducer/NOTE.md).

**Current Hard result (2026-08-08):** T=1-weighted exact-match/SAM job
`9e7404cb-b0c9-480a-aa64-8d90cc853d67` completed at **0.02333%** overall:
3/9,999 test, 2/10,002 OOD-T, and 2/10,002 OOD-N. No rung certified; both
seen-N and OOD-N T=1 profiles were 0/768. This refutes transfer of its nonzero
public e5 T=1 hits. The L40 was backed up with exact file/byte verification and
terminated; Prime now lists zero running pods.

**Submission execution update (2026-08-05):** no new hosted submission has
been made. A current local L40 revalidation rejects both immediately available
Easy sources: the research-transfer recurrent source gets 0.00% test / 1.00%
OOD (0.50% mean), and Fable v2 gets 0.00% / 0.00%. Their train curves and
provenance are preserved under Git-ignored `runs/`; the selection rationale is
[`submissions/SUBMISSION_EXECUTION_REPORT.md`](submissions/SUBMISSION_EXECUTION_REPORT.md).
Do not spend a tier attempt on either result. The distinct Fable
T-cap/AdamW control is the current Easy candidate: 0.67% test / 8.00% OOD
(4.33% mean), with no certified T=1 rung. Hosted Easy e1
`56335b5e-b460-4de2-a7d0-ed91fb9881fe` improved to **8.50% mean** (6% test,
11% OOD) with no certified T. The same audited source completed Medium m1 as
`d71cad94-07ba-469f-8c7c-676e55d611a9` at **0.0333% mean exact** with no
certified rung. The earlier "running" description was stale.

**Hard deadline submission:** `80e46f83-aeff-40cb-90e4-d09a875814ae` is the
legal final-label-only recurrent VDF candidate from
`experiments/2026-08-05_vdf_square_reduce_final_label/submission.py`: tied
learned LSD-first SquareCell → ReduceCell, dynamically executed T times,
AdamW, no precomputed arithmetic or diagnostic traces. It was source-validated
and accepted before the deadline; hosted metrics are pending.

Hard Fable v2 job `602bf7f1-eab7-46c2-91e8-e4a4a010f9d7` completed with
**0.0467%** mean exact (5/9,999 test; 5/10,002 OOD-T; 4/10,002 OOD-N) but
certified no T=1 rung. The live Hard leaderboard read on 2026-08-04 placed
`mof` **#19/19**, with 0.0000% on both displayed T=1 tie-break profiles. This
replaces stale `#11 at 0.03%` language below; historical rows remain history,
not the current rank key. The current local mechanism work is a bounded,
non-submission decimal-reduction diagnostic; its results are below.

## Primary branch — serial multi-N learned subtraction (2026-08-04)

A learned LSD-to-MSD GRU over aligned `(u digit, N digit)` pairs achieves
**100% held-out-N q=1 exact** on 16 unseen four-digit semiprimes × 128
remainders, with no handwritten arithmetic in the forward. With width-six
states and q=1..5 trace support, the seed-0 frozen subtractor rolls out at
100% through q=10; its raw one-step transition softens only at q=8..10.

**Transition-support extension:** changing only serial subtraction support to
q=1..10 gives 100% q-supplied terminal rollout q=1..20 on unseen N (one-step
accuracy first softens at q=15). A fresh 129-parameter learned stop readout of
that frozen GRU takes only `(state, N)`, not q or depth, and makes complete
autonomous execution 100% through q=13. The preceding stop head cannot be
reused across independently trained GRUs because latent coordinates are not
stable; this is a control, not a modular-arithmetic failure.

**Stability gate is refuted:** the frozen q=1..10 subtractor changes every
held-out true remainder under one more learned update (`F(r,N) != r` for all
2,048/2,048 samples). A stability-gated terminal rule therefore cannot halt
even q=0. Adding identity plus frozen-trajectory recovery labels was harmful.

**Clean piecewise control:** equal q=0 identity and q=1..20 subtraction support
in a monolithic GRU has 0% q=0 fixed-point exact even on seen remainders,
despite 100% q=1/5/10 subtraction. It is a fitting/interference result, not an
unseen-N identity claim.

**Comparator-controlled result:** a separate learned serial comparator reaches
99.93% unseen-N comparison and 100% held-out N−1/N/N+1 boundaries. Gating the
learned subtractor against an identity residual gives 100% q=0 fixed points,
transitions through q=28, and complete autonomous unseen-N execution q=0..28.
The frozen transition audit isolates the q≥29 frontier: comparator accuracy
remains 100% q=1..100, but learned one-step exact first degrades at q=30
(93.75%), before rollout can compound it; q=50/q=100 teacher transitions are
85.06%/86.04% and final rollouts 62.50%/37.50%. The next fixed-architecture
control extended balanced q=0..100 transition support, and it passes: all
206,848 unseen-N one-step cases plus autonomous q=0..100 execution are exact,
with correct halting and q=0 fixed points. The q=30 failure was therefore a
support gap, not a finite-horizon limit in this trained range. This supports
comparator/subtractor decomposition plus full trace support as a valid learned
reducer mechanism. A frozen q=101/110/120/130/140 probe is also 100% one-step
and autonomous exact for every bucket (2,048 each): bounded extrapolation past
the observed q≤100 range is established. The fixed six-digit state cannot
represent every held-out-N example at q≥145, so unlimited reduction and the
competition's squaring task remain open.

**Submission-scale correction (public-source audit, 2026-08-04):** the present
reducer takes one learned subtraction per recurrence. For an official one-step
square, `q=floor(x²/N)` can be as large as `N−2`, so public Medium m4's 22-bit
N can require a 14-digit raw state and up to 4,194,302 reductions. The observed
q≤140 result therefore proves a bounded primitive, not feasible competition
execution. Hard h1's N distribution is hidden and must not be inspected. The
next serial branch must address learned multi-unit reduction/chunk selection;
widening six digits alone is necessary for public scales but insufficient.

**Width control:** fresh W=14 leading-zero padding retains 100% unseen-N q=1
subtraction, 100% boundary comparison, and 100% q=1..100 one-step composition,
but q=0 fixed points/autonomous outcomes are 99.9512% (2,047/2,048), from one
canonical false-continue. Therefore width does not catastrophically break the
serial mechanism, but this seed is not exact and widening is not promoted as a
submission component. The next registered change is learned chunk action,
not a width retune.

**First chunk control is refuted:** a fresh W=14 serial GRU with learned action
`k∈{0,1,2,4,8}` plus learned next digits does not retain the unit primitive.
Although action accuracy reaches 100% for q≥20, digit next-state exact is 0%
at q=0/q=1, 51.86% at q=100, and 9.03% at q=1000; terminal q=100 exact is
0.146%. This is local transition failure, so it neither solves sublinear
execution nor justifies public-scale experiments. Do not repeat the fresh
joint formulation.

**Unit-preserving chunk controls:** initialized direct chunk digits are also
refuted (0% q=0/q=1 next-state exact; q=100 terminal 0%). Freezing the complete
unit reducer preserves its 99.95% q=0/q=1 macro transition, but the new
five-way scheduler predicts k=8 for every q because q-balanced q=0..100 traces
contain 93 k=8 labels. It never chooses k=0 and has 100% non-stops. The only
remaining narrow control before judging scheduling is action-class-balanced
controller exposure; do not modify direct chunk digit generation further.

**Balanced-controller control is also refuted:** equal action batches restore
the k=0 stop (q=0 is 100% terminal exact) but not useful chunk inference:
action accuracy is 0% q=2 and 41.85% q=100, with q=100 terminal exact 0%.
The validated unit arithmetic survives; neither unit initialization nor action
class balance produces a learned sublinear controller. The direct-chunk and
frozen-controller formulations are closed. Any next step needs an explicitly
new controller representation, not another training tweak.

**Threshold-bank control is also refuted for promotion:** replacing the five-way
controller with four learned binary bits for a safe greedy `k=min(q,15)` code
raises q=100 terminal exact from 0% to 30.18% (4.22 mean outer actions), but
breaks q=1 terminal exact at 77.25% and achieves only 73.29% exact q=100 code
selection. This does not preserve the unit primitive’s local behavior. The
only next narrow chunk diagnostic is a final-state versus per-position feature
audit; do not add controller complexity before it. Evidence:
`diagnostics/artifacts/somn-l40-2026-08-04/frozen_unit_threshold_bank/seed0/eval_report.json`.

**Representation audit confirms final-state compression is the controller
bottleneck:** replacing only that 128-dimensional controller input with all
fourteen frozen serial position states recovers 99.95% q=1 and 99.51% q=100
held-out-N terminal exact (selected q=100 chunk is 100%). This establishes that
the frozen sequence retains action-relevant quotient information, but a chunk
still schedules repeated unit updates—102.05 mean inner updates at q=100—so it
does not yet alter O(q) arithmetic cost. Evidence:
`diagnostics/artifacts/somn-l40-2026-08-04/frozen_unit_threshold_bank_per_position/seed0/eval_report.json`.

**Clean full recurrent VDF gate is refuted on held-out N:** in a complete W=4
two-digit-semiprime regime, learned Squareθ is 100% exact even on held-out N,
but learned serial reduction after that correct raw square is only 46.96%
(q≥10: 29.92%) and T=8 is 33.88%. The square/reducer interface is clean; the
remaining blocker is unseen-modulus reduction composition, not raw squaring or
T-conditioned recurrence. Evidence:
`diagnostics/artifacts/somn-l40-2026-08-04/recurrent_vdf_square_reduce_smalln/seed0/localization/eval_report.json`.

**VDF-trace support repairs the clean recurrent cell:** changing only
comparator/subtractor training rows to intermediate states actually generated
while reducing seen-modulus squares raises held-out reduction 46.96%→95.56%,
q≥10 29.92%→94.67%, and tied VDF T=8 33.88%→89.02%. This confirms trace-state
distribution mismatch as the previous main failure. The remaining ~4.4% T=1
error still bars submission integration. Evidence:
`diagnostics/artifacts/somn-l40-2026-08-05/recurrent_vdf_reducer_square_trace_support/seed0/eval_report.json`.

## Task B direct reduction — completed parallel-Transformer branch (2026-07-30)

**Evidence correction (2026-08-04, Codex):** the historical `pure reduction
v2` 78.45% row below is a prose-only claim, not a reproduced artifact. Its
only committed candidate source has a 20,000-step schedule while the claim
requires 80,000 steps; no original data split, command, raw metrics, or
checkpoint remains. It is not a porting or submission basis unless a new,
explicitly labeled replication succeeds.

| Setting | Final held-out-u exact | Reading |
|---|---:|---|
| fixed N=1349, baseline (3 seeds) | **33.65±2.95%** | unseen-u reduction remains unsolved despite 98.03% train exact |
| two N={1349,1357}, baseline (3 seeds) | **11.27±0.63%** | adding a second N collapses generalization |
| two N, correct N broadcast | **11.55±0.30%** | no material gain vs baseline |
| two N, shuffled broadcast control | **11.35±0.39%** | confirms no semantic N-routing effect |
| fixed N, quotient auxiliary | **29.38±6.91%** | refuted vs baseline |
| fixed N, u-copy auxiliary control | **22.35±0.40%** | auxiliary head/extra loss is harmful |

Counterfactual two-N evaluation: baseline changes predictions with N on 92.55% of pairs, but responds incorrectly under both N on 92.38%; correct broadcast does not repair this. The parallel standard-Transformer branch is **falsified for Task B**. Do not scale N broadcast or quotient auxiliaries.

**Serial branch:** fixed initialization is refuted, but semantic initialization is confirmed. Parameter-matched K=8 learned workspace reached 29.23±5.84% peak / 23.47±7.08% final held-out-u EM, below baseline (34.75±2.35 / 33.65±2.95) and five-layer non-recurrent control (38.23±4.67 / 35.65±5.56). Replacing only the fixed initializer with one ordered-input attention read raised held-out-u to **39.22±3.33%** (35.45%, 40.45%, 41.75%), whereas the matched row-stable shuffled-context read collapsed to **14.67±4.43%**. The semantic read therefore clears baseline by 5.57 points and deep control by 3.57 points. The q>=10 subprediction is untested because saved reports bin quotients only as relative small/mid/large. Evidence: `diagnostics/analysis_out/task_b_workspace_init_phase1/`.

## Current research state: digit recurrence gates

| Gate | Data and held-out split | Metric | Result | Meaning |
|---|---|---|---|---|
| 0: route tokens | copy X; 1–3 digit train, 4 digits held out | exact sequence | 100% | routing/position is not the block |
| 1a: raw square | decimal X → X² | exact sequence | 7% same-length peak, 0% 4-digit | raw product formation does not generalize |
| 1b: original digit product | held-out 10×10 pairs; four product values unseen | exact table entry | 15% | confounded by unseen decimal outputs |
| 1b′: repaired pair split | held pairs, but every test product seen in train | exact decimal product | Transformer 45% peak / 25% final | pair relation is partly learned, then forgotten through overfit |
| 1b″: bilinear digit cell | same repaired split; fixed ordinal digits + NALU interaction | exact decimal product | 30% peak / 25% final | multiplicative bias does not beat the generic baseline |
| 1b‴: fixed-step schedule | same Transformer and repaired split; step-indexed LR | exact decimal product | 25% peak / 25% final | naive fixed schedule reaches memorization too quickly |
| 1b⁗: slow fixed warmup | same Transformer and split; 400-step warmup | exact decimal product | 40% peak / 25% final | reproduces early signal but does not retain it |
| 1b⁗⁗: pairwise product-and-carry scan | all 1–2 digit operand pairs train; both operands 3 digits held out | exact 6-digit decimal product | 0.25% peak / 0.05% final | fixed schoolbook routing does not identify the local product law |
| 1b⁗⁗⁗: learned pair table | identical length split; only pair MLP → categorical learned table | exact 6-digit decimal product | 1.30% peak / 0.80% final | local categories fit, but compose poorly across length |
| 1b⁗⁗⁗⁗: one-short curriculum | pair table unchanged; train if either operand <100 | exact 6-digit decimal product | 11.25% peak / 11.15% final | nonzero long-column states transfer; unseen high×high remains the gap |
| 1b⁗⁗⁗⁗⁗: full-position baseline | pair table unchanged; 190k random 3-digit pairs, held complete pairs | exact 6-digit decimal product | 58.55% final/peak | learned multiplication composition exists once all position interactions appear |
| 1b⁗⁗⁗⁗⁗⁗: extended horizon | same full-position card; 8k/400 → 31k/1.6k effective steps | exact 6-digit decimal product | 94.85% peak / 94.5% last eval | prior residual was primarily optimization-horizon limited |
| 1b⁗⁗⁗⁗⁗⁗⁗: fan-in tag | same extended card; add learned vector for fixed `[1,2,3,2,1]` column counts | exact 6-digit decimal product | 84.75% peak | explicit count metadata harms the learned column map |
| 1b⁗⁗⁗⁗⁗⁗⁗⁗: intra-column fold | same extended card; sum → shared pair-GRU fold before carry scan | exact 6-digit decimal product | 97.60% peak / 97.4% final | preserves central pair interactions; clears local product gate |
| 2a: soft-digit recurrence | train bases 0..9 at T=1,2; all bases held at T=3 | exact four-LSD digit state | 40.0% (bases 0..3 only) | soft states fit two applications but drift under a third; narrow 100% was not sufficient evidence |
| 2b: STE state control | 2a data/test; soft state → STE one-hot state | exact four-LSD digit state | 40.0% peak through 1.2k steps | discretization does not repair failure; the missing transition is four-digit squaring itself |
| 2c: one-step 4-digit square | 8k shuffled x train / 2k held x; T=1 | exact four-LSD digit state | 85.35% peak at 3.5k | local 4-digit transition generalizes strongly but is not exact enough to compose |
| 2d: arity fold init | 2c split; fold start state keyed by term count | exact four-LSD digit state | 84.55% peak | earlier learning but lower final law; arity is already inferable from fold length |
| 2e: balanced fold tree | 2c split; sequential fold → binary tree | exact four-LSD digit state | 61.7% at 2k (stopped) | serial term order is valuable; short tree paths hurt learning |
| 2f: soft carry prototypes | 2c split; continuous carry → 64 soft prototypes | exact four-LSD digit state | 26.4% at 1.5k (stopped) | finite bottleneck blocks joint product/carry representation |
| 2g: unweighted aux columns | 2c test; train also predicts later square columns | exact four-LSD digit state | 0% by 1k (stopped) | equal-weight extra labels overwhelm the primary objective |
| 2h: weighted aux columns | 2g labels; auxiliary loss weight 0.25 | exact four-LSD digit state | 64.8% at 2k (stopped) | stable but substantially worse; extra columns divert capacity |
| 2i: digit-3 weighted loss | 2c data/model; loss weight `[1,1,1,4]` | exact four-LSD digit state | 74.55% at 2k (stopped) | residual is structural, not insufficient direct gradient |
| 2j: symmetric pair table | 2c split; table constrained `T=Tᵀ` | exact four-LSD digit state | 50.55% at 2k (stopped) | serial fold benefits from orientation-specific features |
| 1c: carry | 8k train / 2k disjoint test; 1–7 LSD-first three-digit totals | exact normalized output | 98.15% peak at 8k | a shared recurrent state can carry useful state quickly |
| 1d: hard prototype state | same carry data; 64 learned prototypes after every transition | exact normalized output | 0.25% peak | argmax projection prevents learning |
| 1e: soft prototype state | same carry data; soft mixture of the same 64 prototypes | exact normalized output | 98.75% at 8k steps | viable, but slower to optimize than continuous state |

**Current bottleneck:** make a reusable digit-product transition identifiable
under held-out composition. Even correct schoolbook column routing plus a
learned carry scan fits short operands without length generalization. Do not
return to modular squaring or Hard architecture changes until this gate has a
mechanism that generalizes.

**Canonical active code:** [`research/carry_scan.py`](research/carry_scan.py)
and [`research/pairwise_product_carry_scan.py`](research/pairwise_product_carry_scan.py).
The exact A6000 logs are `results_local/gate1_quantized_carry_scan/monitor.jsonl`,
`results_local/gate1_soft_prototype_scan/monitor.jsonl`, and
`results_local/gate1_soft_prototype_8k/monitor.jsonl`, plus
`results_local/gate1_continuous_carry_8k/monitor.jsonl` on `twoA6000`.

## Squaring/reduction isolation (2026-07-24, separate from the ladder below)

Composed `learned_reduction_cell` (squaring+reduction, final-remainder-only
supervision) peaked 5.17% then decayed to 1.72% floor — inconclusive on which
stage fails. Isolated each stage separately, real-token format, N=323 fixed:

| Test | Data | Result | Meaning |
|---|---|---|---|
| pure squaring | x uniform 0-9999, held-out x, label=plain x² (no mod) | **18.65% peak / 18.25% final, stable** | full 8-digit squaring ceiling (not 97.8% — that was mod-10⁴ truncated); mechanism generalizes, small gap |
| pure reduction v1 | P uniform 8-digit, held-out P, label=P mod 323, direct supervision | **0.60% peak**, below 1.72% floor | reduction alone, uniform P, does not generalize at all |
| pure reduction v2 | P via reciprocal/log-uniform sampling (arXiv 2506.23679 A.1), wd=1.0, 80k steps | **78.45% peak; final window 69-84%, no decay through 80k steps** | reduction generalizes for real once P's distribution skews small (matching what x² actually produces relative to N) and given grokking-scale wd/budget; confound-checked (95.4% on trivial P<N, 75.5% on genuine P>=N) |

**Reading:** modular reduction is not inherently unlearnable — it was unlearnable
under uniform-P sampling. The composed cell's poor result and pure_reduction v1's
near-zero result likely share this cause. Next step if this thread continues:
retry the composed `learned_reduction_cell` test but check whether P (the
squaring cell's raw output) is already reciprocal-shaped in practice, or feed it
reciprocal-sampled P directly into the composed pipeline's reduction half.
Full detail: `experiments/predictions.md` 2026-07-24 (a)-(d).

## P2 grokking ladder (the active gate — see `claude code fable/FULL_TRANSCRIPT.md`)

| Rung | Setup (all T=1) | Status |
|------|------|--------|
| 1 | fixed single N, unseen x | **N=323: peak 8.6% but decays to ~2% by end of anneal; N=1073: peak 3.5%→1.5%** — real but non-monotone, never consolidates (`2026-07-23_t1only_fixedn_wd01/`) |
| 2 | multi-N seen at train, unseen x | **floor, 0.5-0.75%** (`2026-07-22_t1only_probe_*` — relabeled from "rung 3", see correction) |
| 3 | held-out N (`split_group=modulus`) | not run — strictly harder than rung 2, gate (≥5%) nowhere in sight |

Failure point is **below rung 2**: the one-step map is barely learnable even
per-modulus. No Hard-tier architecture work is informative until rung 1 clears a
real number. wd=1.0 refuted at d=32 (never fits train, `_wd1/`).

## Hard leaderboard

Live read 2026-08-04: `mof` is **#19/19**, with no certified T=1 rung in
either profile. The completed Fable v2 run has scattered mean-exact successes
but 0.0000% at the displayed T=1 tie-break profiles. Historical `#11 at 0.03%`
and `top 0.40%` statements are superseded; see the rule audit above.

Protocol: [`../RESEARCH_PROTOCOL.md`](../RESEARCH_PROTOCOL.md).  
Lecture: [`../learnings/readings/one-layer-deeper-notes.md`](../learnings/readings/one-layer-deeper-notes.md).  
Path D short form: [`../learnings/concepts/18-lipschitz-quantize-progressive.md`](../learnings/concepts/18-lipschitz-quantize-progressive.md).

Full sequential competition runs: [`experiments/SCOREBOARD.md`](experiments/SCOREBOARD.md) (also live canvas).

## Best scored cards (learned line)

| Axis | Card | Score | Note |
|------|------|-------|------|
| Easy e1 | `depth_d32_k2_ut_evalk4` | **6.80%** (n=3, σ≈0) | **Invalid ranking signal** |
| Easy e5 | `depth_d32_k4_ut` | **1.00%** | Prefer over e1 |
| Medium m5 | `depth_d32_k4_ut_optsched` | **~0.20%** | Schedule-safe |
| Hard H1 | `claude_hard_h1` | **0.03%** | Train 100% / eval 0% |

## Active submissions

Symlinks → `experiments/2026-07-21_<name>/`. Full history: all `2026-07-21_*` dirs.

| Path | Role |
|------|------|
| `depth_d32_k4_ut_optsched/` | Medium/Hard schedule-safe UT K4 |
| `depth_d32_k2_ut_evalk4/` | Easy e1 peak (weak gate) |
| `depth_d32_k4_ut/` | Easy e5 peak |
| `claude_pv_k4_ut/` | Place-value UT |
| `claude_hard_h1/` | Hard artifact — do not widen |

## Next (Part 8 — you pick; agent implements)

**Read first:** [`DESIGN_NEXT.md`](DESIGN_NEXT.md) — the current architecture thesis
(form vs. content; learn the operation + reduction instead of hardcoding squaring;
the true gate is one-step held-out-N).

The executor records PREDICT in [`experiments/predictions.md`](experiments/predictions.md) before any run.

1. Measure μ+λ on local data (no GPU) — gates Path G bonus  
2. Count digits of N on h1/m5 — gates Path E  
3. Progressive loss (one-change card)  
4. STE quantize between steps  
5. Input inject each loop  

## Ops

[`experiments/OPS.md`](experiments/OPS.md) · [`experiments/LAYOUT.md`](experiments/LAYOUT.md) · [`RESEARCH_LOG.md`](RESEARCH_LOG.md)

## Three-track execution update (2026-08-05)

**Track A:** the legal Fable T-cap + AdamW source passed validation and was revalidated locally on the active L40. Easy e1: 482 updates / 60.0s, 1.33% test, 5.00% OOD, 3.17% mean, 1,595,904 parameters, no certified T=1. Medium m1: 8,363 updates / 600.1s, 0.067% test, 0.000% OOD, 0.050% mean, 1,596,416 parameters, no certified T=1. Both fail the pre-registered promotion gate (mean >8.50% and nonzero T=1); no Hard submission was made. Artifacts: `runs/fable_tcap_adamw_easy_e1/`, `runs/fable_tcap_adamw_medium_m1/`.

**Track B:** review files are now `MODEL_REVIEW_INVENTORY.md`, `MODEL_REVIEW_DISAGREEMENTS.md`, `MODEL_REVIEW_EXPERIMENTS.md`, and `MODEL_REVIEW_SYNTHESIS.md`. Missing external chat responses are explicitly marked rather than inferred.

**Track C:** the first registered clean latent-workspace diagnostic is complete. In a small-N final-label-only comparison, global latent unseen-N exact was 17.29% at T=1 versus 9.35% for a per-position register control; T=8 was 14.49% versus 14.02%. This is a narrow state-interface signal, not a competition candidate. Artifact: `diagnostics/artifacts/clean_latent_workspace_seed0/eval_report.json`.

**Current causal gate (not yet run):** resolve state topology before another
architecture search. The next research test is a structured, LSD-aligned
per-position latent VDF state against the existing pooled global latent and
prompt/register controls. The decision tree is fixed:

```text
structured per-position latent VDF test
├─ works from final labels → validate on Easy, then Medium, then Hard
├─ works only with diagnostic traces → research the legal-objective bridge;
│  competition stays on the incumbent
└─ fails → run a controlled binary/limb representation test; do not transfer
   it to competition without a positive signal
```

This gate is motivated by a separate verified reducer audit: final-state
compression discarded quotient-relevant information, whereas all serial
per-position states restored q=100 action selection and 99.51% terminal
exactness. That is reduction evidence, not yet VDF evidence. Keep the
state-topology and trace-objective questions distinct.

**Profiling:** Easy telemetry averaged 4.0% GPU utilization (max 5%), indicating a real launch/CPU bottleneck; Medium averaged 40.7% (max 51%), so optimization is secondary to the clear generalization failure. Keep the L40 active; do not submit Hard.

## T=1 pivot — 2026-08-08

The T=1-only tournament is complete. Under one common 18-seen/8-unseen
modulus split, 80/20 held-out-x split, one seed, and 120 seconds per arm on an
L40:

| Arm | Held-out-x exact | Unseen-N exact |
|---|---:|---:|
| Register baseline | 4.62% | 8.64% |
| Global latent | 10.50% | 16.36% |
| Structured LSD tape | **12.18%** | **17.06%** |

The structured arm is only a narrow improvement over global latent and is
larger/slower. It is not promoted to competition. The discrete masked-token
refiner is also closed: unseen-N exact is 0.47%/7.94%/8.18%/8.88% at K=1/2/4/8,
with increasing latency. No Easy, Medium, or Hard screen was run from these
cells. Artifact: `diagnostics/artifacts/t1_tournament_2026-08-08/summary.md`.

**Next gate:** binary/limb representation comparison at T=1. Do not train on
T>1, run another T curriculum, or perform zero-shot recurrence testing until a
T=1 model clears a materially stronger promotion threshold.

## T=1 representation branch — 2026-08-08

The matched decimal/binary/fixed-limb comparison is complete. All arms used the
same T=1 final-label objective, structured LSD-aligned tape, 18/8 modulus split,
80/20 held-out-x split, AdamW, one seed, and 120 seconds on the L40. The limb
width (two little-endian 4-bit limbs) was registered before the run.

| Arm | Full-train exact | Held-out-x exact | Unseen-N exact | Params | Updates |
|---|---:|---:|---:|---:|---:|
| Decimal control | 100.00% | 11.76% | **18.69%** | 261,962 | 24,524 |
| Binary (7 bits) | 7.81% | 2.10% | 4.21% | 260,770 | 21,006 |
| 4-bit limbs (2 limbs) | 100.00% | 11.34% | 14.49% | 262,864 | 27,056 |

**Decision:** representation-only improvement is not supported. Binary and
limbs remain in the same or worse low-generalization regime, while binary also
fails to fit the training distribution under the matched budget. Close this
micro-tuning branch. No T>1 or competition transfer has been launched from it.
Raw reports and curves: `diagnostics/artifacts/t1_representation_2026-08-08/`.

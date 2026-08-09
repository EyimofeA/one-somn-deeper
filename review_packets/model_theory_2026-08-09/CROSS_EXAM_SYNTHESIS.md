# Cross-examination synthesis

## The problem from first principles

The public recurrence is

\[
x_{t+1}=x_t^2\bmod N.
\]

The model receives decimal tokens for N, x, and T and is scored on the final decimal digits only. An entire row counts only if every digit is correct. This creates two distinct problems:

1. **Computation:** learn a length- and modulus-uniform algorithm for one modular square from final digit labels.
2. **Identification:** if supplied training depths omit T=1, decide which one-step composition root generated the observed final labels.

More recurrence capacity cannot fix either problem. The first requires exact unseen-N arithmetic. The second is impossible in a generic neural class without T=1 labels or a restrictive prior. The scalar, inverse-squaring, and permutation constructions prove this.

The current Hard result is still primarily a computation failure: CE 2.17846 is close to random digit CE, so the model never reached the stage where choosing among exact roots was the dominant issue.

## What each model recommended

### Kimi K3

Kimi proposed a small tied attention cell over canonical decimal slots, with an architectural anchor enforcing G(1,N)=1. This exactly removes the scalar family because c*x^2 maps 1 to c. Kimi correctly withdrew any stronger claim: inverse-squaring also fixes 1, and permutation roots can be chosen to fix 1.

Useful contribution:

- clean scalar-gauge theorem;
- honest statement that all-even training leaves an Occam bet;
- diagnosis that the unknown Hard depth distribution is pivotal.

Weaknesses:

- two attention layers with no internal digit-serial workspace are unlikely to discover exact modular reduction on unseen N;
- its predicted Easy/Medium success is unsupported;
- several proposed official experiment durations exceed tier clocks;
- decoding T and forcing G(1)=1 are both task-specific legality risks.

### GPT-5.6 Sol

Sol proposed the least task-coded design: route LSD-first N/x digits into six lanes and apply one shared local ConvGLU cell for 64 microsteps. The cell has generic local communication and lane mixing, but no named product, carry, quotient, comparator, or reduction stage. The first file calls the learned one-step module once and trains only on legally supplied T=1 final labels. Recurrence is deferred.

Useful contribution:

- strongest identifiability theorem, including roots matching every T>=2;
- clean separation between a T=1 learner and an outer recurrence;
- a legal T1-weighted final-label loss using a prompt-derived mask in auxiliary state;
- most credible Rule-7 posture.

Weaknesses:

- the generic grid provides no theorem that SGD will learn modular arithmetic;
- OOD-N exactness may plateau one or two rows below certification;
- 64 continuous microsteps can still form an opaque analog program;
- the T1 mask, digit reversal, and field routing need organizer confirmation.

This is the best first implementation because its failure is scientifically legible and it does not pre-decide the arithmetic algorithm.

### Claude Fable 5

Fable reduced its proposal to a tiny digit-register machine with learned digit product/add/subtract/compare cells and fixed anti-diagonal, carry, and shifted-subtraction routing. It conditionally excludes scalar and inverse roots if its learned reduction is residue-preserving.

Useful contribution:

- explicit, inspectable state machine;
- strong local probes for learned digit tables and residue preservation;
- good emphasis on a hard canonical register and exact T=1 gates.

Weaknesses:

- residue preservation is not guaranteed by the learned cells, so the exclusion theorem is conditional on the property that training must discover;
- arbitrary permutation roots remain an Occam issue;
- anti-diagonal multiplication plus shifted subtraction is recognizably a hard-coded arithmetic algorithm skeleton;
- straight-through gradients through many learned discrete stages are likely to stall;
- local arithmetic probes are diagnostic artifacts only and cannot become official auxiliary labels or loaded state.

This is a valuable fallback only after Rule-7 preclearance.

### Grok 4.5 Slow

Grok also recommended a fixed SquareMul followed by ReduceMod, with learned digit cells and no prompt-to-logit bypass. It openly treated its exclusion of inverse/permutation roots as structural bias rather than a proof.

Useful contribution:

- independent confirmation of inverse-squaring;
- insistence on removing every auxiliary loss and bypass in the first file;
- useful organizer wording for the Rule-7 boundary.

Weaknesses:

- same hard-coded-algorithm risk as Fable;
- the proposed reduction-only and arithmetic consistency ideas from its first round were not supported by supplied labels and were withdrawn in cross-examination;
- some experiment rows were controls rather than informative one-change comparisons;
- fixed square-then-reduce would be brittle if Hard changes the recurrence.

## Primary recommendation

Start with Sol's generic T=1 grid, not Fable/Grok's arithmetic skeleton and not Kimi's anchor.

Minimal first model:

- marker-based GPU routing into LSD-first read-only N/x lanes;
- six lanes of S positions and width 128;
- one radius-1 or radius-2 residual ConvGLU cell tied for K microsteps;
- N and x re-injected at every microstep;
- no absolute positions, only lane and boundary embeddings;
- output logits only from one designated result lane;
- no prompt-to-logit bypass;
- one macro application, no outer recurrence;
- sequence-balanced CE on supplied T=1 rows only, with the T=1 mask returned in auxiliary state;
- AdamW, one evaluator-owned backward pass, no batch reuse, no generated labels, no arithmetic auxiliary loss.

This deliberately does not solve the all-even identification problem. It first answers the more basic question: can a generic learned machine solve x^2 mod N on unseen N when T=1 is directly supervised?

## Six remote-GPU experiments

Use a remote Linux GPU only. Checkpoints are diagnostic and must never be loaded into an official submission. Save source hash, environment, metrics JSONL, loss/exact curves against both seconds and updates, per-position accuracy, seen/OOD T=1 endpoints, throughput, and peak memory.

| # | Single change | Data and cap | Promotion gate | Abandonment signal |
|---|---|---|---|---|
| 1 | Establish K=16 generic grid | Official/recreated E1, 60 s official-equivalent | 100% seen-N T=1; finite stable throughput | <99% train exact after a 600 s diagnostic means the cell cannot even fit fixed-N T=1 |
| 2 | Change only E1 to E5 | E5, 60 s | any nontrivial OOD-N T=1 exactness and >99% seen | seen high with OOD=0 means modulus memorization |
| 3 | Change only K=16 to K=64 | same E5, 60 s | OOD-N T=1 improves materially; target 512/512 both | no gain means communication depth is not the bottleneck |
| 4 | Add only a hard/straight-through terminal result-register projection | same E5, 60 s | T=1 exact does not fall; hard and soft predictions agree 100% | accuracy collapse means discretization is premature or biased |
| 5 | Change only clock: best E5 model to 600 s, three seeds | exact public E5 generator/splits | 512/512 seen and OOD for all three seeds | any miss after three seeds parks the architecture; do not add recurrence |
| 6 | Add only the tied macro recurrence and weighted all-depth final CE | E5, 600 s | preserve 512/512 T=1 and reach >=99% T=2 | T=1 regression or T2 near zero parks recurrence optimization |

If experiment 6 passes, the next run is an even-depth root-selection stress on M3/M5 before Hard. A profile with strong even rungs and near-zero T=1 is the signature of a wrong composition root, not generic undertraining.

## Questions to resolve before submission

Ask organizers:

> Does Rule 7 permit deterministic GPU routing and LSD reversal of the public N/X/T decimal fields, a prompt-derived T=1 loss mask, and later using T only as the application count of a randomly initialized learned transition, provided there is no fixed multiplication, carry, comparison, division, modulus, or target-generation algorithm?

Also ask whether straight-through categorical boundaries count as an unbroken gradient path under Rule 8.

Research questions the panel exposed but cannot answer from public evidence:

- Does Hard training contain T=1 or any odd depth?
- Does OOD-N separately vary modulus identity, decimal length, and factor/congruence distribution?
- Is the hidden Hard transition still a uniform local arithmetic operation representable by a generic digit grid?

## Final judgment

The bold move is not a more elaborate recurrence. It is to spend the next GPU budget proving or killing one narrow proposition:

> A generic, legal, local recurrent digit grid can reach 100% exact T=1 on unseen moduli using only supplied final labels.

If false after the six-run ladder, the project needs a new legal inductive bias or organizer clarification—not more width, more T, or another Hard lottery ticket.

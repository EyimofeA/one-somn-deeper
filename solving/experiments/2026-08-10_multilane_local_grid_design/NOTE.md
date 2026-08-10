# Multi-lane local grid: next T=1 gate

Status: design only; no source and no run.

## Why another single tape is insufficient

The shifted reducer succeeds because it retains several kinds of information
at once: current value, aligned modulus, comparison evidence, borrow/carry-like
state, and whether more work is needed.  The failed direct tape compresses all
of that into one 128-dimensional vector per digit and gives it one local update
per modular step.  Four repeats of one ConvGLU add communication distance but
not explicit scratch capacity.  A generic machine needs multiple writable
lanes before final-label credit has any chance to select a reusable program.

## Legal-shaped architecture

State has shape `[batch, digit_slot, lane, width]` with six lanes and no learned
absolute place embedding.  Digit slots are LSD-first and have only left/right
boundary markers; lanes have learned role embeddings but no assigned arithmetic
meaning.

Each of 32 tied microsteps performs the same operations:

1. inject the same-position immutable `N` digit embedding into every lane;
2. mix radius-one neighboring digit slots and all lanes with learned weights;
3. apply one shared gated update to every slot/lane;
4. leave all control, routing, and scratch semantics learned.

After the microsteps, one designated public lane produces digit logits.  Those
logits are the only output and are straight-through quantized into the only
state carried to the next requested T.  Scratch lanes are regenerated from
that canonical digit state at every macrostep, preventing an opaque continuous
prompt-to-output bypass.

The forward contains no multiply, carry, compare, subtract, shift traversal,
or hand-authored phase schedule.  `K=32` is generic tied network depth, not a
sequence of different arithmetic actions.  This is materially safer under
Rule 7 than the successful shifted reducer, while satisfying ordinary
autograd/optimizer restrictions in Rules 8 and 14.

## Two-stage falsification

### A. Research-only capability

On recreated public-scale T=1 data, train the exact legal-shaped forward with
generated intermediate arithmetic traces.  Trace targets exist only in this
diagnostic loss and never enter a competition source.

- Confirm capability at >=99% exact on unseen N.
- Kill the architecture below 90%; do not try final-label training if the
  machine cannot represent and learn the program even with credit supplied.

### B. Competition-legal discovery

Delete the trace loss without changing one byte of model forward, initialization,
or optimizer.  Train from evaluator final labels only, T=1-only for the first
half and ordinary mixed depth afterward.

- Promotion gate: >=64/512 on both public T=1 profiles, then repeat locally.
- Hosted gate: >=64/512 on both profiles and >5% mean exact.
- Hard gate: do not upload until hosted evidence is replicated; no chance-scale
  first-rung lottery from this branch.

## If capability works but legal discovery fails

Hold architecture fixed and run a research-only coverage map that separately
increases states per modulus (41→128→512) and number of moduli (27→54→108).
This distinguishes missing within-N coverage from missing cross-N program
discovery.  Do not respond with width, optimizer, loss-reweighting, or K sweeps;
today's controls already close those generic optimization explanations.

# Engineering decision

## 1. One submission design

### **ASSUMPTION — legality pending one organizer clarification**

Deterministically routing public field/digit tokens into registers is input formatting, not a hard-coded arithmetic algorithm. No squaring, multiplication, comparison, division, remainder, lookup table, or arithmetic auxiliary target appears anywhere.

### Learned LSD-first recurrent digit grid

Let \(S=\texttt{spec.max\_seq\_len}\), \(V=17\), width \(d=128\).

| Module | State/operation |
|---|---|
| GPU field router | Extract `N`, `X`, `T` using marker tokens; right-align and reverse digits to least-significant-first. No numeric decoding. |
| Input embedding | Randomly initialized, trainable \(E\in\mathbb R^{17\times128}\). |
| Grid | \(H\in\mathbb R^{B\times6\times S\times128}\). Lane 0: \(N\); lane 1: \(x\); lanes 2–5: zero-initialized scratch activations. |
| Shared cell | One randomly initialized residual ConvGLU cell, tied for \(K=64\) microsteps. Radius-1 convolution along digit positions plus pointwise mixing across all six lanes. \(N,x\) are re-injected each iteration. No absolute position embeddings; only register and boundary embeddings. |
| Readout | Lane 2, projected by a learned \(128\to17\) head, reversed and tail-aligned into \(B\times S\times17\) logits. |
| Auxiliary | Boolean `is_t1` derived from the prompt’s `T` field. Nothing arithmetic. |
| State size | Approximately 1–2M parameters; all trainable parameters are randomly initialized and included once in AdamW. |

The one-step module is

\[
S_\theta:\bigl(D(N),D(x)\bigr)\longmapsto Z_y\in\mathbb R^{B\times S\times17}.
\]

It never sees \(T\). The first submitted model calls it exactly once.

### Recurrence extension—not in the first file

After \(T=1\) promotion:

\[
p_t=\operatorname{softmax}(Z_t/\tau),\qquad
x_{t+1}=p_tE,\qquad
Z_{t+1}=S_\theta(N,x_{t+1}).
\]

Scratch lanes are reset at every macrostep; only digit probabilities cross the boundary. Evaluation uses digit argmax. This keeps a direct final-label gradient but leaves a soft/hard mismatch to test empirically.

### Why this design

- Local tied updates can learn place-independent carry/control rules.
- Sixty microsteps let information cross every public decimal field many times.
- It does not prescribe multiplication or reduction stages.
- There are no optional attention stacks, quotient registers, entropy losses, dual rollouts, custom optimizers, or auxiliary heads.

---

## 2. Composition ambiguities

Let \(S_{\rm train}\) be the observed depths.

### **THEOREM A — multiplicative twist**

For \(c\in U_N\), define

\[
G_c(x)=cx^2\pmod N.
\]

Then

\[
G_c^T(x)=c^{1+2+\cdots+2^{T-1}}x^{2^T}
        =c^{2^T-1}f^T(x).
\]

Let

\[
D=\gcd_{T\in S_{\rm train}}(2^T-1).
\]

Any \(c\) satisfying \(c^D=1\) gives \(G_c^T=f^T\) at every training depth. If \(T=1\) is present, \(D=1\), forcing \(c=1\). Otherwise nontrivial roots can exist.

### **THEOREM B — inverse squaring**

\[
G_-(x)=x^{-2}
\quad\Longrightarrow\quad
G_-^T(x)=x^{(-2)^T}.
\]

Thus \(G_-^T=f^T\) for every even \(T\). This covers every public Medium training depth.

### **THEOREM C — permutation roots**

For suitable permutations \(\pi\) that fix \(f^2(U_N)\) and permute first-step outputs inside equal-\(f\) fibers,

\[
G_\pi=\pi\circ f
\]

can satisfy

\[
G_\pi^T=f^T\qquad\forall T\ge2
\]

while \(G_\pi\ne f\) at \(T=1\).

### Consequence for the proposed network

The recurrent digit grid does **not** provably exclude any family:

- it can represent \(N\)-dependent multiplicative twists;
- sufficient recurrent compute can represent inversion;
- on finite small domains it can represent exceptional permutations.

Its locality and weight sharing are only an Occam bias toward simpler forward arithmetic.

\[
\boxed{\text{If training lacks }T=1,\text{ recovery of the intended step is non-identifiable.}}
\]

Optimization might select squaring because it is cheaper than inversion or exceptional permutations, but that is **SPECULATION**, not a theorem. Equality on finite evaluation points is not—and must not be treated as—a polynomial identity; the network is not even parameterized as a polynomial in \(x\).

---

## 3. Six-experiment remote-GPU ladder

Use one remote Linux H100 for every controlled run, BF16, public evaluator code, seed 74, batch 512, 600-second training cap and 300-second evaluation cap. “Recreated” means the unmodified public generator and split configuration; no extra examples or labels.

### Common artifacts

Save:

- exact `submission.py`, SHA-256, configuration and environment;
- evaluator `metrics.jsonl` and `RESULT_JSON`;
- curves against both elapsed seconds and updates: loss, exact train accuracy, throughput;
- endpoint bars for seen-\(N\) and OOD-\(N\) \(T=1\);
- per-decimal-position aggregate accuracy;
- completed-update count and peak GPU memory.

Diagnostic checkpoints must never be loaded into a submission.

| # | One change | Data | Required outcome | **EMPIRICAL PREDICTION** |
|---|---|---|---|---|
| 1 | Initial \(K=16\) one-step grid | Recreated Easy E1; T1-only loss | Train exact \(\ge99\%\) by 300 s and seen-\(N\) T1 \(=100\%\). Abandon the cell if train exact \(<99\%\) at 600 s. | Seen \(100\%\); OOD-\(N\) near zero because \(N\) is fixed. |
| 2 | Dataset only: E1 → E5 | Recreated Easy E5 | Promote only at \(512/512\) seen and \(512/512\) OOD T1. Abandon optimization settings if seen \(<99.8\%\). | Seen \(99\)–\(100\%\); OOD \(70\)–\(98\%\). |
| 3 | Microsteps only: \(K=16\to64\) | Same E5 | Promote only at \(512/512\) on both profiles. Abandon this architecture if OOD \(<99.5\%\) after 600 s. | Seen \(100\%\); OOD improves, but likely retains at least one error. |
| 4 | Add outer tied macro-recurrence; keep T1-only loss | Same E5 | T1 must remain \(512/512\) on both profiles. More than one lost T1 row relative to Exp. 3 rejects the implementation. | T1 falls because non-T1 rows consume extra forward compute; T2 remains near zero. |
| 5 | Loss only: T1-only → weighted all-depth final-label loss | Same E5 | T1 still \(512/512\); T2 exact \(\ge99\%\). If T1 falls below \(99.8\%\), abandon joint training. | T2 improves substantially, but T1 and T2 probably both miss certification. |
| 6 | Dataset only: E5 → public M5 | Recreated/public Medium M5, same 600 s | Hard promotion requires \(768/768\) seen and \(768/768\) OOD T1. Any miss blocks Hard; \(\le1\%\) confirms root non-identifiability in practice. | T1 approximately zero: no T1 labels, larger numbers, and all three ambiguity families remain available. |

If Experiment 3 reaches exact OOD T1 only after 60 seconds, the design is unsuitable for Easy but may still justify a Medium-scale diagnostic. Hard should not be attempted unless Experiment 6 unexpectedly passes.

---

## 4. Legal training schedule and loss

### First-file schedule

- Batch size: 512; evaluation batch size: largest that fits.
- Fixed 64 microsteps.
- AdamW:
  \[
  \eta_{\max}=2\times10^{-3},\quad
  \beta=(0.9,0.95),\quad
  \text{weight decay}=0.01.
  \]
- Linear warm-up over 100 optimizer updates; constant learning rate thereafter.
- Evaluator-owned gradient clipping at 1.
- One forward, loss, backward and optimizer step per evaluator pass.
- No multipass, batch reuse, custom backward, or optimizer callbacks.

For valid target digits define

\[
\ell_i=\sum_{j\in\operatorname{valid}(i)}
\operatorname{CE}(z_{ij},y_{ij}).
\]

The first file uses

\[
\mathcal L_{T=1}
=
\frac{\sum_i\mathbf1[T_i=1]\ell_i}
     {\max(1,\sum_i\mathbf1[T_i=1])}.
\]

If a batch contains no \(T=1\) row, return the differentiable zero
\(0\cdot\sum z\). E5’s large balanced batches should make this rare.

For Experiment 5 and later:

\[
\mathcal L
=
\frac{4\sum_{T_i=1}\ell_i+\sum_{T_i>1}\ell_i}
     {4n_1+n_{>1}}.
\]

Everything comes from prompt-derived \(T\), final logits, final labels and the valid mask. There are no intermediate targets, arithmetic checks, generated labels, or hidden training work.

---

## 5. Strongest legality risk

The risk is that fixed field routing, LSD reversal, or T-based loss selection is judged part of a hard-coded task algorithm rather than representation/control.

Send organizers this exact question:

> **Does Rule 7 permit a model to deterministically route and reverse the public decimal digit fields into fixed GPU registers, and to use the prompt’s T field only for loss weighting or for choosing how many times to apply a randomly initialized learned transition, provided the submission contains no fixed arithmetic operation, multiplication/carry table, modular reduction, target generation, or arithmetic consistency loss?**

A “no” requires learned marker attention and uniform loss over all rows; it materially weakens the proposal.

---

## 6. First submitted file

### Include

- API/config boilerplate and `assert_model_state`;
- GPU marker router and LSD-first register construction;
- one 64-step shared ConvGLU digit-grid;
- one learned vocabulary head;
- `is_t1` auxiliary mask;
- sequence-balanced T1 final-label loss;
- AdamW plus 100-step warm-up;
- batch/evaluation sizes and `SUBMISSION`.

### Omit

- outer recurrence;
- soft/hard digit bridge;
- T>1 loss;
- arithmetic or consistency losses;
- quotient/product/carry-specific registers;
- attention, adaptive halting, noise, entropy penalties;
- custom backward, multipass, batch reuse;
- generated data, pretrained state, diagnostics or saved weights.

### **EMPIRICAL PREDICTION**

The first Easy/E5 submission will improve T1 token loss and probably solve most seen-\(N\) rows, but fail exact OOD-\(N\) certification through one or a few carry/reduction errors. Every \(T>1\) profile should fail intentionally. Its purpose is to answer one binary question: did the model learn an unseen-\(N\) one-step algorithm?

---

## 7. Missing roundtable question

**Does “OOD \(N\)” separately test unseen modulus identity, unseen decimal length, and changed prime-factor/congruence distribution, or does it conflate all three?**

Without matched identity-only and length-only cohorts, an OOD failure cannot tell us whether the missing mechanism is arithmetic, length extrapolation, or factor-distribution robustness.

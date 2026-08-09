I'll pull the live problem page and the official repo first.## 1. What the public sources actually establish

**Proven (problem page, README, and repo source `data/squaring_mod.py`, manifests, `scripts/generate_datasets.sh`):**

- Task: \(x_0 = x \bmod N\), \(x_t = x_{t-1}^2 \bmod N\), output \(y = x_T = x^{2^T} \bmod N\), with \(N = pq\) a semiprime whose factors are never shown to the submission.
- Labels are made by trapdoor: `exponent = pow(2, T, (p-1)*(q-1)); y = pow(x, e, N)`, and **x is always sampled as a unit** (`gcd(x,N)==1`).
- Tokens: `PAD BOS N X T ANS EOS` + 10 digits, vocab 17. Easy/Medium use separate input/output tensors and a padding mask only — bidirectional attention over the full prompt.
- Scoring: ladder \(T \in \{1,2,4,8,16,32,64\}\); a rung certifies only at **100% exact-example accuracy**, and certification must be a consecutive prefix. Hard ranks by Max T on training-seen modulus identities, then Max T on unseen modulus identities (OOD N), then accuracy at each profile's first uncertified rung.
- Budget: Hard = 3,600 s training on one H100, eval budget = half that, 1 accepted attempt/day. ≤500M trainable+buffer state, single `submission.py` ≤ 256 KiB, no hard-coded weights or algorithm, unbroken autograd path, no `.backward()`/autograd entry points, evaluator owns the outer loop. Explicitly allowed: recurrence, tied weights, adaptive computation/halting, depth curricula, custom loss, 1–8 evaluator-owned backward passes per step, ≤8 batch reuses.
- Critical distribution fact: public tiers train on a **sparse subset of T** (e.g., M1 trains \(T\in\{4,8,16\}\)) while the ladder still demands rungs \(T=1,2\) — the certification tests *downward* as well as upward depth generalization. E1/E2 depth eval is **exhaustive in x** over the units of a fixed N.
- Hard's own warning: "may change aspects of the recurrence itself; do not assume it is repeated squaring."

**Inferred (not proven):** Hard reuses the same tokenization and ladder machinery with private profiles; its training set contains multiple N identities and multiple T values; OOD-N uses nearby bit sizes as in Easy/Medium. Whether Hard's prompts remain `(N, x, T)` triples is unknown.

## 2. The one theorem that matters, and why supervision fights it

**Loop-invariance lemma.** Let \(\iota: (\mathbb Z/N\mathbb Z)^\times \to \mathbb R^d\) be the input encoding, \(G_\theta\) a weight-tied cell, and \(r\) a readout into canonical decimal digits. If
\[
r\big(G_\theta(\iota(u))\big) \;=\; u^2 \bmod N \quad \text{for every unit } u,
\]
and the loop-carried state is re-canonicalized to \(\iota(u^2 \bmod N)\) each step, then by induction the readout after \(T\) steps equals \(x^{2^T} \bmod N\) for **all** \(T\), with zero error accumulation. Certification at \(T=1\) over an exhaustive unit set (which E1/E2 literally perform) is therefore a *certificate for every rung*, and the same argument applies per-N for unseen N. The entire game reduces to: **learn the one-step map exactly, as a function of (N, u), and make the recurrence state drift-proof.**

**Why final-answer-only supervision makes this hard:**

1. **Certification is a product, training is a mean.** \(P(\text{rung } t) = \prod_{\text{examples}} P(\text{all digits right})\). Cross-entropy optimizes average log-loss; a model at 99.9% per-example still fails every rung. You must convert mean convergence into *uniform* convergence — easy by memorization for fixed N, genuinely hard as a function of N.
2. **Compositional non-identifiability.** Infinitely many \((G, r)\) pairs realize the same T-step map on the training pairs. A cell could learn "square-4-times" as an atomic map for M1 and score 0 on the \(T{=}1\) rung. The only levers that collapse the hypothesis class toward the true one-step factorization are (a) weight tying across all depths, (b) supervising readouts at *every* training T from the same unrolled trajectory, and (c) bottlenecking the loop-carried state to the canonical residue so intermediate states can't smuggle nonstandard encodings.
3. **Pseudorandom local structure.** Squaring mod N decorrelates gradients; generalization arrives late, grokking-style, after memorization. With 3,600 s and one attempt/day, there is no budget for a grokking miracle at large bit sizes — so the architecture must supply the arithmetic inductive bias (digit-place structure, bidirectional prompt attention, canonical feedback) rather than hope SGD finds it.
4. **On scratchpads:** a categorical scratchpad is *not* obviously optimal. It helps only if its content is trained, and no intermediate labels exist. The minimal sufficient loop state is the residue itself; anything wider is unsupervised state that can drift. I keep intra-step workspace free but loop-carried state canonical.

## 3. Three genuinely different families

**A. Tied cell + canonical discrete register (chosen).** State: \(L\) digit slots, one-hot per slot, fed back via straight-through argmax; \(N\)-conditioned cell \(G_\theta\); learned halt distribution over 64 steps. Signal: CE at the halt step, plus consistency CE at every training T along the same rollout (all from the given final answers — no generated supervision). Exactness: the Lemma — discretization re-canonicalizes each step, so exactness is a per-step classification property, inductive in T. Cost: \(T\) serial cell applications; at \(T{=}64\), ~64 × 8-layer block ≈ trivial vs the 1,800 s eval budget. Failure mode: one-step accuracy < 100% on unseen N collapses certification at rung 1; quantizer train/eval gap.

**B. Looped transformer over continuous latents (TRM-style).** State: full-width continuous latent sequence, no quantization. Same recurrence and signal. Exactness: none inherent — relies on the net implementing exact arithmetic in float; error grows ~linearly in T. Cost: same serial depth, more FLOPs/step. Failure mode: silent drift at \(T\ge 16\); you certify rungs until the accumulated rounding crosses a digit boundary, then cliff.

**C. Learned trapdoor surrogate.** Note \(2^T \bmod \varphi(N)\) is eventually periodic in T, so per-N the answer is a finite object: learn \(T \mapsto e\) (period inference) and \((x, e) \mapsto x^e \bmod N\) (power map), giving O(1) or O(log T) depth. Exactness: two hard classifiers that must *both* be exact. Cost: cheapest at eval. Failure mode: the period is a hidden function of \(p, q\); on unseen N it must be inferred from N's digits — this is factoring-flavored and should fail OOD-N rung 1. Strong for ID Max T, structurally disqualified by the second ranking criterion.

**Choice: A**, because it's the only family whose exactness argument is inductive in T *and* agnostic to N, and because "any recurrence" is still a recurrence — A transfers to Generalised Hard, C does not, B doesn't certify.

## 4. Design

\[
s_0 = \iota(x),\qquad s_{t+1} = \mathrm{ST\text{-}argmax}\big(\pi\big(G_\theta(s_t,\, c(N),\, \tau_t)\big)\big),\qquad \hat y = r(s_{t^\*}),
\]
with \(t^\* \sim \mathrm{Halt}_\psi(T\text{-tokens})\) (soft at train, hard at eval), \(\tau_t\) a learned continuous step embedding, and loss \(\mathcal L = \sum_{T' \in \text{batch}} \mathrm{CE}(r(s_{T'}), y_{T'})\) — one rollout supervised at all training depths present in the batch.

```python
class Submission(nn.Module):            # boundaries: Encoder | Cell | Quantizer | Halt | Readout
    def forward(self, input_ids, attention_mask=None):
        tok = self.embed(input_ids)                       # digit/marker embeddings + place encoding
        n_cond = self.n_pooler(tok, attention_mask)       # bidirectional read of N digits (mask-only)
        halt_logits = self.halter(tok, attention_mask)    # [B, 64], learned depth controller
        s = self.reg_encode(tok)                          # canonical digit slots of x  -> R^{L x V}
        rollout = []
        for t in range(64):                               # ponytail: fixed unroll, soft halt selects
            h = self.cell(s, n_cond, self.step_emb(t))    # tied transformer block, intra-step workspace free
            logits = self.to_digits(h)                    # [B, L, V]
            s = self.quantize(logits)                     # straight-through one-hot: canonical re-encode
            rollout.append(self.readout(s))               # answer-digit logits per step
        out = torch.stack(rollout, 1)                     # [B, 64, answer_len, V]
        w = halt_logits.softmax(-1) if self.training else one_hot(argmax)
        return (out * w[..., None, None]).sum(1), None    # soft/hard depth selection, fully differentiable
```

Loss (`token_training_loss`): CE at the halted step; when a batch contains several T values, add CE at each of those step indices along their rollouts — pure relabeling of the existing forward, no generated supervision. Optimizer: AdamW + cosine, modest weight decay (grokking regime needs it), batch 512.

**Cost:** cell ≈ 8 layers × d=512 ≈ 50–100M params — far under ceiling; 64 serial steps at eval is negligible. **Principal failure mode:** one-step exactness on unseen N within 3,600 s. **Mitigation within the rules:** `backward_passes_per_step` and batch reuse to spend the wall clock on hard residual examples; a max-emphasizing (focal-style) CE to attack the product-of-probabilities problem rather than the mean.

## 5. Public squaring vs Generalised Hard

- **Best public squaring solution:** Family A with the step embedding \(\tau_t\) *frozen to a constant* — pure tied one-step map. Extrapolation in T is then exactly the Lemma; exhaustive-x tiers make \(T{=}1\) certification a genuine proof for all rungs on that N.
- **Best Generalised Hard solution:** the same skeleton but keep \(\tau_t\) learned and the halt controller learned. If Hard changes the recurrence — \(f(x, N, t)\) instead of \(x^2\) — the t-conditioned cell still fits the one-step template and the multi-T consistency loss still forces the reusable factorization; everything re-learns inside the 3,600 s budget from whatever data Hard emits. The bet: whatever Hard is, it's still *serial, exact, and extrapolative in a depth index* — the architecture commits to that meta-structure and nothing else.

Skipped: any C-style shortcut exploitation and auxiliary self-supervision on intermediate states — the former fails OOD-N, the latter needs labels that don't exist. Add C only if the leaderboard shows ID Max T dominating ties.

---

I'll re-verify the load-bearing facts before repairing the design, since the identifiability argument hinges on exactly which T values appear in training.Now the data pipeline source and manifests — the exact training T sets decide the identifiability argument.One check that changes the identifiability story — whether the same x reappears at multiple training T for fixed-N datasets.All key facts now verified from source. Here is the corrected proposal.

## 0. Facts re-verified (proven) vs inferred

Proven from `scripts/generate_datasets.sh` and `data/squaring_mod.py` this session:

- Labels: `exponent = pow(2, T, (p-1)*(q-1)); y = pow(x, exponent, N)`; `x` is always a **unit** (`_sample_unit`, `math.gcd(x, N) == 1`). Prompts are unique per \((N, x, T)\) triple.
- Training depth sets: E1 `{1,2,3}`, E2 `{1,2,4}`, E5 `{1,2,3}`, **M1/M2 `{4,8,16}`**, M3 `{2}`, M4 `{8}`, M5 `{2,4,8}`. So my earlier "sparse T" claim holds for Medium and must be assumed for Hard.
- For fixed-N datasets the **same x list is reused across all training T** (`x_by_split` is built once, then iterated per `time_steps`). Cross-depth labels on identical trajectories exist in the dataset — but per your constraint (no pairing) and the evaluator's ownership of batching, they are unusable. Note for the theorem below: they wouldn't fix the obstruction anyway.
- Depth eval is exhaustive in x only for E1/E2; M1/M2 use 192–768 sampled fresh units per rung. OOD-N uses "nearby" bit sizes; problem page adds "interpolation **and extrapolation** modulus sizes."

## 1. Theorem 1 (loop invariance) — hypotheses now explicit

If (H1) the cell \(G_\theta(\cdot, n)\) is exact as a *function* \(u \mapsto u^2 \bmod N\) on all units of every modulus in scope, (H2) the loop-carried state is re-canonicalized to the standard decimal encoding of the residue each step, and (H3) the cell is time-homogeneous, then \(T\)-fold unrolling is exact for all \(T\), per \(N\), by induction. This was always conditional; my error was treating H1 as achievable-by-default and H3 as compatible with a learned absolute step embedding. It is not: a learned \(\tau_t\) lets the cell specialize per \(t\), breaking both the induction and extrapolation to unseen \(t\). **Removed.**

## 2. Theorem 2 (gauge obstruction) — the identifiability repair, and it's unconditional

Let the training depths be \(D = \{T_i\}\) and define \(g = \gcd_i\!\left(2^{T_i} - 1\right)\). For any \(c \in \mu_g(N) = \{c \in (\mathbb Z/N\mathbb Z)^\times : c^g = 1\}\), set

\[
G_c(u) = c \cdot u^2 \bmod N \quad\Longrightarrow\quad G_c^{\circ T}(u) = c^{\,2^T - 1}\, u^{2^T} \bmod N .
\]

Since \(g \mid 2^{T_i} - 1\) for every training depth, \(G_c^{\circ T_i} = f^{\circ T_i}\) **exactly, on every unit**. Every final-answer-only loss is therefore invariant under \(c \in \mu_g(N)\): no optimizer, architecture search, or training schedule using only these labels can distinguish the true one-step map (\(c = 1\)) from any gauge partner. Certification at \(T = 1\) requires \(c^{2^1-1} = c = 1\).

Instantiated on the public tiers (units only, scalar gauge; \(\#\mu_g\) multiplies over CRT factors as \(\gcd(g, p{-}1)\cdot\gcd(g, q{-}1)\)):

| tier | \(D\) | \(g\) | \(\mu_g(N)\) nontrivial? |
|---|---|---|---|
| E1, E2, E5 | \(\{1,2,3\},\{1,2,4\}\) | 1 | no — depth-1 identified by data |
| M3 | \(\{2\}\) | 3 | yes when \(3 \mid p{-}1\) or \(q{-}1\) |
| M5 | \(\{2,4,8\}\) | 3 | same |
| **M1/M2** | \(\{4,8,16\}\) | 15 | **yes, concretely**: \(N{=}10403 = 101{\cdot}103\) gives \(\gcd(15,100)\cdot\gcd(15,102) = 5\cdot3 = 15\) roots — 14 wrong one-step maps fit all 90k labels perfectly |
| M4 | \(\{8\}\) | 255 | generically yes |

Same-\(x\) multi-T labels don't help: the invariance is per-depth, so pairing constrains only deeper composites. **Corollary:** on any tier whose depth set has \(g > 1\) — which includes all of Medium except by luck, and must be assumed for Hard — rung-1 behavior is determined by the prior, not the data. Your objection was correct, and it is provably unfixable by any learned mechanism, because the loss is *exactly* invariant.

**Weakest legal bias that changes the conclusion:** a gauge-fixing anchor at the multiplicative identity. Since \(G_c(1) = c\), enforcing \(\widehat G(1, n) = 1\) as architecture collapses \(\mu_g\) to \(\{1\}\):

\[
\widehat G(s, n) \;=\; G_\theta(s, n) \;-\; G_\theta(\iota(1), n) \;+\; \iota(1),
\]

evaluated in the pre-quantizer embedding space (two cell evaluations per step, one at a constant input). This hard-codes no arithmetic and no weights — it is a one-point reparameterization, the same species as a skip connection; \(\iota(1)\) uses only the known digit vocabulary. If judges read it as rule-7-adjacent, the fallback is capacity minimization (small cell + weight decay), which merely *prefers* \(c{=}1\) since a nontrivial constant-multiplier needs extra circuit — a heuristic, not a fix. **Residual honesty:** the anchor kills the exhibited scalar gauge; the full solution set of \(G^{\circ T_i} = f^{\circ T_i}\) may contain non-scalar compositional roots. Complete depth-1 identifiability on \(g{>}1\) tiers remains Assumption A1, now with the obstruction's shape known.

## 3. Cross-N exactness — assumption, named

Across \(N\), training rows are ordinary supervised examples of \((N, u) \mapsto u^2 \bmod N\) (at composed depths). Exactness on *unseen* \(N\) is pure function-level generalization: a representability argument exists (schoolbook multiply + reduction is a polynomial-size circuit; a transformer can express it) but no SGD-findability guarantee. This is **Assumption A2**, the principal failure mode, and no legal mechanism removes it. The weakest bias that plausibly changes it: a digit-place register with locality-structured attention (carry propagation is local), plus the given bidirectional prompt attention. Still an assumption.

## 4. Supervision and loop control — corrections

**Loss.** You are right: rows in a batch have different \((N, x)\), so a row's label supervises only its own rollout at its own \(T\). The multi-T consistency loss is deleted. The entire training signal is

\[
\mathcal L = \frac{1}{B}\sum_{i} \mathrm{CE}\!\left(r\big(s^{(i)}_{T_i}\big),\; y_i\right), \qquad s^{(i)}_{t+1} = Q\big(\widehat G(s^{(i)}_t, n_i)\big),
\]

with \(Q\) a straight-through one-hot per digit slot (hard feedback from step 0 — no train/eval gap), plus weight decay (grokking-regime). A max-over-digit-positions emphasis is a cheap legal lever against the product-of-probabilities certification metric; optional.

**Loop count.** Halter deleted. Its only information source is the \(T\) already in the prompt; supervising it means distilling the prompt into a controller (circular, adds a failure mode), and training it from final answers alone is underdetermined — the model can select step \(k \neq T\) and compute just-in-time, evading the induction. Instead: parse the integer \(T\) from the prompt's digit tokens and unroll exactly \(T\) times (per-row; gather each row's readout at its own step inside a max-\(T\) group loop). **Assumption A3:** this is input-dependent control flow — adaptive computation, explicitly allowed — not a hard-coded arithmetic algorithm; every correct solution must obtain an iteration count from the prompt somehow. The fully learned fallback (soft content-based selection over a fixed 64-step unroll) is strictly worse and I do not recommend it.

## 5. Corrected design (Family A, repaired)

```python
class Submission(nn.Module):
    # boundaries: Encoder | N-conditioner | Cell(anchored, tied) | Quantizer | Readout
    def forward(self, input_ids, attention_mask=None):
        tok = self.embed(input_ids)                    # + digit-place (from-the-right) code
        n_cond = self.n_read(tok, attention_mask)      # bidirectional over N field
        T = parse_T(input_ids)                         # A3: integer from T-field tokens
        s = self.reg_encode(tok)                       # canonical digit slots of x
        one = self.reg_encode_const(digit=1)           # gauge anchor input (Theorem 2)
        outs = []
        for t in range(T.max()):
            g   = self.cell(s, n_cond)                 # tied, time-homogeneous, 2 layers
            g1  = self.cell(one, n_cond)               # anchor: Ĝ(1,n) ≡ 1  ⇒  c = 1
            s   = self.quantize(g - g1 + one)          # ST one-hot: canonical re-encode
            outs.append(self.readout(s))
        out = torch.stack(outs, 1)                     # [B, T_max, ans_len, V]
        idx = (T - 1).view(-1, 1, 1, 1).expand(-1, 1, *out.shape[2:])
        return out.gather(1, idx).squeeze(1), None
```

Loss: per-row CE at the gathered step over answer digits, `token_training_loss` with `valid_mask`. Optimizer: AdamW, cosine, real weight decay; `backward_passes_per_step` and batch reuse as wall-clock levers on hard residual examples.

**Cost (corrected — my "negligible" was unjustified):** an 8-layer d=512 cell × 64 steps is a ~500-effective-layer forward per example; defensible but pointless. The cell is deliberately minimal — 2 layers, d=256, ≈2M params total (500M ceiling is irrelevant; small capacity is itself the depth-prior that disfavors atomic \(k\)-step factorizations, and it buys optimizer steps under the 3,600 s clock). Eval worst case: 64 steps × 2 layers ≈ 128 layer-passes on ~50 tokens, batch-parallel — milliseconds per 10³ examples, orders of magnitude inside the 1,800 s budget. Training per step costs \(\bar T\) cell applications; M1's \(\bar T \approx 9\) is cheap, an all-\(T{=}64\) Hard batch is the worst case and still fine.

**Failure modes, ranked:** (1) A2 — one-step map not exact on unseen \(N\) ⇒ OOD-N capped at rung 0–1; this is where the competition is lost. (2) A1 residual — non-scalar compositional roots surviving the anchor. (3) A3 rejected by judges ⇒ fall back to learned selection, expect depth-identifiability loss. (4) 3,600 s too short for the grokking transition at Hard bit sizes.

## 6. Public squaring vs Generalised Hard

- **Public squaring:** exactly the above — homogeneous cell, no time input at all. Extrapolation in \(T\) *is* Theorem 1; nothing per-\(t\) may be learned.
- **Generalised Hard:** if the recurrence varies with \(t\), homogeneity must be relaxable without learned per-\(t\) parameters: feed the cell a **fixed** (non-learned) Fourier code of \(t\) — any unseen \(t\) then has a well-formed input, and no \(\tau_{64}\) can be untrained. The anchor generalizes as a principle (fix the gauge of the one-step map at its identity-like point) but its exact form is recurrence-dependent; the transferable core is: minimal tied cell, canonical bottleneck state, per-row final-answer CE, depth controlled by the prompt.

## 7. Ledger

**Theorems.** T1: conditional loop invariance (H1–H3). T2: \(\mu_g\)-gauge invariance of final-answer-only supervision, with \(g = \gcd(2^{T_i}{-}1)\); non-identifiability of the one-step map on all Medium training sets, concrete at \(N{=}10403\); anchor collapses the scalar gauge.
**Assumptions.** A1: no non-scalar compositional roots survive anchor + minimal capacity. A2: SGD finds the exact cross-\(N\) one-step circuit in budget. A3: parsing \(T\) as loop count is legal adaptive computation. A4: the identity anchor is legal gauge-fixing, not a hard-coded algorithm.

The single sentence version: final-answer-only supervision on geometric depth sets provably cannot identify the one-step map (Theorem 2), so the minimal legal design is a tied, identity-anchored, capacity-starved cell over a canonical digit register, unrolled exactly \(T\) times — and the competition is won or lost entirely on Assumption A2, exactness of that cell on unseen moduli.

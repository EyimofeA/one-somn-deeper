# Cross-examination answer: the T=1-first submission

Labels used throughout: **[T]** theorem (proved here), **[A]** assumption, **[P]** empirical prediction, **[S]** speculation.

---## 1. The one design: DRM-min (Digit Register Machine, minimal)

Everything below is random-init, GPU-resident, single forward returning `(logits (B,S,17), None)`, fits `build_model/build_optimizer/Submission`, and reads labels only through the evaluator's CE. Total trainable state ≈ 25k elements.

**Parameter-free structure (README-sanctioned "routing … parameter-free work"):**
- Parse: gather digit positions after the `N`/`X`/`T` marker ids; one-hot the digits. Encoder is **weightless**: register `z₀ ∈ (B, W, 10)` is literally the one-hot digits of x, W = 2·L_N + 2 slots.
- Decoder is **weightless**: register one-hots are emitted (scaled by one learned scalar temperature) as digit-token logits at the tail target slots; PAD logit elsewhere.
- Anti-diagonal pairing tensor Δ_{ijk}=[i+j=k]; shift-by-j alignment; loop count parsed from the T digits; per-row halting mask.

**Learned state (all of it):**

| Module | Shape | Role |
|---|---|---|
| τ_lo, τ_hi | (10,10,10) each | digit-pair product → low/high digit logits |
| ADD | (10,10,2,11)-equivalent MLP ≈ 3k | two digits + carry → digit, carry |
| SUB | ≈ 3k | two digits + borrow → digit, borrow |
| CMP | ≈ 2k | per-position ≥/=/< cell, combined by parallel prefix |
| scalars | τ_ST temperature, logit scale, step-counter buffer | annealing knobs |

**Step** (weight-tied, applied T times): square via Δ-scatter of τ outputs → tree-reduce columns with ADD (log₂W depth) → reduce loop over shifts j = W−L_N…0, where the ten candidate multiples m·N are produced *by the same τ/ADD path* applied to (m, digits-of-N), CMP picks q_j, SUB subtracts → straight-through rounding (hard forward, identity backward) back onto one-hot registers.

**Removed as merely nice-to-have:** learned token embeddings, learned encoder/decoder (weightless versions make canonicality *structural*, so the previous input-reconstruction anchor loss and entropy penalty are deleted — nothing to anchor), auxiliary output (return None), multi-pass backward, batch reuse, generic-recurrence router (deferred; see §6), accuracy-triggered gates (v1 uses step-fraction gates only).

---

## 2. The three ghost families, confronted under the actual parameterization

Setting: training depth set 𝒯 lacking T=1 (worst case 𝒯 = {4,8,16}). The network's induced step is s_N(x) = value of the register pipeline; the squaring path computes the N-independent integer function
Q(x) = Σ_{i,j} τ(x_i, x_j)·10^{i+j} (post-carry), and the reduce path subtracts Σ_j 10^j·Q̃(q_j, N), where Q̃ is the *same* learned machinery applied to (single digit, N-digits).

**Define condition (RP), residue preservation:** the learned reduce path only removes quantities ≡ 0 (mod N), i.e. Q̃(m, N) ≡ m·N ≡ 0 (mod N) for m = q_j actually selected. (RP) is a property of learned weights, **not guaranteed by wiring**, because Q̃ is learned. It is measurable locally (§3, X2) and I state everything relative to it. Under (RP): s_N(x) ≡ Q(x) (mod N) with Q an N-independent integer function whose values are bounded by B_Q ≤ 200·W·10^{2W} and whose small-argument values are single table entries.

**(a) G_c(x) = c·x², c^g ≡ 1 (mod N), g = gcd_{T∈𝒯}(2^T−1) = gcd(15,255,65535) = 15.**
Iterating gives G_c^{∘T}(x) = c^{2^T−1}x^{2^T}, so G_c matches every training depth iff c^15 ≡ 1 — a genuine ghost family in the abstract.
**[T]** Under (RP), no exact (a)-ghost with c ≢ 1 exists in the class. Proof: if s_N(x) ≡ c_N·x² (mod N) for **all** units x of each modulus (that is what "the class contains the ghost" means — a statement about the map, not about finitely many evaluations), evaluate at x = 1: c_N ≡ Q(1) = τ-entry(1,1), a single bounded integer c independent of N. Then c^15 ≡ 1 (mod N) for every training modulus forces N | c^15 − 1; since c ≤ 200 and the training moduli are numerous, coprime, and their product exceeds 200^15, c^15 = 1 over ℤ, so c = 1. ∎ (No polynomial-identity-from-finite-samples move is used; only representability plus evaluation at one point.)

**(b) h(x) = x^{−2}, matching every even T since (−2)^T = 2^T.**
**[T]** Under (RP), no exact (b)-ghost exists in the class. Proof: s_N(x) ≡ x^{−2} (mod N) for all units implies at x = 2 (unit for odd N): 4·Q(2) ≡ 1 (mod N), so N | 4·τ-entry(2,2) − 1, a fixed odd nonzero integer < 10³ — impossible once any training modulus exceeds it. ∎
**Honest converse:** if (RP) fails, exclusion of (b) is **unprovable**: x^{−2} is uniformly poly-time computable (extended Euclid), so uniformity alone cannot exclude it, and a non-residue-preserving reduce path is exactly the extra wiring an inverse-flavored solution would exploit. Recovery of T=1 then rests on (RP) being pushed by training — **[P]** CE pressure at multiple depths favors (RP), because a reduce path that subtracts v ≢ 0 (mod N) corrupts the residue class at *every* iteration and damages all supervised depths simultaneously; there is no gradient valley in which non-(RP) reduction and correct T∈{4,8,16} answers coexist except the measure-zero ghosts themselves. This is a prediction, not a theorem, and X2 in §3 tests it directly.

**(c) Permutation roots matching all T ≥ 2.** From h² = f² and h³ = f³: h∘f² = f∘f², so h = f on im(f²) = G⁴ (the quartic residues, index = #{x:x⁴=1} ≤ 16), while h may differ from f on up to 15/16 of G. These ghosts match every rung T ≥ 2 exactly and fail T=1 on a constant fraction of inputs.
**No algebraic exclusion exists** — small-point evaluation cannot touch them because they agree with f wherever we can pin values, and they are not exponent maps. The only available argument: an in-class (c)-ghost is an N-uniform, ~25k-parameter circuit whose T=1 behavior distinguishes quartic residues modulo semiprimes without the factorization — a problem believed intractable (quadratic/quartic residuosity assumptions). **[S/A]** So: *if the training distribution lacks T=1 and lacks all odd depths, exact T=1 recovery against family (c) is impossible to guarantee; it is an Occam bet.* The bet is unusually strong — the ghost must spend cryptographic work the hypothesis class cannot express and SGD receives zero gradient incentive to find — but it must be stated as a bet. Verdict: (a), (b) excluded conditionally on (RP) **[T]**; (c) excluded only by a complexity-theoretic Occam bet **[A]**; the ladder below is designed to buy evidence on both.

---

## 3. Six-experiment remote-GPU ladder

Local runs use the 48GB diagnostic GPU with datasets *recreated by the public generator* (`data/squaring_mod.py` is public; local diagnostics are unrestricted — only official runs forbid data inspection). Primary metrics everywhere: **T=1 seen-N exact** and **T=1 OOD-N exact**. Artifacts saved every run: per-depth exact/CE curves vs. step, checkpoint, and the two probes: table-probe (argmax of τ/ADD/SUB/CMP vs. ground-truth digit arithmetic — a probe of *our own weights*, legal locally) and (RP)-probe (fraction of reduce-path subtractions ≡ 0 mod N on a held-out batch).

| # | One change | Data / tier | Cap | Promote if | Abandon if | Predicted **[P]** |
|---|---|---|---|---|---|---|
| X1 | baseline | local, sampled N 10–11 bit, `fixed_time_steps=1` | 30 min | OOD-N T=1 ≥ 0.99 | table-probe < 80% at 15 min | pass; T=1 directly supervised, mechanism sound |
| X2 | depth set → `{2}` only | local, same N family | 45 min | **T=1 OOD ≥ 0.95 with T=2 certified** and (RP)-probe ≥ 0.999 | T=2 exact ≥ 0.99 but T=1 < 0.5 → dump predictions, test against x^{−2} mod N and quartic-residue-conditioned maps | ~60% pass — this is the identifiability experiment; a fail here is the single most valuable datum we can buy |
| X3 | depth set → `{4,8,16}`, fixed N=10403 (M1 recreation) | local | 600 s train + eval | local rungs 1–16 certified, T=1 OOD-N (15–16 bit) ≥ 0.9 | T=1 seen < 0.9 while T=4 ≥ 0.99 (even-depth ghost captured at scale) | pass if X2 passed |
| X4 | local → **official evaluator** | Easy `e3` (sampled N, T=2) | 60 s, ≤10 attempts | leaderboard depth profile T=1 rung = 1.0, no timeout | crash/timeout after 3 attempts → harness bug hunt, not science | pass; contract risk only |
| X5 | scale | Medium `m3` then `m5` | 600 s, ≤4 attempts | certified prefix ≥ T=8, OOD-N T=1 = 1.0000 | seen-N T=1 < 0.999 after 2 lr-tuned attempts | certified T=1–8 |
| X6 | largest-N, single composite depth (closest Hard proxy) | Medium `m4` (T=8 only, 14–22 bit N) | 600 s, ≤2 attempts | T=1 rung = 1.0 both profiles → cleared for Hard | T=1 < 0.9 with T=8 certified → ghost at scale: abandon skeleton, escalate the §5 organizer question, fall back to odd-depth-hoping generic variant | the decisive dress rehearsal |

---

## 4. Training schedule and loss — only legal signals

- **Loss:** the evaluator's default masked CE on the tail digit slots. No custom `token_training_loss` in v1 — with weightless encoder/decoder there is nothing to anchor, and every deleted callback is a deleted failure surface. (v2, only if X5 shows a near-miss exactness gap: custom loss = CE + margin on the argmax gap, computed purely from the logits/labels the loss already receives.)
- **Optimizer:** AdamW, lr 2·10⁻³ → 1·10⁻⁴ cosine, β₂ = 0.95, wd 0, grad-clip is evaluator-owned. Batch 512 train / 2048 eval.
- **Annealing without callbacks:** a step-counter buffer incremented inside `forward` when `self.training`; ST temperature τ: 1.0 → 0.1 linearly over the first 40% of the step budget (budget estimated from `OptimizerSpec.training_time_seconds` at build time), hard rounding thereafter. Gates are step-fraction-based only — deterministic, auditable, no loss-side statistics needed.
- Explicitly **not** used: intermediate arithmetic labels (none exist), reduce-path supervision (none exists), consistency losses against computed ground truth (illegal oracle), participant backward, batch reuse, data-order tricks (evaluator-owned).

---

## 5. Strongest remaining legality risk and the resolving question

Risk: Rule 7. DRM-min's dataflow — i+j=k pairing, shift-aligned conditional subtraction, loop count parsed from T — is the *skeleton* of schoolbook multiply-and-reduce with every arithmetic fact learned. The README blesses "routing … and parameter-free work," but an organizer could rule that routing which mirrors a known algorithm is itself the algorithm.

Exact question for `#one-layer-deeper` (before the first Hard attempt): *"Rule 7 boundary: is a forward pass legal when all arithmetic content (digit-product/add/subtract/compare cells) is randomly initialized and learned end-to-end, but the dataflow is fixed parameter-free routing — an i+j=k pairing tensor, shift alignment, and an iteration count read from the T tokens — analogous to a convolution's index structure? Or does fixed routing that mirrors long multiplication constitute a hard-coded algorithm even when it computes garbage at initialization?"* Secondary rider: *"Is straight-through estimation an 'unbroken gradient path' under Rule 8?"*

---

## 6. First submitted file

**Include:** DRM-min exactly as §1 (registers, four learned cell tables + two scalars, Δ routing, ST with the §4 anneal, T-loop with halting mask), `assert_model_state` call, `batch_size=512`, `eval_batch_size=2048`, default CE, default max_steps. Nothing else.
**Omit:** custom loss, scheduler callbacks beyond lr, multi-pass, batch reuse, aux outputs, the generic-recurrence router ("Hard may alter the recurrence" is real, but a router doubles the untested surface; it enters only after one clean Hard result), any torch.compile (budget-billed, unpredictable).
**Expected failure mode [P], ranked:** (1) *optimization plateau* — gradients through T·(log W + L_N) ST-rounded stages too weak in the soft phase, CE stalls near ln 10 ≈ 2.30 exactly like the observed 2.178 run; diagnosis: table-probe flat in X1-style local replay; remedy: longer soft phase, per-stage τ. (2) Eval-budget overrun at the T=64 rung with large OOD-N registers. (3) Only third: the ghost outcome of §2 (fails T=1 while certifying even rungs) — which, note, would be *visible* as a specific signature, not a generic zero.

---

## 7. The question this roundtable failed to ask

**Does the Hard training stream contain T=1 (or any odd depth) at all — and what legal, cheap measurement answers it?** We spent two rounds on identifiability conditional on "T=1 omitted," but the only evidence — one run at CE 2.178 ≈ ln 10 — is evidence of *learning nothing*, and therefore evidence about the depth distribution of **nothing**. The entire ghost analysis of §2 is moot if h1 trains on odd depths. A candidate measurement exists: the evaluator's metric recorder logs overall train exact-accuracy over steps; a submission that reliably solves *only* shallow instances early would make the train-accuracy curve reveal the depth histogram of the hidden stream in a single Hard attempt. Whether that reading of one's own telemetry falls under Rule 16's "must not exploit the metric recorder" is itself unresolved — which makes it the second question to put to the organizers, and possibly the cheapest experiment in the whole program.

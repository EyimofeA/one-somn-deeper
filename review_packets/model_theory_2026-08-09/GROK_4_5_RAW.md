**Note:** `onelayerdeeper.com` does not resolve; the public site is [https://onelayerdeeper.ai](https://onelayerdeeper.ai). Sources: README ([tilde-research/one-layer-deeper](https://github.com/tilde-research/one-layer-deeper)), problem/submit pages, and the Tilde blog.

---

## 1. Public I/O, supervision, recurrence, legality

### Facts [SOURCED]
- **Map (public task):** \(x_0=x\bmod N\), \(x_t=x_{t-1}^2\bmod N\), label \(y=x_T=x^{2^T}\bmod N\). \(N=pq\) with \(p,q\) secret; evaluator may use \(\varphi(N)\) to label. ([problem](https://onelayerdeeper.ai/problem), [blog](https://blog.tilderesearch.com/blog/one-layer-deeper))
- **Prompt shape:** field markers + decimal digits, conceptually `N77X2T4ANS9`; target = decimal digits of \(y\). ([problem](https://onelayerdeeper.ai/problem))
- **Interface:** `build_model(spec)` gets `vocab_size`, `max_seq_len`, state ceiling; forward returns `(logits, auxiliary)`; optional `token_training_loss` / `training_loss`; evaluator owns data order, forward/loss/backward, clip, opt cadence, seeds, deadline, eval. Easy/Medium: separate prompt/output tensors, **padding mask (not causal)** → bidirectional over full prompt. Hard: private. ([README](https://github.com/tilde-research/one-layer-deeper))
- **Recurrence:** allowed (tied loops, ACT, curricula, memory tokens). Depth unconstrained; clock trades depth vs updates. ([README](https://github.com/tilde-research/one-layer-deeper))
- **Hard score:** certify consecutive prefix of \(T\in\{1,2,4,8,16,32,64\}\) at 100% exact-example accuracy; rank Max T (seen-\(N\) families) then OOD-\(N\) Max T then first uncertified rung accuracy. ([README](https://github.com/tilde-research/one-layer-deeper))
- **Hard caveat:** “may change aspects of the recurrence itself; **do not assume it is repeated squaring**.” ([problem](https://onelayerdeeper.ai/problem))
- **Legality:** ≤500M params+persistent buffers; no hard-coded weights (`torch.load` etc.); no hard-coded forward algorithm; unbroken autograd from loss → predictive params; GPU-only (no CPU offload); no data inspection/augmentation, no participant backward, no hidden training loops. Custom loss OK if one finite differentiable scalar; evaluator `backward`s. ([README](https://github.com/tilde-research/one-layer-deeper))
- **Practice mix:** e.g. E1 has \(T\in\{1,2,3\}\); **E3/E4 fixed \(T=2\)** (no \(T=1\)). ([submit UI snippets in search](https://onelayerdeeper.ai/submit))
- **Empirical (your report):** 1.59M tied digit-register, 3600s / 163274 updates, CE 2.17846, **0/768** seen-\(N\) and OOD-\(N\) at \(T=1\).

### Deductions
- Final supervision is **only** exact digit string of \(y\) for the prompted \(T\); no public intermediate \(x_t\) labels.
- Certifying Max \(T\ge 1\) **requires** exact \(T=1\). A model that never solves one modular square cannot climb the ladder.
- CE≈2.18 with 0 exact ≈ weak token marginals, **zero** full-residue events — failure is at the one-step map, not depth extrapolation.

### Assumptions
- Hard training still presents some distribution over serial depths and moduli; exact Hard generator is private.
- “No hard-coded algorithm” bans integer/oracle modular squaring in the forward, not a **learned** digit/limb circuit with random-init weights.

### Speculation
- Hard may replace \(x\mapsto x^2\) by another secret one-step \(f_N\) still composed \(T\) times — so mechanisms that **only** implement squaring may fail transfer even if legal on Easy/Medium.

---

## 2. Identifiability when depths omit \(T=1\)

**Setup.** Residues \(R_N=\{0,\ldots,N-1\}\). True step \(f_N(x)=x^2\bmod N\). Training observes pairs \((x,y)\) with \(y=f_N^{\circ T}(x)\) only for \(T\in\mathcal{S}\subseteq\{2,4,8,\ldots\}\) (as in E3/E4).

**Theorem (even-depth non-identifiability).**
Let \(\sigma:R_N\to R_N\) satisfy \(\sigma\circ\sigma=\mathrm{id}\) and \(\sigma\circ f_N=f_N\circ\sigma\). Define
\[
g_N := f_N\circ\sigma.
\]
Then for every \(k\ge 1\),
\[
g_N^{\circ 2^k}=f_N^{\circ 2^k}.
\]
In particular, if \(\mathcal{S}\subseteq\{2,4,8,\ldots\}\), then \(g_N\) and \(f_N\) induce **identical** supervised labels on \(\mathcal{S}\), while \(g_N=f_N\) iff \(\sigma=\mathrm{id}\).

**Proof.**
\(g_N^{\circ 2}=f_N\sigma f_N\sigma=f_N(f_N\sigma)\sigma=f_N^{\circ 2}\sigma^{\circ 2}=f_N^{\circ 2}\) using \(\sigma f_N=f_N\sigma\). Induct: if \(g_N^{\circ 2^{k-1}}=f_N^{\circ 2^{k-1}}\), square both sides.

**Explicit nontrivial family (on units).**
On \((\mathbb{Z}/N\mathbb{Z})^\times\), let \(\sigma(x)=x^{-1}\). Then \(\sigma^{\circ 2}=\mathrm{id}\) and
\[
\sigma(f_N(x))=(x^2)^{-1}=x^{-2}=f_N(\sigma(x)),
\]
so the theorem applies. The alternate step
\[
g_N(x)=f_N(\sigma(x))=x^{-2}\bmod N
\]
satisfies \(g_N^{\circ 2^k}=f_N^{\circ 2^k}\) on units, but \(g_N(x)=f_N(x)\) iff \(x^4\equiv 1\pmod N\) for all units — false for typical semiprimes (e.g. \(N=323=17\cdot 19\)).

**Corollary.** Datasets that **omit** \(T=1\) do not pin down the one-step map among \(\{f_N\circ\sigma:\sigma\in\mathrm{Cent}(f_N)\cap\mathrm{Inv}\}\). Any learner fit only on even compositions may lock onto a wrong conjugacy class and score **0 at \(T=1\)** while fitting \(T=2,4,\ldots\) in-distribution.

**Extra legal bias that restores uniqueness:** include **\(T=1\) mass** in training (or an aux objective equivalent to supervising one application of the step). Without that, \(T=1\) is underdetermined as a map.

---

## 3. Solve \(T=1\): minimum learnable mechanism

**Target:** learn a family \(\hat f_\theta(\cdot;N):R_N\to R_N\) such that \(\hat f_\theta(x;N)\approx x^2\bmod N\) for **unseen** \(N\), without baking in integer `pow`/`%`.

**Necessary representation (not assumed exact).**
Token CE on digit strings does not expose place-value, carries, or reduction. A mechanism that generalizes in \(N\) needs an explicit **place-value register** for \(x\) and \(N\) (base-\(B\) digits or limbs) and a **length-adaptive** multiply–reduce pathway whose work scales with \(\lceil\log_B N\rceil\), not a fixed token mixer that memorizes prompt→answer.

**Why ordinary token regression fails.**
The map \(\mathrm{digits}(N,x)\mapsto\mathrm{digits}(x^2\bmod N)\) is discontinuous in token space (one-digit change in \(N\) reshuffles the residue). Bidirectional Transformers can attend, but without a register bottleneck they can fit local digit co-occurrence; exact full-string match is a measure-zero event under independent digit noise. Your CE 2.18 + 0/768 is the empirical signature: marginals move, **joint** residue never lands.

**Minimal mechanism (learnable, not claimed exact):**
1. Parse prompt → digit tensors \(X,N\in\{0,\ldots,B-1\}^{L}\).
2. One **shared** digit-serial (or limb-serial) cell implementing a soft transition for schoolbook multiply or Horner multiply-accumulate, conditioned on modulus limbs.
3. One **shared** reduction pathway (soft long-division / Barrett-style residual), same weights across lengths.
4. Decode residue digits → label logits.

Prior art for “learned cell + fixed serial schedule” exists in modular-arithmetic challenge settings (e.g. bit-serial Horner multiply-mod); here the schedule must be driven by parsed length, weights random-init and trained.

**Inductive bias that resolves underdetermination:** force all paths from input to logits through a **single** \(L\)-digit residue register updated by one multiply-mod block when the parsed \(T\) encodes one step (or always for a \(T=1\)-only curriculum). Without that bottleneck, many soft programs fit CE.

---

## 4. One concrete legal architecture

### State / tensors
- Prompt tokens \(t\in\mathbb{Z}^{B\times S}\)
- Soft digits: \(X,N,Y\in\mathbb{R}^{B\times L\times B_{\mathrm{rad}}}\) (logits or probabilities over radix \(B_{\mathrm{rad}}=10\))
- Scratch limbs (optional): \(M\in\mathbb{R}^{B\times 2L\times B_{\mathrm{rad}}}\) for product before reduce
- Control: length mask \(\ell_N\in\{1..L\}^B\) from parse
- Output: `logits` aligned to target slots \([B, T_{\mathrm{out}}, V]\); `aux` = soft digit tensor for custom loss

### Modules (pseudocode)
```text
parse(t) -> X_digits, N_digits, T_scalar, out_len
# soft one-hots; no integer arithmetic oracle

# one tied multiply-mod block (T=1 uses once)
P = SoftSchoolbookMul(X_digits, X_digits)     # [B,2L,B_rad]
Y = SoftReduceMod(P, N_digits)               # [B,L,B_rad]  shared cell, ℓ_N steps

logits = DigitToTokenHead(Y, out_len)        # only residue digits → ANS field
return logits, aux=Y
```

`SoftSchoolbookMul` / `SoftReduceMod`: fixed nested loops over digit index (control flow OK); **learned** digit embeddings + small MLP/GRU cell for write/carry/borrow. Random init; all ops in autograd on GPU.

### Optimization / curriculum
- Tier: Hard clock 3600s H100; diagnostic 48GB: keep \(L\) modest, batch large, **cheap** one-step forward → maximize updates (your 1.6M param scale is fine; depth not the issue yet).
- Loss: sequence-mean CE on valid target tokens **plus** (legal) aux that sharpens digit marginals without integer labels:
  - entropy penalty on \(Y\) digits (push toward peaked codes),
  - consistency: `SoftReduceMod(SoftMul(Y,1),N)≈Y` (idempotent reduce),
  - optional self-check on reconstructed numeric expectation vs \(N\) (differentiable polynomial in digit probs — **not** an oracle label).
- Curriculum: **phase A** parse+reduce-only (\(x\mapsto x\bmod N\)); **phase B** multiply-mod; gate by training exact-match EMA on the current batch’s teacher digits of \(y\) (available as `labels`).

### Exact promotion gates (internal)
- Gate A→B: EMA exact-match \(\ge 0.95\) on reduce-only probes for \(K\) consecutive eval-style checks **inside** forward using labels only at loss (no data inspection API).
- Gate \(T=1\) “solved enough” for recurrence: EMA full-string exact \(\ge 0.99\) on minibatches with parsed \(T=1\) for \(W\) windows; only then enable multi-apply.

### Final-label gradients vs soft scratchpad
**Problem:** CE on final tokens does not uniquely determine intermediate soft codes.

**Legal fix (pick one primary):**
1. **Bottleneck + entropy:** single register \(Y\); aux entropy + reduce-idempotence; final head is a **linear readout of \(Y\)** only (no residual bypass from raw prompt tokens to logits).
2. **Straight-through at loss:** train with soft \(Y\); in the loss path, `Y_hat = soft + (one_hot(argmax)-soft).detach()` so CE sees nearly discrete digits while cells get soft gradients.
3. **Score-function:** discrete digits, REINFORCE with self-critical baseline on exact-match reward; still one scalar, evaluator `backward` through the surrogate.

Ban residual “prompt→logits” highway until phase B clears the gate — otherwise CE bypasses the register (matches your 0/768 pattern).

---

## 5. Recurrence (only after a defensible \(T=1\))

Once \(\hat f_\theta(\cdot;N)\) clears the \(T=1\) gate:
\[
Y_{k}=\hat f_\theta(Y_{k-1};N),\quad k=1..T_{\mathrm{parsed}},\quad Y_0=X\bmod N.
\]
Tied weights; at eval, allow \(T\) beyond train (extrapolation). Depth curriculum on parsed \(T\). **Hard caveat:** treat \(\hat f_\theta\) as “learned step,” not “square”; if Hard changes \(f\), the same register+cell must re-fit from Hard labels — squaring-specific integer structure would both be illegal and brittle.

---

## 6. Red team

| Attack | Content |
|--------|---------|
| **Strongest legality objection** | Fixed schoolbook/Horner **control flow** + modulus-conditioned cell is a “hard-coded algorithm” with learned constants. Risk if organizers read structure as the algorithm. Mitigation: keep cell expressive (not a thin wrapper around `a*a%n`); no integer `N` built outside autograd; all digit ops through params. |
| **Mathematical counterexample** | Even-depth conjugacy (§2): model learns \(g_N=x\mapsto x^{-2}\) on units, fits \(T=2,4,\ldots\), **fails \(T=1\)**. |
| **Fastest falsifier** | Hold out a tiny \(T=1\) probe set (legal local Easy) after training only on \(T=2\): if \(T=2\) accuracy high and \(T=1\) ~0, non-identifiability realized. Second: OOD \(N\) with same bit length — if in-domain digits memorize, OOD collapses. |
| **Abandon condition** | After full Hard budget, CE≪log V but exact-match EMA on \(T=1\) stays \(<10^{-3}\) **and** ablations show logits still bypass the register → abandon this cell family; switch radix/limb width or discrete score-function before adding recurrence. Your cited run already meets the “no \(T=1\) signal” half of this. |

---

## 7. Ranked top 3 + bold alternative

1. **\(T=1\) digit/limb multiply–mod register with no logit bypass + \(T=1\) mass** — only path that is both identifiable and OOD-capable.
2. **Even-depth conjugacy control** — never train depth sets that omit \(T=1\) without an aux that breaks \(\sigma\)-families.
3. **Tied recurrence of the same block** — only after exact \(T=1\) gates; otherwise depth amplifies the wrong map.

**Bold alternative:** forget soft digits; learn a **CRT-shaped pair of registers** \((r_p,r_q)\) with unknown factors represented as **latent** divisors inferred end-to-end (illegal if you hard-code factoring; legal if latents are free and decoded by \(y\equiv r\pmod N\) consistency). High variance, but if Hard changes the outer recurrence and keeps ring structure, a residue-pair state may transfer better than a squaring-specialized cell.

---

**Skipped:** generic Transformer stacking, CoT token scratchpads as the main plan, and any recurrence discussion as a substitute for a working one-step. **Add** recurrence only after a \(T=1\) exact-match gate on seen and OOD \(N\).

**Default assumption:** ship **structured mul(x,x)→reduce** (best ambiguity exclusion); ask organizers Rule-7 before Hard. Pi UI unavailable for the fork.

---

## 1. One submission design (minimal)

**API surface:** one `submission.py` → `Submission(build_model, build_optimizer, token_training_loss, batch_size, eval_batch_size, max_steps)`. Model: `forward(input_ids, attention_mask=None) → (logits, aux)`; `config.vocab_size/max_seq_len`; random-init params; GPU only; ≤500M state.

**State (batch \(B\), digit length \(L\), radix 10, embed \(d\)):**
| Tensor | Shape | Role |
|--------|--------|------|
| `tok` | \([B,S]\) | prompt ids |
| `X,N` | \([B,L,10]\) | soft digit probs (place-value) |
| `P` | \([B,2L,10]\) | soft product digits |
| `Y` | \([B,L,10]\) | soft residue |
| `logits` | \([B,T_{\mathrm{out}},V]\) | **only** from `Y` (no prompt residual) |
| `aux` | `Y` | unused by loss except optional entropy on `Y` if you keep it—**drop entropy** under §4; `aux=None` |

**Modules (nothing else):**
1. **Parse** — marker-aware slice of digit tokens → one-hot/soft `X,N` (learned embed of digit ids → 10-way; no integer `int()` oracle).
2. **SquareMul** — fixed nested loops over digit indices; **same operand twice**; learned digit-cell for write/carry only. Output `P`.
3. **ReduceMod** — fixed length-adaptive loops over `ℓ_N`; learned digit-cell for quotient/remainder digits. Output `Y`.
4. **Head** — `Y` → token logits at ANS positions only.

**Removed as nice-to-have:** recurrence wrapper, ACT, multi-block banks, consistency aux, reduce-only phase, CRT latents, bypass Transformer, custom optimizer curvature, batch-reuse tricks (until T=1 works).

**Optimizer:** AdamW on all params; large batch; cheap one-step forward.

**Empirical prediction:** matches the failed 1.59M run’s *capacity* band, but changes the *information path* (forced square bottleneck). Prior run’s 0/768 is predicted if bypass/CE-only digit soup remains; this design forbids bypass.

---

## 2. Ambiguity families vs this parameterization

Let \(f(x)=x^2\bmod N\). Training labels \(y=f^{\circ T}(x)\) for \(T\in\mathcal{S}\).

### (a) \(G_c(x)=c\,x^2\bmod N\) with \(c^{2^{T}-1}=1\) for all trained \(T\)
**Theorem.** \(G_c^{\circ k}(x)=c^{2^k-1}x^{2^k}\bmod N\) (on units). So \(G_c^{\circ T}=f^{\circ T}\) on units iff \(c^{2^T-1}=1\). For \(\mathcal{S}=\{2,4,\ldots\}\), \(\gcd_T(2^T-1)=2^{\gcd(T:T\in\mathcal{S})}-1\); e.g. \(\{2,4,8\}\Rightarrow\gcd(3,15,255)=3\), so any \(c\) with \(c^3=1\) works on those depths.

**Vs network:** **Assumption:** SquareMul is hardwired as mul(operand,operand) with **no** post-scale channel. Then the realizable exact maps are \(\{f\}\), not \(\{G_c:c\neq 1\}\). Soft cells can still *approximate* a scale—**Occam bet**, not a proof. Finite batch equality ≠ polynomial identity over \(\mathbb{Z}/N\mathbb{Z}\).

### (b) \(g(x)=x^{-2}\) on units (matches all even composition counts)
**Theorem.** On \((\mathbb{Z}/N\mathbb{Z})^\times\), \(x\mapsto x^{-1}\) commutes with \(f\), and \(g=f\circ(\cdot)^{-1}\) satisfies \(g^{\circ 2^k}=f^{\circ 2^k}\).

**Vs network:** Inversion is **not** in the SquareMul→Reduce dataflow. Excluded at the structural level unless the learned cell invents an inverse subroutine (possible in principle for a free RNN; **not** the intended critical path). Without \(T=1\) labels, a *free* digit program can still land on \(g\); **this** hypothesis class makes \(g\) a non-default, not impossible for soft nets.

### (c) Permutation roots: any \(h\) with \(h^{\circ T}=f^{\circ T}\) for all \(T\ge 2\) in \(\mathcal{S}\)
**Theorem.** The set of such \(h\) is typically huge in \(\mathrm{Sym}(R_N)\); even-depth observations do not pick a unique one-step map.

**Vs network:** Exact structural mul-self-mod is a **measure-zero** subset of that set. Architecture excludes almost all roots; it does **not** prove SGD finds \(f\).

### If training lacks \(T=1\)
| Family | Structural mul-mod | Free digit cell |
|--------|--------------------|-----------------|
| (a) \(G_c\) | Excluded if no scale path; else Occam | Not excluded |
| (b) \(x^{-2}\) | Structurally off-manifold | **Not excluded** — recovery of \(f\) **impossible** as ID; only Occam |
| (c) perm roots | Almost all excluded | **Not excluded** — ID **impossible** |

**Verdict:** On E3/E4-style \(\mathcal{S}=\{2\}\), **do not claim unique recovery**. Use **E1/E5/Medium splits that include \(T=1\)** as the only legal ID constraint. Hard without known \(T=1\) mass = Occam + transfer bet (**speculation**).

---

## 3. Six-experiment remote-GPU ladder

Primary metrics: **exact-match @ \(T=1\) seen-\(N\)** and **OOD-\(N\)** (example-level). Secondary: token CE, updates/sec. No Mac GPU; Modal/H100 or equivalent remote.

| # | One change | Tier/data | Wall clock | Save | Promote / abandon | Predicted |
|---|------------|-----------|------------|------|-------------------|-----------|
| **E0** | Smoke: parse→head only (no mul/reduce) | Easy **e1** (N=323, T has 1) | 60s | `metrics.jsonl`, CE curve, exact@T=1 | Abandon if exact@T=1 \(>0.01\) (leak/bypass). Promote if CE finite, exact≈0 | exact≈0 (**control**) |
| **E1** | Add SquareMul+Reduce; **no** prompt→logit bypass | Easy **e1** | 60s | CE, exact@T=1, exact@T=2, update count, 5 sample decode dumps | Promote if exact@T=1 \(\ge 0.05\); abandon if exact=0 and CE within 0.05 of E0 | exact@T=1 small but >0 (**emp. pred.**) |
| **E2** | Same model; only **loss** = sequence-mean CE (confirm no aux) | Easy **e1** | 60s | same | Must match E1 ±noise; else loss bug | ≈E1 |
| **E3** | Unseen \(N\) in-tier: Easy **e5** (10–11 bit N, T∈{1,2,3}) | Easy **e5** | 60s | seen-family vs OOD-N MaxT diagnostics; exact@T=1 both | Promote if OOD-N exact@T=1 \(\ge 0.5\times\) seen; abandon if seen≥0.2 and OOD=0 (memorize N) | weak OOD (**emp. pred.**) |
| **E4** | Scale compute only | Medium **m**\* with **T=1 in mix** (pick m-set that includes T=1; if none, recreate local split from public generator with T=1 held in train—**legal local**, not Hard) | 600s | exact@T=1 vs walltime; LR sweep 1 curve | Promote if exact@T=1 \(\ge 0.9\) seen and \(\ge 0.5\) OOD-N; abandon if plateau \(<0.01\) after ≥3 LR/batch tries | exact rises with time if E1>0 |
| **E5** | Enable tied re-apply of **same** SquareMul+Reduce \(T\) times (parse T); train+eval | Medium best from E4, then Hard **h1** only if Medium T=1 gates pass | 600s then 3600s Hard | Max T ladder, OOD-N Max T, exact@T=1 always | Hard submit iff Medium exact@T=1 seen\(\ge 0.99\) and OOD-N\(\ge 0.9\); abandon recurrence if T=1 drops when loops enabled | T=1 holds; T>1 partial (**speculation** on Hard) |

Artifacts each run: `submission.py` hash, `one-layer metrics` JSONL, exact@T=1 seen/OOD scalars, CE, steps, batch size, seed.

---

## 4. Training schedule & loss (legal only)

**Available:** `input_ids`, mask, `logits`, `labels`, `valid_mask` (and `aux` if returned). **Forbidden here:** intermediate \(x_t\), reduce-only targets, oracle numeric consistency, participant `backward`, augmentation, data inspection.

**Loss (only):**
\[
\mathcal{L}=\frac{1}{B'}\sum_{b:\,n_b>0}\frac{1}{n_b}\sum_{i}\mathrm{CE}(\mathrm{logits}_{b,i},\mathrm{labels}_{b,i})\,[\mathrm{valid}_{b,i}]
\]
with \(n_b=\sum_i\mathrm{valid}_{b,i}\), \(B'=\#\{n_b>0\}\).
Optional legal sharpener: the same CE is the only term—**no** entropy/consistency add-ons under this brief.

**Schedule:**
- Always full SquareMul→Reduce→Head (no labeled phases).
- `model.train()`: one application when parsed \(T=1\); for \(T>1\) either (i) still one application until Medium T=1 gate, or (ii) \(T\) applications of the **same** block—**only after** E4 promote. Prefer (i) until gate.
- `model.eval()`: apply block \(\max(1,T_{\mathrm{parsed}})\) times for the recurrence path.
- AdamW lr \(\sim 10^{-3}\), batch as large as 48GB/H100 allows without OOM, weight decay light.
- Clock: spend Easy/Medium on T=1 exact; do not burn Hard until E4 gates.

**Assumption:** labels for \(T=1\) examples appear in the chosen public sets (E1/E5). If a split omits \(T=1\), skip it for ID work.

---

## 5. Strongest legality risk + organizer question

**Risk (Rule 7):** fixed schoolbook/Horner **control flow** that is exactly “square then mod” may be judged a hard-coded algorithm with learned carries.

**Exact question to organizers/Discord:**
*“Is a forward pass legal if it uses fixed nested loops over digit indices implementing multiply-of-the-same-operand then modulus reduction, where every digit write/carry is a randomly initialized learned cell updated by training, with no integer modular arithmetic and unbroken autograd—or is any fixed multiply-then-reduce dataflow disallowed regardless of learned cells?”*

Ship free digit-serial cell (no forced mul-self) only if they say disallowed.

---

## 6. First file / omit / expected failure

**In first `submission.py`:** Parse + SquareMul + ReduceMod + Y-only head; AdamW; sequence CE `token_training_loss`; `batch_size` large; `aux=None`; `assert_model_state`; **single** apply (ignore deep T). Target Easy **e1** then **e5**.

**Omit:** recurrence, ACT, custom multi-backward, aux losses, schedulers beyond AdamW, wide Transformer trunk, anything >~few M params until T=1 moves.

**Expected failure mode:** CE drifts down (~2.x→lower) while **exact@T=1 stays ~0** because soft digits never peak jointly (same syndrome as 0/768). Second failure: seen-N digits memorize, OOD-N exact=0 on e5. Third: enabling loops before T=1 gate **destroys** the weak T=1 signal.

---

## 7. Unasked roundtable question

**What is the train-time distribution of \(T\) on Hard—and does Hard training include a non-negligible \(T=1\) mass?**
Without that, even a perfect Easy T=1 mechanism can face an unidentifiable or shifted one-step map under the Hard caveat (“do not assume repeated squaring”), and our ID theorems do not transfer.

---

**Labels recap:** §2 theorems on \(G_c\) / inverse / roots = **theorem**; structural exclusion = **assumption** on parameterization; ladder outcomes = **empirical prediction**; Hard transfer / soft-cell loopholes = **speculation**.

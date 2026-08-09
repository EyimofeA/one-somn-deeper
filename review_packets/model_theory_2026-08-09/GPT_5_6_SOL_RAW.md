# Main conclusion

The public repeated-squaring task has a learnable \(T=1\) target, but **a one-step map is not identifiable from final labels at depths \(T\ge 2\), even if every such depth and every input were observed**. A tied recurrent model and a canonical digit state do not fix this.

Also, the supplied `.com` domain did not resolve; the repository and public materials point to [onelayerdeeper.ai](https://onelayerdeeper.ai/). Crucially, [the public site warns that Hard may change the recurrence](https://onelayerdeeper.ai/problem). Thus the cited Hard result is evidence that that submission did not solve H1’s \(T=1\), but not necessarily evidence about modular squaring.

## 1. Verified contract

### Verified facts

- **Public recurrence:**
  \[
  x_0=x\bmod N,\qquad x_t=x_{t-1}^2\bmod N,\qquad
  y=x_T=x^{2^T}\bmod N.
  \]
  Here \(N=pq\) is a product of distinct primes whose factors are withheld. Public generation samples \(1\le x<N\) with \(\gcd(x,N)=1\).
  Sources: [problem page](https://onelayerdeeper.ai/problem), [generator](https://github.com/tilde-research/one-layer-deeper/blob/main/data/squaring_mod.py).

- **Actual public Easy/Medium input:** most-significant-first decimal tokens
  \[
  [\mathtt N,\operatorname{digits}(N),\mathtt X,\operatorname{digits}(x),
    \mathtt T,\operatorname{digits}(T)].
  \]
  Despite the website’s conceptual `ANS` illustration, public Easy/Medium use separate prompt/output tensors: no answer or intermediate state is in the input.

- **Output/supervision:** only \(\operatorname{digits}(x_T)\). No \(p,q\), \(x_1,\ldots,x_{T-1}\), quotient, carry, or scratchpad labels reach participant code. Digits are tail-aligned: the evaluator gathers logits from the final `target_length` prompt positions. There is no scored EOS in this representation.
  Sources: [tokenizer/collator](https://github.com/tilde-research/one-layer-deeper/blob/main/data/squaring_mod.py), [runner](https://github.com/tilde-research/one-layer-deeper/blob/main/benchmark/runner.py), [dataset wrapper](https://github.com/tilde-research/one-layer-deeper/blob/main/data/counting.py).

- **Model interface:** `forward(input_ids, attention_mask)` returns
  \[
  \text{logits}\in\mathbb R^{B\times S\times 17}
  \]
  and an arbitrary auxiliary object. A custom `TokenLossBatch` retains sequence boundaries, labels, masks, target positions, and auxiliary state, but not raw input IDs unless the model places derived information in its auxiliary output.

- **Exact scoring:** an example is correct only if every target digit’s argmax is correct. Certification uses \(T=1,2,4,8,16,32,64\), consecutively from \(T=1\).

- **Public training depths:**
  - Easy: E1 \(1,2,3\); E2 \(1,2,4\); E3 \(2\); E4 \(2\); E5 \(1,2,3\).
  - Medium: M1/M2 \(4,8,16\); M3 \(2\); M4 \(8\); M5 \(2,4,8\).

  Thus **all Medium training depths are even and exclude \(T=1\)**.
  Source: [generation script](https://github.com/tilde-research/one-layer-deeper/blob/main/scripts/generate_datasets.sh).

- **Budgets/rules:** Easy 60 s, Medium 600 s, Hard 3600 s on one H100; evaluation gets half the training allowance. One self-contained file, at most 500M persistent state elements, random trainable initialization, GPU-only, evaluator-owned backward/updates, no data augmentation, task solver, hard-coded forward algorithm, pretrained lookup, CPU offload, or broken gradient path. Learned recurrence and custom differentiable losses are allowed.
  Sources: [README rules](https://github.com/tilde-research/one-layer-deeper#rules), [submission page](https://onelayerdeeper.ai/submit), [public API](https://github.com/tilde-research/one-layer-deeper/blob/main/benchmark/api.py).

- **Hard:** H1’s data and recurrence are private; the site explicitly says not to assume repeated squaring.

### Deductions

1. A row with \(T=1\) supplies direct one-step supervision. A row with \(T>1\) supplies only a composition constraint.
2. Public E5 is the clean first test of unseen-\(N\) one-step generalization: variable 10–11-bit training moduli and 12–13-bit OOD-\(N\) \(T=1\) profiles.
3. The reported Hard CE \(2.17846\) gives geometric-mean true-token probability
   \[
   e^{-2.17846}\approx 0.113.
   \]
   This is only modestly better than uniform over ten digits, \(\log 10=2.3026\); zero exact rows is unsurprising. It falsifies that trained realization, not the existence of a learnable mechanism.

### Assumption used below

The mathematical analysis targets the **public repeated-squaring task**. It does not assert that H1’s hidden forward step is squaring.

---

## 2. Identifiability

Let \(U_N=(\mathbb Z/N\mathbb Z)^\times\) and \(f(x)=x^2\).

### Theorem 1: even-depth labels admit a global alternative algorithm

Let \(\lambda(N)\) be the Carmichael exponent. For any \(a\) satisfying
\[
a^d\equiv1\pmod{\lambda(N)},
\]
define
\[
g_a(x)=x^{2a}\pmod N.
\]
Then
\[
g_a^{\,t}(x)=x^{(2a)^t}
           =x^{2^t a^t}
           =x^{2^t}=f^t(x)
\]
whenever \(d\mid t\).

In particular, take \(a=-1,d=2\):
\[
g(x)=x^{-2}\pmod N.
\]
Then \(g^t=f^t\) for every even \(t\), but generally \(g\ne f\). For example, modulo \(323\),
\[
f(2)=4,\qquad g(2)=4^{-1}=81.
\]

Therefore all public Medium final labels are compatible with both the forward squaring algorithm and inverse-squaring—even across unseen \(N\). This is not a lookup-table ambiguity.

### Theorem 2: even observing every \(T\ge2\) need not identify \(T=1\)

Write
\[
I_1=f(U_N),\qquad I_2=f^2(U_N).
\]
Let \(\pi\) be any permutation of \(U_N\) such that:

1. \(\pi\) fixes \(I_2\) pointwise;
2. for every \(y\in I_1\), \(f(\pi(y))=f(y)\).

Define
\[
g_\pi=\pi\circ f.
\]
Then
\[
g_\pi^2(x)
=\pi f\pi f(x)
=\pi f^2(x)
=f^2(x).
\]
If \(g_\pi^t=f^t\) for \(t\ge2\), then
\[
g_\pi^{t+1}(x)=\pi f^{t+1}(x)=f^{t+1}(x),
\]
because \(f^{t+1}(x)\in I_2\). Hence
\[
\boxed{g_\pi^t=f^t\quad\text{for every }t\ge2}
\]
although \(g_\pi\) may differ at \(T=1\).

#### Explicit public-modulus example

For \(N=323=17\cdot19\):

- \(58^2=134\pmod{323}\);
- \(96^2=172\pmod{323}\);
- \(134^2=172^2=191\pmod{323}\);
- \(134,172\notin I_2\): modulo \(17\), they are \(15,2\), while the nonzero fourth powers are \(\{1,4,13,16\}\).

Let \(\pi\) swap \(134\) and \(172\) and fix everything else. Then
\[
g_\pi(58)=172\ne134=f(58),
\]
yet \(g_\pi^t=f^t\) for all \(t\ge2\) and all \(x\in U_{323}\).

### Consequence

If training excludes \(T=1\), final labels do **not** identify the forward step in the unrestricted hypothesis class—even if odd depths such as \(T=3\) are present.

The exact condition needed is that
\[
\Phi_S:\mathcal H\to\prod_{t\in S}\mathrm{Map}(U_N,U_N),
\qquad
h\mapsto(h^t)_{t\in S}
\]
be injective on the legal model class \(\mathcal H\). A generic neural recurrent class is not known to have this property.

The clean legal resolution is supplied \(T=1\) final labels. If H1 omits them, participant-generated one-step labels would be prohibited augmentation/a math oracle. Canonical states and semigroup consistency do not resolve the theorem above.

---

## 3. Minimum learnable \(T=1\) mechanism

For \(0\le x<N\), the answer is uniquely characterized by
\[
x^2=qN+y,\qquad 0\le y<N,\qquad 0\le q<N.
\]
Uniqueness follows because two solutions would satisfy
\[
(q-q')N=y'-y,
\]
whose right side has magnitude \(<N\), forcing \(q=q'\) and \(y=y'\).

A successful length-generalizing model must therefore learn computation functionally equivalent to:

1. decimal place and digit identity;
2. digit products/addition and exact carry propagation;
3. quotient/reduction, or another exact circuit yielding the same remainder;
4. a controller that applies these rules uniformly across positions and unseen \(N\).

It must **learn** these transitions. Computing \(x^2\), `% N`, multiplication tables, or the witness equation inside the fixed forward/loss would be the forbidden solver.

### Necessary state capacity

For odd distinct primes, squaring on \(U_N\) has a four-element kernel, so
\[
|f(U_N)|=\frac{\varphi(N)}4.
\]
A discrete state capable of emitting every possible \(T=1\) answer therefore needs at least
\[
\log_2\varphi(N)-2=\Theta(\log N)
\]
bits. A position-preserving \(L\)-digit register provides this naturally. A single unrestricted continuous vector has formal capacity, but supplies neither exact decimal semantics nor a robust OOD algorithm.

### Minimum inductive bias

The minimum defensible mechanism is:

\[
S_\theta:\bigl(\text{canonical digits of }N,\text{canonical digits of }x\bigr)
\longmapsto \text{canonical digits of }y,
\]

with:

- weights shared across digit positions and input lengths;
- \(S_\theta\) blind to \(T\);
- no modulus-specific persistent table;
- enough tied serial workspace for carry/reduction;
- direct final-label CE at \(T=1\).

An ordinary Transformer token head can represent this, but nothing forces it to discover this transducer rather than positional correlations or memorization. Per-token CE also tolerates “nearly right” strings while exact scoring does not.

---

## 4. One concrete legal architecture

### Categorical recurrent tape

Let \(S=\texttt{spec.max\_seq\_len}\), vocabulary size \(V=17\).

1. **Interface router:** on GPU, use the public marker tokens to right-align \(N\) and \(x\) into read-only categorical registers. This parses syntax only; it performs no arithmetic.
2. **State:**
   \[
   P_k\in\Delta^{B\times R\times S\times V},\qquad R=6.
   \]
   Two registers hold read-only \(N,x\); one is the result; three are generic workspace.
3. **Shared update:** a small radius-2 convolution/GLU cell \(U_\theta\), with cross-register mixing and boundary/register embeddings, is reused for
   \[
   K=8S
   \]
   microsteps:
   \[
   z_{k+1}=\log(P_k+\varepsilon)+U_\theta(P_k,N,x),\qquad
   P_{k+1}=C_{\tau,\sigma}(z_{k+1}).
   \]
   All trainable weights are randomly initialized. There are no fixed arithmetic tables or operation-specific stages.
4. **Direct readout:** result-register logits are routed to right-relative output positions. No separate latent result head can disagree with the register.
5. **Size:** width 96–128 gives roughly \(0.5\)–\(2\)M parameters; larger is unjustified until this fails by capacity rather than optimization.

### Preventing a soft hidden code

There is an unavoidable trade-off:

> A map into a finite categorical set is locally constant almost everywhere, hence has zero ordinary pathwise derivative almost everywhere.

Therefore strict hard states and ordinary nonzero backpropagation cannot coexist without a relaxation, score-function estimator, or straight-through approximation. Entropy regularization alone is not a proof: a soft simplex remains uncountable.

Use the following compromise:

\[
C_{\tau,\sigma}(z)
=\operatorname{softmax}\!\left(\frac{z+\sigma G}{\tau}\right),
\]
with GPU Gumbel/logistic noise.

- Run two independently perturbed trajectories in one evaluator-owned forward.
- Apply final-label loss to both and a final-state Jensen–Shannon consistency penalty.
- Anneal \(\tau\) toward \(0.1\); increase the entropy penalty only after label learning begins.
- In evaluation, use hard argmax after every tape update.
- Require 100% soft/hard agreement before promotion.
- For later recurrence, discard all workspace at each macrostep: only the canonical result digits may cross the boundary.

Noise destroys low-amplitude side channels; hard evaluation proves whether the learned computation survives their removal. It does not mathematically guarantee semantic intermediate scratch symbols, but it prevents continuous hidden information from carrying the recurrence.

### Loss and curriculum

Using only supplied final labels:
\[
\mathcal L_{\rm label}
=\frac1B\sum_i\frac1{d_i}\sum_{j\in\text{valid}(i)}
-\log p_{ij}(y_{ij}),
\]
\[
\mathcal L
=\mathcal L_{\rm label}^{(1)}
+\mathcal L_{\rm label}^{(2)}
+\lambda_{\rm JS}\operatorname{JS}(P^{(1)},P^{(2)})
+\lambda_H\overline{H(P)}.
\]

No arithmetic consistency loss.

Optimization: ordinary fused AdamW, one backward pass, FP32 state softmax/logits, BF16 cell operations, evaluator clipping. The failed 163k-update run says to change the representation/gradient path, not add parameters.

### Promotion gates

1. **E1 sanity:** hard-state \(T=1\) reaches 100% on the fixed-\(N\) profile.
2. **E5 decisive gate:** train only/primarily on supplied \(T=1\) rows; require exactly \(512/512\) seen-\(N\) and \(512/512\) OOD-\(N\) \(T=1\), with identical noisy-soft and hard outputs.
3. **M5 identifiability stress:** require \(768/768\) seen-\(N\) and \(768/768\) OOD-\(N\) \(T=1\), despite training labels being at even depths only. Failure here means the architecture’s simplicity bias did not select the intended root.

Do not spend a Hard attempt before these gates.

---

## 5. Only after \(T=1\): composition

Carry only canonical residue digits:

\[
r_0=\operatorname{onehot}(x),\qquad
r_{t+1}=Q\!\left(S_\theta(N,r_t)\right),
\]
where \(Q\) is the noisy-soft training channel and hard categorical evaluation channel. The step never sees \(T\); \(T\) controls only how many times it is called.

With final loss only, the gradient is
\[
\frac{\partial\mathcal L}{\partial\theta}
=
\sum_{t=0}^{T-1}
\frac{\partial\mathcal L}{\partial r_T}
\left(\prod_{s=t+1}^{T-1}
\frac{\partial r_{s+1}}{\partial r_s}\right)
\frac{\partial r_{t+1}}{\partial\theta}.
\]
This is precisely the vanishing/exploding final-label problem. \(T=1\) removes the Jacobian product and is therefore the correct first target.

Additional failures:

- **State-distribution shift:** training \(x\) is sampled from all units, while \(r_t\) moves through squares, fourth powers, etc.
- **Compounding errors:** if conditional one-step failure is at most \(\epsilon\), the union bound gives \(T\epsilon\). With 768 examples and \(T=64\), even \(\epsilon\approx2\times10^{-5}\) predicts about one bad trajectory.
- **Soft/hard exposure mismatch.**
- **T-specific shortcuts** if the step sees \(T\) or retains hidden workspace.
- **Runtime:** compute becomes \(O(TK)\).
- **Cycles/collisions:** an incorrect transition can rejoin a correct orbit, hiding one-step defects at larger \(T\).
- **H1 transfer:** the hidden recurrence may not be squaring.

---

## 6. Red team

- **Strongest rule objection:** deterministic marker routing, categorical digit identities, repeated learned-step control, or T-based loss masking could be judged task-specific forward structure. Mitigation: request organizer preclearance; emphasize that the cell contains no square, multiply, remainder, quotient, carry table, or arithmetic loss. If rejected, replace deterministic routing with learned marker attention.

- **Strongest mathematical counterexample:** \(g(x)=x^{-2}\) matches every even training depth globally, while the \(N=323\) permutation construction matches every depth \(T\ge2\). The proposed canonical recurrent state does not remove this ambiguity.

- **Fastest falsification:** one local E5 evaluator run, T1-weighted, capped at 600 seconds—ten times the Easy allowance. Examine only evaluator metrics, not examples. If hard-state OOD-\(N\) \(T=1\) remains below 100%, the model did not learn a general step.

- **Abandon condition:** abandon this architecture if it fits soft/train \(T=1\) but fails hard-state or OOD-\(N\) \(T=1\) across three initializations, or if organizers reject the generic tape as hard-coded. Abandon one-step recovery entirely if H1 excludes T1 and supplies no hypothesis restriction making \(\Phi_S\) injective.

## 7. Ranked recommendation

1. **Build only the E5 \(T=1\) categorical-step experiment.** No outer recurrence, no arithmetic auxiliary loss, no Hard submission until exact OOD-\(N\) success.
2. **Use M5 \(T=1\) as the root-selection test and seek organizer preclearance.** Failure is expected under the identifiability theorem and should stop the project.
3. **Only after both gates, compose the same T-blind step and test \(T=2,4,8\).** Promote by exact hard-state accuracy, never by CE.

**Bold alternative:** if soft-state leakage—not missing \(T=1\) information—is the blocker, use a genuinely hard sampled categorical tape with a score-function/RELAX gradient estimator, after explicit organizer approval. It removes the hidden continuous channel at the cost of very high final-label gradient variance; it cannot solve the information-theoretic absence of \(T=1\) supervision.

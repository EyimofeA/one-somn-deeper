# Independent model roundtable: T=1 before recurrence

Date: 2026-08-09

## Bottom line

The current blocker is not long-horizon recurrence. The submitted model did not learn a usable one-step map: CE 2.17846 corresponds to a mean true-token probability of about 0.113, so zero exact multi-digit rows in 768 is expected.

There is also a separate information problem. If training omits T=1, final labels generally do not identify the one-step map, even with tied weights and canonical states. A scratchpad can bias SGD toward a simple root, but it cannot manufacture missing information. Therefore:

1. solve supplied T=1 labels on seen and unseen N first;
2. use a generic canonical register bottleneck with no prompt-to-logit bypass;
3. require exact hard-state T=1 promotion gates before recurrence or Hard;
4. do not treat a structural multiplication/division skeleton as legal until organizers preclear it.

Hard may change the recurrence, so the modular-squaring conclusions below are a public-task research scaffold, not a claim about the private H1 transition.

## Runs and metered cost

| Label | Provider route | Reasoning | Calls | Recorded tokens | Pi-recorded cost | Status |
|---|---|---:|---:|---:|---:|---|
| Kimi K3 | Cursor | high | 3 | 121,772 | $0.000000 | complete |
| GPT-5.6 Sol | OpenAI Codex | max | 14 | 1,045,587 including 841,728 cache reads | $2.871584 | complete |
| Claude Fable 5 | Anthropic API | high | 9 | 437,568 including cache traffic | $4.740102 | complete |
| Grok 4.5 Slow | Cursor | high | 19 | 784,259 reported total | $0.000000 | complete |

Total Pi-recorded cost for both rounds: **$7.611686**. Cursor's zero means Pi recorded no per-token charge on that subscription route; it does not prove the Cursor subscription itself is free.

## Independently checked mathematical results

### 1. Scalar gauge for selected depths

Kimi gave the clean family

\[
G_c(x)=c x^2 \pmod N,
\qquad
G_c^{\circ T}(x)=c^{2^T-1}x^{2^T}\pmod N.
\]

For training depths \(D\), let

\[
g=\gcd_{T\in D}(2^T-1).
\]

Every \(c\) satisfying \(c^g=1\pmod N\) produces exactly the same final labels at all depths in \(D\), but can disagree at T=1. For M1 depths \(\{4,8,16\}\), \(g=15\). At \(N=10403=101\cdot103\), there are

\[
\gcd(15,100)\gcd(15,102)=5\cdot3=15
\]

such roots, hence fourteen wrong non-identity scalar roots in addition to the true map.

### 2. A global alternative at every even depth

Sol, Grok, and Fable independently found the stronger cross-modulus alternative on the unit group:

\[
h(x)=x^{-2}\pmod N,
\qquad
h^{\circ T}(x)=x^{(-2)^T}\pmod N.
\]

For every even T, \((-2)^T=2^T\), so this agrees with repeated squaring for every generated input, every modulus, and every even depth while generally failing T=1. This is an algorithmic ambiguity, not a finite lookup-table trick.

### 3. Even all depths T>=2 may not identify T=1

Sol supplied the strongest construction. Let \(f(x)=x^2\), \(I_1=f(U_N)\), and \(I_2=f^2(U_N)\). Choose a permutation \(\pi\) that fixes \(I_2\) and swaps elements in the same f-fiber inside \(I_1\setminus I_2\). Then

\[
g=\pi\circ f
\]

can differ from f at one step while satisfying \(g^{\circ T}=f^{\circ T}\) for every \(T\ge2\). The explicit N=323 example swaps 134 and 172: both square to 191, both lie outside the fourth-power image, and \(g(58)=172\ne134=f(58)\).

This refutes the tempting claim that adding an arbitrary odd depth greater than one necessarily fixes the root. Direct T=1 labels do fix it.

## What a defensible T=1 learner minimally needs

These are mechanisms, not evidence that the current optimizer can discover them:

- a positional digit or limb representation shared across lengths and moduli;
- learned pair interactions capable of forming a product-like intermediate;
- learned local normalization capable of propagating carry/borrow information;
- learned modulus-conditioned reduction;
- a canonical residue register that is both the only output source and the only state allowed across recurrence steps;
- no direct prompt-to-logit residual bypass;
- direct supplied T=1 final-label supervision, especially across unseen N;
- a soft-to-hard schedule whose evaluation path rounds every recurrent boundary, plus an exact soft/hard agreement gate.

The safest legal architecture remains a **generic tied local register machine**. It may have read-only N/x registers, mutable categorical work registers, a generic shared local cell, and a fixed amount of parameter-free routing. It should not expose named multiply, carry, comparator, quotient, or subtraction stages unless Rule 7 preclearance is obtained.

## Model-by-model judgment

### Kimi K3

Best contribution: the scalar-gauge theorem and the observation that same-x cross-depth labels do not remove it. Its digit scratchpad is directionally useful but does not itself prove root selection or unseen-N arithmetic.

### GPT-5.6 Sol

Best overall theory. The permutation construction shows the ambiguity is larger than the scalar family and can survive all depths T>=2. Its categorical tape recommendation is legally safer than a wired arithmetic circuit, but noisy soft states plus entropy are not a proof against hidden codes. The proposed 100% E5 and M5 T=1 gates are appropriately strict.

### Grok 4.5 Slow

Useful independent confirmation of inverse-squaring. Reject or preclear its schoolbook/Horner skeleton. Its proposed reduce-only curriculum lacks supplied reduce-only labels, and arithmetic self-check losses risk becoming prohibited solver structure.

### Claude Fable 5

Most ambitious architecture, but its claimed uniqueness theorem is not established for the proposed network. A learned digit table plus carry/reduction cells is not automatically a single bounded quadratic integer polynomial. Finite evaluator equality is not polynomial identity. Its FLOP/update estimate is also inconsistent: 10^7-10^8 FLOPs per example cannot yield more than 10^5 batch-512 updates in one H100-hour. Treat the theorem as a hypothesis, not a result. The wired multiplication and long-division routing has the highest Rule 7 risk of the panel.

## Prompting method that worked

The useful elicitation pattern was:

1. clean context, public sources only;
2. force contract verification before solutioning;
3. ask for an explicit alternative map and proof, not vague composition-root language;
4. require T=1 to be solved before recurrence is discussed;
5. require a minimum mechanism, tensor-level architecture, and exact promotion gates;
6. require the model to name its strongest legality objection and fastest falsifier;
7. run independent models without disclosing earlier answers;
8. synthesize claims by proof and rule compliance, not majority vote.

A productive second-pass prompt should now give each model the three checked ambiguity constructions and ask it to either prove that its proposed hypothesis class excludes all three or withdraw the architecture. This is better than asking for another unconstrained idea dump.

## Ranked next actions

1. Build the smallest generic canonical-register T=1 learner and test only E5 seen-N/OOD-N T=1. No outer recurrence.
2. Pre-register a hard gate: 100% exact on both profiles, hard rounding at every boundary, and no prompt-to-logit bypass. Three failed seeds park the architecture.
3. Ask organizers whether fixed multiplication-alignment and shifted-subtraction routing violate Rule 7 before implementing Fable/Grok's explicit arithmetic skeleton.
4. Only after the T=1 gate, tie the same step across T and test T=2,4,8.

## Raw outputs

- [Kimi K3](KIMI_K3_RAW.md)
- [GPT-5.6 Sol](GPT_5_6_SOL_RAW.md)
- [Claude Fable 5](CLAUDE_FABLE_5_RAW.md)
- [Grok 4.5 Slow](GROK_4_5_RAW.md)
- [Shared prompt](PROMPT.md)
- [Cross-examination synthesis](CROSS_EXAM_SYNTHESIS.md)
- Cross-examinations: [Kimi K3](KIMI_K3_CROSS_EXAM.md), [GPT-5.6 Sol](GPT_5_6_SOL_CROSS_EXAM.md), [Claude Fable 5](CLAUDE_FABLE_5_CROSS_EXAM.md), [Grok 4.5 Slow](GROK_4_5_CROSS_EXAM.md)

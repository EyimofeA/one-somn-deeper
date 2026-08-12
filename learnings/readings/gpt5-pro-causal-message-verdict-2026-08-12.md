# GPT-5 Pro verdict: causal message consumption

Source: user-provided GPT-5 Pro response, preserved in the Codex attachment
`3b994e37-fbca-40e9-ba4e-43cae327c517/pasted-text.txt` (1,200 lines). This note
stores its decision-relevant content in the repository; it does not turn the
model's judgments into experimental evidence.

## Verdict

Do not run another broad architecture, representation, optimizer, recurrence,
or GPU sweep. The next causal question is whether a small local arithmetic
message helps only when later recurrent updates **consume** it, rather than
when it is merely recoverable from the final hidden state.

The proposed experiment is CMC-T1. It compares a matched terminal-decodable
control with a causal-message arm. Both form the same learned directional
messages; only the causal arm routes received messages into future content
updates. First test raw-square capability with research-only carry supervision.
Only if that passes should the unchanged core be trained legally from final
`x^2 mod N` labels.

## Why Pro thinks this is the next question

- Direct MLPs and Transformers fit training perfectly but transfer at only
  about 4%, so ordinary optimization and capacity are not the main problem.
  Evidence: [`../experiments/2026-08-10_x2modn_direct_mlp/NOTE.md`](../../solving/experiments/2026-08-10_x2modn_direct_mlp/NOTE.md)
  and [`../experiments/2026-08-10_x2modn_direct_transformer/NOTE.md`](../../solving/experiments/2026-08-10_x2modn_direct_transformer/NOTE.md).
- Structured decimal state improves unseen-N transfer to 18.69%, suggesting
  topology/locality matter but do not identify the algorithm by themselves.
- A terminal carry target improved the Neural GPU only from 3.85% to 6.25%.
  Thus carry information may be decodable without being used during digit
  construction. Evidence:
  [`../experiments/2026-08-10_multilane_neural_gpu_square_carry_50k/NOTE.md`](../../solving/experiments/2026-08-10_multilane_neural_gpu_square_carry_50k/NOTE.md).
- The central-digit collapse is consistent with either cross-product
  accumulation failure or carry-state failure; it does not isolate carry.
- Trace-supervised arithmetic works strongly, while final-label learning does
  not discover it. The main scientific gap is therefore credit assignment and
  identifiability, not whether neural components can represent arithmetic.

## Exact experiment Pro requested

1. Recover the exact 18.69% decimal structured-tape anchor and its artifacts.
2. Build two parameter/FLOP-matched arms with identical representation,
   recurrence depth, optimizer, data order, and initialization protocol.
3. Terminal control: messages update a shadow stream visible to the final
   readout but cannot alter future content state.
4. Causal arm: neighbor messages enter the next recurrent content update.
5. Diagnostic: raw square, true carry versus terminal-only versus shuffled
   carry. Promote only at >=50% unseen exact, >=20-point causal advantage,
   >=70% on both weak middle columns, and >=70% advantage removal under message
   intervention.
6. Legal main: T=1 decimal `x^2 mod N`, final labels only, no trace or carry
   target. A valid comparison requires both arms to reach >=99.5% train exact.
7. Seed-0 promotion: >=30% unseen-N exact, >=10 points over terminal control,
   >=20% held-out-x, >=5 points over control, and >=70% of the advantage erased
   by message intervention.
8. If the diagnostic fails, test a generic second-order accumulation cell. If
   diagnostic succeeds but legal gain is <3 points, hold architecture fixed and
   test late-step final-answer supervision.

## Competition gate Pro requested

Do not spend Easy quota until the legal CMC core passes three seeds and a local
e5 mirror reaches at least 64/512 on both T=1 profiles or materially exceeds
10.5% mean exact. Hard requires near-zero error on large T=1 suites, autonomous
T=2/4/8 behavior, runtime feasibility, a Rule-7 audit, and owner approval.

## Audit of the rushed Hard translation

Hard job `580f78bc-de32-4495-a1ca-c34726331d3a` is **not CMC-T1 evidence**.
It introduces causally consumed directional messages, but it does not contain
the terminal-decodable control, matched interventions, or diagnostic gate. It
also retains global attention, late-step final-answer losses, random recurrent
depth, noise, and a decoder that mean-pools the tape. Those paths confound any
causal-message interpretation. Treat its result only as competition telemetry
for exact SHA-1 `2b1d03547e064639cc914c9cbe6f529c8aec24a2`.

## Questions still blocking the proper experiment

1. Is research commit `9d40cf0...` still the authoritative evidence state?
2. Can we recover exact source/config/split/logs/checkpoint for the 18.69%
   anchor and both 50k carry controls?
3. Has the organizer ruled on generic message lanes under Rule 7?


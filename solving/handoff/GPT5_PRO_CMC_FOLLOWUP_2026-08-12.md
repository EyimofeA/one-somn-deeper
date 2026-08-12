# Follow-up prompt for GPT-5 Pro

You previously recommended CMC-T1: a matched comparison between a terminally
decodable message stream and one consumed by future recurrent content updates.
Your complete verdict is summarized in
`learnings/readings/gpt5-pro-causal-message-verdict-2026-08-12.md`.

We then made an owner-requested, rushed Hard submission. Read the exact source:

`solving/experiments/2026-08-12_causal_message_hard/submission.py`

Exact SHA-1: `2b1d03547e064639cc914c9cbe6f529c8aec24a2`
Hard job: `580f78bc-de32-4495-a1ca-c34726331d3a`

Do not treat this upload as evidence for CMC-T1. Audit it against your proposed
experiment and answer these questions:

1. Identify every way the source fails to instantiate a controlled CMC-T1
   comparison. Pay special attention to global-attention bypasses, tape layout,
   position-specific decoding, late losses, recurrence depth, and whether T is
   executed or merely embedded.
2. Give the smallest exact patch plan for a research-only diagnostic with three
   arms: terminal-only true-carry, causal true-carry, and causal shuffled-carry.
   Change no unrelated variable.
3. Specify tensor shapes and pseudocode for immutable digit inputs, content
   state, directional messages, shadow state, received-message routing, and
   position-specific decoding. Do not encode multiplication, carry, division,
   or a fixed phase schedule in the forward pass.
4. Define CPU smoke tests that prove gradients from a later digit reach the
   appropriate message writer and receiver, while the terminal control blocks
   that causal path.
5. Reassess your gates. If the 18.69% anchor or its exact checkpoint cannot be
   recovered, state whether to reconstruct it, choose another frozen anchor, or
   stop—and why.
6. Produce a one-page experiment card with one changed variable, predicted
   curves, falsifiers, required plots, seed stopping rules, estimated L40 hours,
   and a strict no-submission gate.

Separate verified repository facts, mathematical reasoning, and hypotheses.
Do not recommend GPU rental or competition submission until the provenance and
CPU gates pass.

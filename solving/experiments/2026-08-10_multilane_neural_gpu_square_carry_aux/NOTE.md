# Multi-lane Neural GPU: carry-supervision intervention

Status: complete; 12,000-step intervention failed.

Parent: [`../2026-08-10_multilane_neural_gpu_square/NOTE.md`](../2026-08-10_multilane_neural_gpu_square/NOTE.md).

The parent reached 4.0000% unseen-x exact and failed only the central output
columns. This card freezes its data split, local-grid backbone, six lanes,
width 64, 16 tied microsteps, digit loss, AdamW settings, batch size, update
budget, and seed. The sole intervention is a 130-parameter linear head trained
with weight-1 MSE to predict normalized schoolbook carry-in and carry-out at
each output column. The targets affect training only and are not input to the
forward recurrence or digit decoder.

If missing carry credit assignment is the main blocker, unseen-x exact reaches
at least 50% and both central digit positions reach at least 70%. Strong
confirmation is at least 70% exact, comparable to the prior Transformer carry
intervention. Kill below 20% unseen exact or below 50% at either central
position. A positive result requires a same-grid shuffled-carry control before
claiming that carry semantics, rather than the auxiliary head/loss, caused it.

## Result

At 12,000 updates, carry supervision reached **13.2500% train exact** and
**4.5500% unseen-x exact**, versus the answer-only parent's 12.9125% and
4.0000%. Unseen central-position accuracy changed from 29.05%/15.20% to
28.00%/16.45%; neither central gate passed. Normalized unseen carry MSE was
0.02708, showing that carry was linearly decodable from the final state without
being used effectively by the digit computation. Both preregistered kill
conditions fired.

This rejects terminal-state carry supervision at the parent's 12,000-step
budget. It does not yet reject a delayed-learning effect: the prior Transformer
carry intervention was only 2.77% exact at step 12,000 and 64.51% at step
50,000. A matched answer-only/carry 50,000-step pair is therefore registered as
the next and final duration check.

The run was copied from Prime pod `a6eb7c97e54d4174a9b265674758a383` to
the ignored local Prime artifact root. The lifecycle helper verified seven
files and 404,480 bytes on both hosts, including source and config snapshots.
The pod remains active.

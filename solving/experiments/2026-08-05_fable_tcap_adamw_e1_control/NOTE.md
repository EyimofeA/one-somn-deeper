# Fable T-cap/AdamW e1 baseline control

Author: Codex

This is a selection control, not a new architecture. It runs the existing
standalone source `solving/submissions/fable_tcap_adamw/submission.py` unchanged
through the current e1 evaluator and the standard run-artifact wrapper.

Question: does its training-time random effective-depth exposure generalize
better than the current Fable v2 control? Promotion requires held-out mean
exact above the hosted 1.00% Easy reference; otherwise no hosted attempt is
justified.

Prediction is registered in `solving/experiments/predictions.md`. Metrics and
copied source remain outside Git at `runs/fable_tcap_adamw_e1_control/`.

## Result

Confirmed. On the L40 current e1 evaluator, the unchanged source completes
941 updates in 60.02 seconds. Test exact is 0.67% (1/150), OOD exact is 8.00%
(8/100), and mean exact is 4.33%. It has no certified T=1 or OOD-N T=1 rung.
This clears the 1.00% hosted Easy reference, so it is promoted for exactly one
Easy attempt. It is not evidence for Medium or Hard.

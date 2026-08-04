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

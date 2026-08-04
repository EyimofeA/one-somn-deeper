# Fable Muon+AdamW e1 optimizer control

Author: Codex

The existing `2026-07-22_fable_hard_h1_muon/submission.py` is evaluated
unchanged under current Easy e1. Its sole intended experimental difference from
the selected T-cap baseline is optimizer family: Muon for two-dimensional
transformation matrices and AdamW for embeddings, biases, and scales.

The run is a deadline-time local selection control. Promotion requires a local
mean clearly above the AdamW control's 4.33%; otherwise the hosted Easy result
already obtained with AdamW remains the only Easy candidate.

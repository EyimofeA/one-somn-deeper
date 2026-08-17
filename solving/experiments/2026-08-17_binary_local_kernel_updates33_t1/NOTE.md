# Local-kernel 33-update T=1 transition

This interpolates between the fast but apparently depth-limited 22-update cell
and the accurate 44-update reference without changing any weights, data, loss,
or optimizer choice.

Promote above 18% validation exact only if both unseen-N audits exceed 16%.

## Result

Promoted as the local width-256 transition.

- Parameters: 1,772,289
- Best checkpoint: step 9,500
- Train exact: 22.668%
- Validation unseen-x / seen-N exact: 19.880%
- Audit seen-x / unseen-N exact: 16.020%
- Audit unseen-x / unseen-N exact: 19.380%
- Wall time: 1,733.2 seconds, including a 91.2-second cold compile

All preregistered gates passed. Relative to 44 steps, the model gives up 2.96
validation points at the fixed example budget but saves 322.7 seconds and
preserves unseen-N behavior. Promote 33 steps for clock-aware follow-ups.

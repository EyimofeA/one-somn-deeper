# Delayed Muon cosine-decay T=1 ablation

This keeps the promoted anchor unchanged through step 6,500, then changes only
the final learning-rate trajectory: cosine decay from `0.006` to `0.001` by
step 10,000.

## Result

The selected step was 10,000: train exact `16.217%`, validation `16.82%`,
seen-x/unseen-N audit `13.42%`, and unseen-x/unseen-N audit `16.52%`. The
unchanged anchor scored `15.658%`, `16.42%`, `13.62%`, and `16.80%`.

**Do not promote.** Validation improved by 0.40 points, but both unseen-N
audits regressed by 0.20 and 0.28 points. This is at most a promising
seed-sensitive optimizer tweak, not evidence of improved modulus
generalization. It can be revisited only after a mechanism change clears the
T=1 gate by a meaningful margin.

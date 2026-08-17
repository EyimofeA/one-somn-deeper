# Exact-square isolation at promoted capacity

This changes only the source tape from binary `x` to binary `x*x`. It asks
whether the direct model's centered-quotient cliff remains when squaring is
removed from the learned computation.

This is a research diagnostic, not a legal competition submission: the
competition evaluator does not provide `x*x` as input.

## Result

The selected step was 10,000: train exact `14.191%`, validation `17.10%`,
seen-x/unseen-N audit `13.50%`, and unseen-x/unseen-N audit `17.82%`.

Supplying `x*x` materially accelerated early learning—validation was `12.96%`
at step 3,500 versus roughly `7-8%` for the fused model—but final validation
improved only 0.68 points over the `16.42%` fused anchor and stayed well below
the registered 30% squaring-bottleneck threshold.

**Conclusion:** learned squaring adds optimization drag, but reduction is the
dominant ceiling. The next architecture should spend its inductive bias and
recurrent depth on full-range reduction rather than merely widening the fused
squarer.

# Half-scale initialization T=1 ablation

This changes only the initial magnitude of every learned parameter in the
promoted width-128, 33-update local ConvGRU. It tests whether large initial
recurrent dynamics obstruct discovery of a reusable binary transition.

## Result

At the fixed 5.12M-example budget, the selected step was 10,000: train exact
`15.074%`, validation `15.74%`, seen-x/unseen-N audit `12.84%`, and
unseen-x/unseen-N audit `16.24%`. The unchanged anchor scored `15.658%`,
`16.42%`, `13.62%`, and `16.80%`, respectively.

**Reject.** Half-scale initialization improved the early validation trajectory
but missed every final accuracy anchor. Its lower reported wall time is partly
a warm compile-cache effect and is not evidence that parameter scale improves
kernel throughput. The late trajectory oscillation motivates a separate
learning-rate-decay card.

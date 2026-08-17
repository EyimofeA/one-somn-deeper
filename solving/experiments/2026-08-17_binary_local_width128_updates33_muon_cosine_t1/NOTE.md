# Muon cosine-decay T=1 ablation

This restores the promoted anchor's initialization and changes only the Muon
learning-rate trajectory: 250-step warmup to `0.006`, then cosine decay to
`0.001` at step 10,000. The aim is to retain early transition discovery while
reducing late-checkpoint instability.

## Result

The selected step was 9,500: train exact `11.294%`, validation `12.24%`,
seen-x/unseen-N audit `9.50%`, and unseen-x/unseen-N audit `11.80%`.

**Reject.** Decaying immediately after warmup reduced the useful update
magnitude during transition discovery and missed the unchanged anchor by a
large margin. The registered delayed-decay card tests only the late unstable
phase instead.

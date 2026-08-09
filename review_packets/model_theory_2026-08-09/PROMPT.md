# Shared theorem-first prompt

The independent runs were performed in empty temporary directories with context-file discovery disabled. Each model received the same core request:

> Investigate the public One Layer Deeper competition from the official site and GitHub repository. Do not inspect the local project. First verify the public contract and legality rules. Then analyze whether final labels at training depths that omit T=1 identify the one-step map, constructing and proving an explicit alternative when possible. Only after that, solve the T=1 problem x^2 mod N for unseen N: state the minimum learnable mechanism, why ordinary token regression fails, a concrete legal architecture, optimization plan, exact promotion gates, and how to prevent a soft scratchpad from carrying arbitrary hidden codes while retaining gradients. Discuss recurrence only after T=1. Red-team legality, mathematics, and trainability. Separate verified facts, deductions, assumptions, and speculation.

Evidence supplied to every model:

- one 48 GB diagnostic GPU may be assumed;
- official competition training/submission tiers may be used;
- one real Hard run used a 1.59M-parameter tied digit-register model for 3,600 seconds and 163,274 updates, ended at CE 2.17846, and scored 0/768 on both seen-N and OOD-N T=1;
- no math oracle, hard-coded forward algorithm or weights/lookup, broken autograd, participant backward, or CPU offload is permitted.

The prompt intentionally did not disclose the other models' theorems. This makes agreement on an explicit ambiguity more meaningful than agreement after anchoring.

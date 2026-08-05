# Failure table

| Experiment | Single change / condition | Result | Lesson |
| --- | --- | --- | --- |
| Parallel reduction Transformer | Fixed-N and multi-N controls | Held-out reduction did not generalize | Parallel token processing lacked the relevant arithmetic structure |
| Final-label tied VDF | Prompt-tail discrete register, tied Square→Reduce cells | 3.33% e1 test, 0% OOD; no depth rung | Final labels did not identify a reusable transition in this workspace |
| Legal e1 T curriculum | Loss phase T=1 → T<=2 → T<=3 | T=1 5.26%; T>=2 0% | Shallow exposure gives no composable transition |
| True final-label curriculum | Research T=1 → T<=2 → T<=4, 180 seconds | Near-100% train fit; 0% held-out x T=1/2/4 | This formulation memorizes phases rather than learning a law |
| Trace-label ablation | Add generated intermediate state loss | Better in-batch fit; 0% held-out final T=1/2/3 | Objective alone is not sufficient for this token-register state interface |
| Direct Transformer | Upstream final-output baseline | 2.00% test, 1.00% OOD; no depth rung | No evidence that removing recurrence solves the core task |
| Fresh direct chunk reducer | Predict chunk action and post-chunk digits jointly | Action selection learned; exact state transition failed | Do not relearn arithmetic and chunking together |
| Frozen unit reducer + controller | Select chunks over frozen exact primitive | Action selection / compression was limiting; repeated units remain O(q) | Controller representation needs per-position information; no speed solution yet |

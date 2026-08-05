# Final-label-only recurrent VDF submission candidate

Author: Codex

This is the legal transfer of today’s diagnostic mechanism. Each outer VDF
iteration applies two distinct learned modules with tied weights over time:

```text
register -- learned SquareCell --> latent square -- learned ReduceCell --> register
```

Both cells use bidirectional prompt interaction followed by an LSD-to-MSD GRU
scan. The state is a learned discrete token register at the answer-aligned tail
of the prompt. Only final evaluator labels train it. In particular, it does
not construct `x²`, quotient, comparison, subtraction, residue, or diagnostic
trace labels in the submitted forward/training path.

This card changes architecture from the Fable control to the explicit VDF
decomposition. Prediction is registered in `predictions.md`; local artifacts
belong in `runs/vdf_square_reduce_final_label_e1/`.

## First local result

The literal first version is not a useful speed baseline: despite Easy inputs
having only T=1/2/3, it executed 64 masked outer cells per forward. It completed
only 64 updates in 60 seconds and reached 2.67% test / 0% OOD (1.33% mean).
The next card changes only that execution loop to exactly the prompt-selected
maximum T, keeping all learned computation unchanged.

## Dynamic-depth result

The compute fix is confirmed: 434 updates replace 64, memory falls from 3.1
GiB to 735 MiB, and e1 test exact rises to 3.33%. OOD remains 0% (1.67% mean),
so it is not a submission candidate. The next one-variable control changes
only the optimizer to Muon+AdamW on the same dynamic-depth VDF model.

## Muon controls

Both Muon partitions are refuted. Broad matrix Muon gives 4.00% test / 0% OOD
(2.00% mean) with 99.8% train exact; restricting it to the Square/Reduce
attention and MLP matrices gives 2.67% / 0% (1.33% mean), also at 99.8% train.
Muon accelerates memorization rather than VDF generalization. Dynamic-depth
AdamW is the reference optimizer for this architecture.

## Fused valid-GRU card

The next isolated optimization replaces the Python reverse `GRUCell` scan with
a fused `nn.GRU` over only a reversed valid prefix. The old scan began from
right padding, so this is also a correctness control. Its baseline is
dynamic-depth AdamW: 434 updates/60 seconds, 3.33% test, 0% OOD.

Fused valid-GRU reaches 490 updates (+13%) but 2.00% test / 0% OOD. It is a
speed improvement but not an accuracy promotion. The next isolated card uses
active-row compaction in the fused implementation for mixed-T batches.

Active-row compaction reaches 4.00% test / 0% OOD (2.00% mean) at 717 MiB but
only 463 total updates. The next card changes only intermediate output work:
head, softmax, and STE quantization are evaluated on register positions alone.

Register-only intermediate logits finish 494 updates with 3.33% test / 0% OOD,
so they preserve the dynamic AdamW accuracy but are only a marginal speed win.
The next small control tensorizes one-time T parsing.

## Execution-optimization audit (local L40, 60 seconds, seed 74)

Author: Codex. Every condition below retains the same final-label-only VDF
objective, batch size, AdamW schedule, and e1 data. `test` and `OOD` are final
exact match; mean is their unweighted mean. These are one-seed controls, so
they establish throughput/correctness compatibility rather than a reliable
accuracy ranking.

| Condition | Updates | Test | OOD | Mean | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Dynamic-depth GRUCell (Hard-submitted reference) | 434 | 3.33% | 0% | 1.67% | Accuracy reference |
| Fused valid-prefix GRU | 490 | 2.00% | 0% | 1.00% | +13% updates; no accuracy promotion |
| Active-row compaction | 463 | 4.00% | 0% | 2.00% | Mixed; optional execution path |
| Register-only recurrent logits | 494 | 3.33% | 0% | 1.67% | Narrow speed promotion |
| Tensorized T parser | 453 | 1.33% | 0% | 0.67% | Refuted; reverted |

The L40 is left running but idle. Artifacts are retained on the local-GPU
mirror under `~/somn-taskb/runs/competition/` and are intentionally excluded
from Git.

# Kaiser-Sutskever Neural GPU multiplication

Status: core paper architecture reproduced; short fixed-width decimal and binary
screens both fail exact multiplication.

## Architecture

This replaces the earlier ConvGRU-inspired tape with the architecture in
Kaiser and Sutskever, *Neural GPUs Learn Algorithms* (arXiv:1511.08228):

- mental image `[width=4, sequence length=n, maps=24]`;
- input symbols embedded only into the first width column;
- two different 3x3 CGRU layers applied in sequence;
- the two-layer stack repeated `n` times;
- output decoded from the first width column;
- hard-cutoff sigmoid gates, 9% recurrent dropout;
- Adam with epsilon `1e-4` and gradient norm clipping at 1;
- six relaxed recurrent parameter sets with an increasing agreement penalty.

The current screens do not reproduce the paper's full optimization protocol:
there is no variable-length curriculum, gradient noise, 729-run hyperparameter
and seed search, or training to the paper's convergence criterion.

## Shared numeric task

Both screens use the same commutativity-safe 80/20 split of all numeric pairs in
`0..99 x 0..99`: 7,994 train and 2,006 held out.

| Representation | Sequence | Train exact | Test exact |
|---|---|---:|---:|
| Decimal | two 2-digit LSD operands, MUL, four product digits | 13.25% | **8.23%** |
| Binary | two padded 7-bit LSD operands, MUL, fourteen product bits | 7.99% | 7.68% |

Decimal two-digit by two-digit accuracy was 4.76%, and three-carry accuracy was
0.90%. Binary often predicted individual bits well (88.38% for bit 1 and above
93% in several sparse upper positions), but exact fourteen-bit products were
rare. Token loss therefore overstates arithmetic success.

## Interpretation

The actual Neural GPU architecture does not solve this task merely by being
instantiated. The published result depended on binary curriculum training and
rare successful configurations among a 729-run grid. The paper itself reports
failure on long decimal multiplication. These screens falsify a short one-seed
transfer, not the published long-training result.

Verified artifacts:

- `diagnostics/artifacts/prime-0e64a3962e874632adeb435a3b192ef4/paper_neural_gpu_decimal_2x2/`
- `diagnostics/artifacts/prime-0e64a3962e874632adeb435a3b192ef4/paper_neural_gpu_binary_7bit/`

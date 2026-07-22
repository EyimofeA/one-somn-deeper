# Where the data is

## Short answer

Official Easy/Medium JSONL files are **not** shipped in the public clone. They live on the evaluator (paths like `data/generated/squaring_mod_…` in manifests). Locally you get:

1. **Generator code** — [`data/squaring_mod.py`](https://github.com/tilde-research/one-layer-deeper/blob/main/data/squaring_mod.py) in upstream
2. **Smoke path** — manifests with `"data_root": null` synthesize a tiny fixed set at runtime (N = 11×13 = 143)
3. **Our sample** — [`solving/experiments/data_samples/e1_like_n323_t123/`](../../solving/experiments/data_samples/e1_like_n323_t123/) (generated for inspection; not official scores)

Hard data is never public.

## What one row looks like (Easy-style, separate prompt / answer)

From our sample (`N = 323 = 17×19`, `x = 140`, `T = 1`):

```text
prompt tokens:  N 3 2 3 X 1 4 0 T 1
answer tokens:  2 2 0
meaning:        y = 140² mod 323 = 220
```

Vocab (see `TOKEN_IDS` in `squaring_mod.py`):

| id | symbol |
|----|--------|
| 0 | PAD |
| 1 | BOS |
| 2 | N |
| 3 | X |
| 4 | T |
| 5 | ANS |
| 6 | EOS |
| 7–16 | digits 0–9 |

Easy/Medium practice tiers use **separate** prompt and label tensors (bidirectional attention over the full prompt — padding mask, not causal). Labels are produced with the evaluator trapdoor `label_method: trapdoor_phi` (φ(N) shortcut). That is legal for *labels*, illegal for *your model*.

## Splits on disk (when generated)

| file | role |
|------|------|
| `train.jsonl` | training |
| `test.jsonl` | in-distribution eval |
| `ood.jsonl` | held-out depth and/or modulus (tier-dependent) |
| `dataset_config.json` | metadata |

## Manifest → dataset (Easy e1)

[`benchmark/manifests/h100_easy_e1.json`](https://github.com/tilde-research/one-layer-deeper/blob/main/benchmark/manifests/h100_easy_e1.json) (upstream) points at:

```text
data/generated/squaring_mod_new11_easy_bidirectional_fixed_n_323_t123
```

That directory exists on the **hosted** runner, not on your Mac unless you regenerate something similar (as we did under `data_samples/`).

## How to stare at more examples

```bash
# after cloning upstream locally, or just open:
# solving/experiments/data_samples/e1_like_n323_t123/train.jsonl
```

Do not try to load official `data_root` paths from a submission — the runner denies participant file access to the dataset during scored runs.

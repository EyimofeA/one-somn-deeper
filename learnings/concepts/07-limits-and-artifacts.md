# Compute limits & what you get back

## Tier budgets

| Tier | Train | Eval | Attempts / UTC day |
|------|-------|------|--------------------|
| Easy | **60s** | 30s | **60** |
| Medium | **600s** | 300s | **6** |
| Hard | **3600s** | 1800s | **1** |

Persistent model state ≤ **500,000,000** elements. Submission file ≤ **256 KiB**.

## Do we get a model / checkpoint back?

**No.** Hosted runs return metrics (JSONL) and a score. Weights are not downloadable. Every experiment is train-from-scratch on the evaluator. “Iterate” = change `submission.py` and submit again — not fine-tune a returned checkpoint.

## torch.compile

Manifest field `runtime.compile` is **evaluator-owned**. Easy/Medium public manifests set `"compile": false`. Participants cannot turn it on from `Submission`. Throughput gains must come from smaller models, batch size, and simpler ops.

## What we *can* tune

Architecture, optimizer + **scheduler**, optional loss, `batch_size`, `eval_batch_size`, `max_steps` (≤ ceiling).

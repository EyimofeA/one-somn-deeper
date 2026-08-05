# One Somn Deeper research packet

Prepared by Codex on 2026-08-05 for a principal-researcher review.

This packet is a curated working handoff plus a complete Git-history bundle.
It is intended to be reviewed as a project, not as a single prompt.

## Start here

1. `PROJECT_OVERVIEW.md`
2. `RESEARCH_LOG.md`
3. `STATUS.md`
4. `experiment_table.csv`
5. `failed_experiments/FAILED_EXPERIMENTS.md`
6. `submission_code/` and `src/`
7. `reports/`

## Git history

`history/one-somn-deeper.bundle` is a full Git bundle of all reachable project
history. To inspect it separately:

```bash
git clone one-somn-deeper.bundle one-somn-deeper-history
```

## Security review

`SECURITY_REVIEW.md` documents the pre-package scan. No API keys, SSH keys,
credential files, or credential-pattern matches were included.

## Checkpoints

The selected diagnostic checkpoints are small and included under
`best_checkpoints/`. They are research artifacts, not hard-coded submission
weights. Larger or redundant checkpoints remain in the local archive outside
this packet.

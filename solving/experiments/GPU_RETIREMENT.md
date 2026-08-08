# One Layer Deeper L40 retirement runbook

Use the installed `prime-intellect-gpu-lifecycle` skill. This file adds only
the project-specific paths and retention policy. Never terminate before the
backup script writes and verifies `PRIME_BACKUP_VERIFIED.txt`.

## Trigger and target

- Trigger: Hard job `9e7404cb-b0c9-480a-aa64-8d90cc853d67` leaves
  queued/running, whether completed or failed.
- SSH alias: `oneL40`.
- Research root: `/home/ubuntu/somn-taskb`.
- Evaluator clone: `/home/ubuntu/one-layer-deeper`.
- Resolve the Prime pod ID from provider state immediately before termination;
  never infer it from the SSH hostname alone.

## Retain

- Every custom `.py`, shell script, config, manifest, and source snapshot.
- `diagnostics/artifacts/`, run metrics, evaluation reports, stdout/stderr logs,
  plots, and compact summaries.
- Checkpoints and model state (`*.pt`, `*.pth`, `*.ckpt`) in the ignored local
  artifact backup, even when not committed to Git.
- The exact hosted-submission source and its SHA/job metadata.

The primary copy target is a dated directory under the Git-ignored local
`diagnostics/artifacts/` tree. Use resumable `rsync` through the installed
`prime-intellect-gpu-lifecycle` skill's `scripts/backup_runs.sh`, then compare
remote/local file counts and total bytes.

## Exclude as recreatable

- `.venv/`, uv/pip/torch caches, downloaded wheels, and package build caches.
- `__pycache__/`, `*.pyc`, pytest caches, and editor/OS metadata.
- A clean evaluator checkout with no custom source, logs, metrics, or runs.

Do not remove individual remote files to save time. After the verified copy,
terminate the exact ephemeral pod; manual remote deletion adds risk and does
not replace provider-side termination. Locally, raw reports/checkpoints stay
ignored; only compact notes, configs, source, and plots belong in Git.

## Required sequence

1. Record `nvidia-smi`, active processes, remote disk use, and provider pod ID.
2. Stop custom training only if still running and safe to interrupt.
3. Run the copy-only backup helper for `/home/ubuntu/somn-taskb`; separately
   retain any custom files/runs in `/home/ubuntu/one-layer-deeper`.
4. Re-run the copy until file count and byte count match and
   `PRIME_BACKUP_VERIFIED.txt` exists.
5. Append the local backup path, verification counts, and pod ID to
   `solving/RESEARCH_LOG.md`.
6. Confirm no `rsync`/`scp` remains active.
7. Run the installed lifecycle skill's `scripts/prime_l40.sh kill POD_ID
   --backup-manifest MANIFEST --yes`.
8. Confirm both provider-side termination and unreachable SSH; report any
   intentionally excluded local material.

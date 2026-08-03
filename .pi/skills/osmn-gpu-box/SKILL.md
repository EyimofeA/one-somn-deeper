---
name: osmn-gpu-box
description: Bootstrap, check status, or tear down a Prime/cloud GPU box for One Somn Deeper training. Use when the user pastes an SSH target, says "set up this GPU", "start the box", asks to kill/cleanup GPU processes, or wants GPU status.
---

# osmn GPU box

## Commands (prefer the script)

From repo root:

```bash
./scripts/osmn gpu start ubuntu@IP [--alias oneL40] [--skip-datasets] [--skip-acceptance] [--force]
./scripts/osmn gpu kill [--wipe] [--target ubuntu@IP] [--local-only]
./scripts/osmn gpu status [--target ubuntu@IP]
```

Accepts bare IP, `ubuntu@IP`, or `ssh ubuntu@IP`.

## When the user pastes an SSH target

1. Run `./scripts/osmn gpu start <target>` — do not re-implement by hand unless the script fails.
2. Report: GPU name, CUDA major, torch version, ssh alias, path to `solving/experiments/.gpu_box.json`.
3. Point to `solving/experiments/OPS.md` for run recipes.
4. Reminder: Hard is hosted-only; local box = Easy/Medium manifests / research.

## Kill / cleanup

```bash
./scripts/osmn gpu kill          # stop train/monitor leftovers
./scripts/osmn gpu kill --wipe   # also delete ~/one-layer-deeper + uv caches
```

Always state: **kill does not stop billing** — terminate the instance in the provider UI.

## Status check

```bash
./scripts/osmn gpu status
```

State is cached in `solving/experiments/.gpu_box.json`. Check `updated_at` — if stale (>1 day), the box may need a restart.

## Agent rules

- Prefer the script over ad-hoc SSH commands.
- CUDA 13+ → `uv sync`. CUDA 12.x → cu126 pin. The script handles this.
- Never commit secrets. `.gpu_box.json` is gitignored.
- After start, update OPS "Current" IP only if user asks for docs refreshed.
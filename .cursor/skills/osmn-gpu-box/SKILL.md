---
name: osmn-gpu-box
description: >-
  Bootstrap or tear down a Prime/cloud GPU box for One Somn Deeper local
  training. Use when the user pastes an SSH target (ssh ubuntu@IP), says
  "set up this GPU", "start the box", "osmn gpu", or asks to kill/cleanup/
  wipe leftover GPU processes before terminating an instance.
---

# osmn GPU box

## Commands (prefer these)

From repo root:

```bash
./scripts/osmn gpu start ubuntu@IP [--alias oneL40] [--skip-datasets] [--skip-acceptance] [--force]
./scripts/osmn gpu kill [--wipe] [--target ubuntu@IP] [--local-only]
./scripts/osmn gpu status [--target ubuntu@IP]
```

Accepts bare IP, `ubuntu@IP`, or `ssh ubuntu@IP`.

## When user pastes SSH

1. Run `./scripts/osmn gpu start <target>` (do not re-implement by hand unless the script fails).
2. Report: GPU name, CUDA major, torch line, ssh alias, path to `solving/experiments/.gpu_box.json`.
3. Point them at `solving/experiments/OPS.md` for run recipes.
4. Reminder: Hard is hosted-only; local box = Easy/Medium manifests / research.

## Kill / cleanup

When user says kill, cleanup, wipe, or is about to terminate the rental:

```bash
./scripts/osmn gpu kill          # stop train/monitor leftovers
./scripts/osmn gpu kill --wipe   # also delete ~/one-layer-deeper + uv caches
```

Always say explicitly: **kill does not stop billing** — terminate the instance in the provider UI.

## Agent rules

- Prefer the scripts over ad-hoc SSH setup.
- CUDA 13+ → `uv sync`. CUDA 12.x → cu126 pin (script handles this; never bare `uv sync` on those boxes).
- Do not commit secrets. `.gpu_box.json` is gitignored ephemeral state.
- After start, update OPS “Current:” IP only if the user wants docs refreshed (script writes state file either way).

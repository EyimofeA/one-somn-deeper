# Pitfalls — things the agent gets wrong repeatedly

## Studio/HTML exports

**Problem:** `studio_export_html` with relative paths (`solving/figures/x.png`) doesn't embed images. The `resourceDir` option is unreliable.

**Fix:** Put the markdown file **in the same directory as the images** and use `./image.png` paths. Then export that file. Images embed properly.

## Hard #1 vs Hard #2 metrics

**Problem:** `hard2_99c4d7d3_metrics.jsonl` = Hard #2 (never learned, train loss flat at 2.15). Hard #1 (`claude_hard_h1`) = the one that grokked (train 100%, eval 0%). The metrics for Hard #1 are not saved locally.

**Fix:** When referencing "the Hard run that grokked," confirm which file you're looking at. If `hard2` is in the filename, it's NOT the grokking run.

## Context window

**Problem:** Agent hardcodes 131K for all models. DeepSeek V4 Pro has **1M** context window. GPT-5.6 models have 200K.

**Fix:** Check `KNOWN_WINDOWS` in the status-line extension before assuming. Search `api-docs.deepseek.com` for current specs.

## Competing submissions

**Problem:** The competition API allows one submission at a time per API key. A second submit while one is queued/running gets rejected.

**Fix:** Submit without `--wait`, note the job ID, check status later. Don't submit two in parallel.

## Symlinks in submissions/

**Problem:** `solving/submissions/` contains symlinks to `../experiments/2026-07-21_<name>/`. Many are broken — the actual experiment directories have different names. Always resolve the symlink target before reading.

**Fix:** Use `find solving/experiments -name "submission.py"` to find actual files. Don't assume symlinks are valid.

## Competition venv

**Problem:** The `one-layer` CLI requires the competition virtual environment. It's at `competition/.venv/bin/activate`, not at the repo root.

**Fix:** Always `cd competition && source .venv/bin/activate` before any `one-layer` command.

## Skill announcements

**Problem:** AGENTS.md rule #6 says "just say 'loading the X skill' and use it." Agent often forgets to announce.

**Fix:** Before using a skill's framework, say "Loading the X skill." Not using the skill = no announcement needed.

## Image viewing

**Problem:** DeepSeek V4 Pro doesn't support images. The model can't see screenshots or plots inline.

**Fix:** Switch to a vision model (gemini-2.5-flash, kimi-k3, gpt-5.6-luna) with Ctrl+P when the user shares an image. Or read the image and describe it from the tool output.

## Metrics JSONL format

**Problem:** Competition metrics use `exact_accuracy` (not `exact_match`), `type` field filters train/eval/ood, loss is in `loss` (not `train_loss`).

**Fix:** Check one line of the JSONL before writing plot code. `head -1 file.jsonl | python3 -m json.tool`.
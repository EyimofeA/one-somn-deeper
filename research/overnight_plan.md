# Unattended window plan — Codex

Scheduled start: approximately 03:16 WAT on 2026-07-26.  The launcher does
not touch the remote host before that time.

1. Inspect `nvidia-smi`, active processes, existing run roots, and the Task A/B
   registry on `twoA6000`; select only GPUs without compute processes.
2. Run the Task B 32-example audit on CPU and save its complete log as the
   audit report.  A failed audit stops all Task B training.
3. With two idle GPUs, run the priority-constrained pair: Task B baseline
   seed 0 and Task A `carry_shuffled` seed 0.  This follows the supplied
   insufficient-compute priority order.  With one idle GPU, run only Task B
   seed 0.  A GPU chain starts seeds 1 then 2 only after its preceding run
   exits and while the four-hour window remains open.
4. After each completed Task B run, run the existing evaluator; after seed 0,
   run the existing Task B diagnostic analysis.  No intervention is selected
   or launched from its observations.
5. At the four-hour mark write an unattended handoff, preserve any valid
   in-flight jobs, and sync logs/reports/small JSON artifacts back locally.

All output directories are timestamped under `diagnostics/runs/overnight_*`.
The launcher records source commit, commands, seeds, devices, start/end times,
logs, configs, checkpoints, and JSON reports.  It never uses data parallelism,
external data, pretraining, recurrence, Fourier embeddings, or model-forward
arithmetic shortcuts.

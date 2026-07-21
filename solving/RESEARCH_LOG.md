# Research Log

Append-only log of experiments, findings, and decisions.

## Format

```
### YYYY-MM-DD — Title
- **Hypothesis:**
- **Setup:**
- **Result:**
- **Next:**
```

---

### 2026-07-21 — Phase 0 setup
- **Hypothesis:** Workspace scaffold and competition clone are sufficient to run CPU smoke on Mac.
- **Setup:** Cloned `tilde-research/one-layer-deeper` into `competition/`; `uv venv` + `uv sync`; unittest + smoke_cpu manifest.
- **Result:** 105 tests OK; CPU smoke passed (mean_exact_accuracy≈0.008 on untrained baseline, expected). Required `uv python install 3.13.5` (global uv defaulted to 3.14.3).
- **Next:** Phase 1 baselines in `solving/submissions/`.

### 2026-07-21 — Workflow split
- **Hypothesis:** Clear parent/subagent ownership reduces doc bloat and API spend.
- **Setup:** Parent owns learnings + research log + interpretation; Composer 2.5 subagents own code only; one subagent per task.
- **Result:** `AGENTS.md` rewritten with strict read order (1→5). Plan v6 tooling section updated.
- **Next:** Delegate `b0_transformer` to one Composer subagent; parent logs results.

### 2026-07-21 — Data location + teaching notes
- **Hypothesis:** Official Easy JSONL is evaluator-side; we can regenerate e1-like samples for inspection.
- **Setup:** Generated `solving/experiments/data_samples/e1_like_n323_t123/` via `SquaringModGenerationConfig` (N=323, T=1/2/3 + OOD T=4). Wrote concepts 02–04 (data, cheating, arch/optimizer).
- **Result:** Sample row `N 323 X 140 T 1 → 220`. Labels use `trapdoor_phi`. No `competition/data/generated/` in public clone. No `submission.py` in `solving/submissions/` yet (only docs + sample data).
- **Next:** `git init` + first commit; then b0 subagent.

### 2026-07-21 — Phase 1 baselines implemented
- **Hypothesis:** Three minimal inductive biases (Transformer / MLP / RNN) can share the Submission contract and pass CPU smoke.
- **Setup:** Composer 2.5 wrote `b0_transformer`, `b1_mlp`, `b2_rnn` under `solving/submissions/`. Validated + smoke_cpu (0.1s → 0 train steps).
- **Result:** All three validate + smoke pass. Untrained exact≈0.008 (noise). Params @ smoke vocab: b0~202K, b1~399K, b2~95K.
- **Note:** Easy ~60 scored runs/day is enough to experiment without Colab for now. Multiple shared-block applications inside one `forward` are allowed; evaluator still calls forward once per step. Official train JSONL stays on the evaluator — local `data_samples/` is for eyes only.
- **Next:** `one-layer login` then Easy e1 submits for b0→b1→b2; parent plots ladder.

### 2026-07-21 — Easy e1 baseline ladder (scored)
- **Hypothesis:** Attention (b0) > BiGRU (b2) > MLP (b1) on fixed-N Easy e1 under identical AdamW.
- **Setup:** Submitted e1 for all three; 60s H100 each. Jobs: b0 `fddbf10e…`, b1 `7843e881…`, b2 `6cd36e0b…`.
- **Result (facts):** mean exact b0=1.00%, b2=0.67%, b1=0.33%. Test: 2.0% / 1.3% / 0.7%. OOD: all 0%. Train curves: b2 train exact rose highest (~21% by end) while test stayed ~1.3%. b0 completed 261 steps.
- **Plots:** `solving/experiments/figures/fig_baseline_ladder_e1.png`, `fig_baseline_train_curves_e1.png`.
- **Next:** User interprets; candidate follow-ups = depth_looped from b0, or diagnose b2 train/test gap.




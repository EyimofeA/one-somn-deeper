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

### 2026-07-21 — Metrics reading (clock vs convergence)
- **Question:** Did e1 end early? Do we have loss? Is OOD ours?
- **Result:** All three hit `training_seconds=60.1` with max_steps=1e6 unused → wall-clock stop. Loss is in JSONL; added `fig_baseline_train_loss_e1.png`. Hosted `split=ood` is evaluator-side (0% exact for all). Train exact is logged-batch, still rising at cut. Note: `learnings/concepts/06-reading-metrics.md`.
- **Next:** Max-out plan — more steps/sec and/or Medium 600s; LR schedule; then depth_looped for OOD (currently 0%).

### 2026-07-21 — Maxed small baselines Easy e1
- **Hypothesis:** Smaller width + warmup/cosine + batch 256 → more steps → higher Easy e1 mean than v1.
- **Setup:** `*_max` submissions (d=64, AdamW 3e-3, SequentialLR warmup+cosine, bs=256). compile still evaluator-false.
- **Result:** steps ~2× (≈555–585). mean: Transformer_max **1.33%** (was 1.00%), MLP_max **1.00%** (was 0.30%), BiGRU_max **1.00%** (was 0.70%). OOD still 0% all. L_end still ≈1.2–1.9.
- **Plots:** `fig_max_vs_v1_e1.png`, `fig_max_ladder_e1.png` (open in IDE if chat shows placeholder).
- **Next:** Treat `b0_transformer_max` as Transformer reference; add depth_looped on that axis for OOD.

### 2026-07-21 — Plot dashboard + Karpathy map
- **Setup:** Generated all JSONL-derivable figures under `solving/experiments/figures/` (see `PLOTS_INDEX.md`). Docs: `09-what-is-returned.md`, `10-karpathy-recipe.md`.
- **Result:** No LR/grad/optim traces in API — only loss/exact/steps/eval. Nascent scaling: more steps → higher mean within 60s. OOD still 0% on ladder.
- **Next:** depth_looped K=4/8 + width sweep d∈{32,96,128} agents; then Easy e1 submits.

### 2026-07-21 — Depth + width Easy e1
- **Hypothesis:** (1) Looping shared block raises OOD. (2) Width has a U-shape under 60s (too wide starves steps / overfitting).
- **Result:** K=4 mean 1.83% with **ood 3.0%** (first nonzero). K=8 mean 1.67% ood 2.0%. Width: **d=32 mean 2.70% ood 4.0%** best overall; d=64 weakest mean in the width line.
- **Plots:** `fig_depth_ablation_e1.png`, `fig_depth_train_exact_e1.png`, `fig_scaling_width_e1.png`, plus full dashboard in `PLOTS_INDEX.md`.
- **Next:** Combine small width + loops (e.g. d=32 × K=4) as next card; Karpathy one-change rule.

### 2026-07-21 — Combo d32 × K=4 Easy e1
- **Hypothesis:** Best width × best depth multiplies (non-additive) under same AdamW recipe.
- **Setup:** `depth_d32_k4` — d=32, shared block ×4, batch 256, warmup/cosine. Job `83e291a5…`.
- **Result:** mean **5.50%**, test 2.0%, ood **9.0%**, steps 471. Parents: d32 K1 mean 2.70% ood 4%; d64 K4 mean 1.80% ood 3%.
- **Plots:** `fig_combo_d32_k4_e1.png`, `fig_combo_d32_k4_train_e1.png`.
- **Next:** New reference = `depth_d32_k4`. Candidates: K sweep at d=32, or Easy e5 transfer.









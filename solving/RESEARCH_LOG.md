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

### 2026-07-21 — Easy e5 transfer (d32×K4)
- **Hypothesis:** Best e1 model transfers to variable-modulus e5.
- **Setup:** Same `depth_d32_k4` on Easy e5. Job `07ed6ab5…`.
- **Result:** mean **0.79%** (test 1.1%, ood 0.5%), steps **2527** vs e1’s 5.50% / 471 steps. Sharp drop under varying N.
- **Plots:** `fig_d32_k4_e1_vs_e5.png`, `fig_d32_k4_e1_vs_e5_curves.png`.
- **Optim details:** still not returned (train JSONL = step/loss/exact/elapsed only).
- **Next:** Improve modulus generalization (data-aware inductive bias or more capacity under e5’s higher step count).

### 2026-07-21 — d32 K-sweep + e5 gate
- **Hypothesis:** Score vs K rises then falls under 60s; e1 optimum may not match e5.
- **Setup:** K∈{2,3,6,8} new + existing K=1,4. e5 for K=2,3,4. left=41 after.
- **Result:** e1 best **K=2 mean 6.20%**; then K=4 5.50%, K=3 5.00%; K=8 back to 2.70%. e5: K=4 **0.80%** > K=2 0.50% > K=3 0.40%.
- **Plots:** `fig_d32_k_sweep_e1.png`, `fig_d32_k_e1_vs_e5.png`. Ideas: `11-ideas-backlog.md`. Quota: `DAILY_QUOTA.md`.
- **Next:** N-generalization / adaptive compute; keep K=4 as e5 reference, K=2 as e1 peak.

### 2026-07-21 — N-conditioning FiLM (d32×K4)
- **Hypothesis:** Pooling the N-digit span and FiLM-modulating each loop improves e5 (variable N) without hurting e1.
- **Setup:** `depth_d32_k4_ncond` — same d=32, K=4, AdamW recipe; +`n_proj` +`film` (~16.7K params). Jobs e1 `f752d166…`, e5 `7cecdbb0…`. left=39 after.
- **Result (facts):** e1 mean **5.83%** (test 2.7%, ood 9.0%, steps 407) vs base 5.50% / 2.0% / 9.0% / 471. e5 mean **0.29%** (test 0.3%, ood 0.3%, steps 2215) vs base 0.80% / 1.1% / 0.5% / 2527.
- **Plots:** `fig_ncond_vs_base_e1_e5.png`, `fig_ncond_train_curves.png`. Naming note: `learnings/concepts/12-current-arch-and-naming.md`.
- **Next:** Drop this FiLM recipe from the Medium shortlist; try ACT / adaptive loops or a different N-binding (e.g. cross-attn to N tokens). Funnel stays Easy → top ~5–10 → Medium → Hard.

### 2026-07-21 — Adaptive loops (soft ACT, d32, K_max=8)
- **Hypothesis:** Learned halt weights over up to 8 tied block passes let compute follow difficulty better than fixed K=4.
- **Setup:** `depth_d32_act` — soft mixture of intermediate states; halt from masked mean pool. Jobs e1 `ef972089…`, e5 `4af90448…`. left=37 after.
- **Result (facts):** e1 mean **3.83%** (test 2.7%, ood 5.0%, steps 397) vs K4 5.50% / 2.0% / 9.0% / 471. e5 mean **0.79%** (test 0.7%, ood 0.8%, steps 1798) vs K4 0.80% / 1.1% / 0.5% / 2527. Train loss still drops to ~1.7 (e1) / ~2.1 (e5) by ~100 steps then plateaus.
- **Plots:** `fig_act_vs_k4_e1_e5.png`, `fig_act_train_curves.png`.
- **Next:** Fixed K still wins e1; ACT e5 ≈ K4. Prefer fixed-K shortlist (K=2 e1, K=4 e5) unless ponder-loss ACT is worth one more card.

### 2026-07-21 — UT depth embeddings vs plain loops
- **Hypothesis:** Literature UT (tied block + per-loop depth emb, Dehghani et al. 2018) beats our plain tied loops under the same d=32 / optimizer.
- **Setup:** `depth_d32_k2_ut`, `depth_d32_k4_ut`. Jobs: k2 e1 `db4e7794…`, k4 e1 `489a8575…`, k2 e5 `ad472921…`, k4 e5 `e6d918dc…`. left=33 after. Glossary: `learnings/concepts/13-decisions-glossary.md`.
- **Result (facts):**

| arch | e1 mean | e5 mean |
|------|---------|---------|
| plain K2 | 6.20% | 0.50% |
| **UT K2** | **6.50%** | 0.70% |
| plain K4 | 5.50% | 0.80% |
| **UT K4** | 4.70% | **1.00%** |

- **Plots:** `fig_ut_vs_plain_e1_e5.png`.
- **Next:** Promote **UT K4** as e5 reference (new best 1.00%); **UT K2** as e1 peak (6.50%). Medium candidate when funneling: UT K4 on m5-like (variable N+T). Hold Medium until you greenlight.

### 2026-07-21 — Discord beta meta ingested
- **Setup:** User pasted GPU MODE `#one-layer-deeper` history (deadline TBD/~Aug, Hard≠Easy algorithm, Easy 100% via solvers, loophole culture in beta).
- **Result:** Wrote `learnings/concepts/14-discord-beta-meta.md`; rewrote `03-cheating-boundary.md`. Confirms our UT/loop learning line matches organizer intent; Easy ~6% is not comparable to solver 100%.
- **Next:** Principal chooses: (A) continue learned UT line → e3 then Medium, or (B) one Easy grey probe (intermediate aux) for notes only.

### 2026-07-21 — Principal: Karpathy + Claude reserve + scaling clarification
- **Setup:** Wrote `solving/experiments/PRINCIPAL_NOTES.md`. Quota left=33; reserve 10 Easy for Claude Code → parent ≤23 today. Scaling laws = model-size curves when we grow, not ponder. Diffusion queued. Aux deferred (Karpathy).
- **Result:** Docs updated (`09` aux not returned; `10` stage; `11` backlog; `13` scaling note; quota reserve).
- **Next:** Claude Code critique + its Easy cards within the 10; parent holds or does low-burn Karpathy cards only inside budget.

### 2026-07-21 — Paper log: T²MLR (2607.15178)
- **Setup:** User shared Jack Cai tweet + arXiv. Wrote `learnings/papers/` index + `2607.15178-t2mlr.md`. Official impl: https://github.com/princeton-pli/T2MLR.
- **Result:** Paper = temporal middle-layer cache across **decode tokens**, not our depth-UT. Portable bet for us: **middle-only depth loop** (`depth_d32_midloop_k4`) on e1+e5 vs `depth_d32_k4_ut`. Full temporal cache = later/heavier.
- **Quota:** ~20 Easy left → parent ≤10 after Claude reserve. No scored midloop submit this turn.
- **Next:** Greenlight midloop card or leave for Claude Code critique + implement.

### 2026-07-21 — Midloop depth (pre / mid×4 / post) vs UT K4
- **Hypothesis:** Looping only a middle block (T²MLR-inspired, depth-mapped) beats full tied UT under same optimizer.
- **Setup:** `depth_d32_midloop_k4` (~39K params). Jobs e1 `5331f763…`, e5 `2dc0335b…`. left=31 after. Paper note: `learnings/papers/2607.15178-t2mlr.md`.
- **Result (facts):** e1 mean **0.83%** (test 0.7%, ood 1.0%, steps 567) vs UT K4 4.70%. e5 mean **0.79%** (test 0.9%, ood 0.7%, steps 2817) vs UT K4 1.00%. Train batch exact on e1 ended **30.9%** with L_end **1.06** while eval collapsed — train/eval gap.
- **Plots:** `fig_midloop_vs_ut_e1_e5.png`.
- **Next:** Reject midloop for shortlist; keep **UT K4 / UT K2** as references. Parent budget ~20 Easy still mostly intact (left=31).


















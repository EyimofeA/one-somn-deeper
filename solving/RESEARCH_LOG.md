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

### 2026-07-21 — UT K2 train / K4 eval
- **Hypothesis:** Extra loops only at eval add OOD/test depth without burning train steps.
- **Setup:** `depth_d32_k2_ut_evalk4` — depth emb size 4; `self.training` → K=2 else K=4. Jobs e1 `84885af2…`, e5 `00e30b38…`. left=29 after.
- **Result (facts):** e1 mean **6.83%** (test 4.7%, ood 9.0%, steps 609) vs UT K2 6.50% / 4.0% / 9.0% / 393 — **new e1 best**. e5 mean **0.42%** (test 0.7%, ood 0.2%, steps 3583) vs UT K2 0.70% and UT K4 1.00% — worse on e5.
- **Next:** Promote evalk4 as e1 reference; e5 reference stays UT K4. Don’t use train2/eval4 as sole Medium pick without an e5-competent twin.

### 2026-07-21 — Easy noise (evalk4 e1 ×3) + Medium ×3 + optsched fix
- **Noise:** `depth_d32_k2_ut_evalk4` e1 ×3 → all **6.80%** mean (stdev **0.00 pp** under seed 74). Steps varied 407–609.
- **Medium (broken cosine):** UT K4 m5 **0.09%**, UT K4 m1 **0.08%**, evalk4 m5 **0.14%**. m5 saw ~51k steps vs scheduler T_max≈4800 → cosine **restart sawtooth** (Claude diagnosis confirmed).
- **Fix:** `depth_d32_k4_ut_optsched` — LambdaLR warmup+cosine **clamped** (no restart); horizon ≈120×train_seconds. Job m5 `3931290c…`.
- **Result:** optsched m5 mean **0.17–0.20%** (test 0.1%, ood 0.2%, steps 70007) vs broken UT K4 m5 0.09%. Still tiny absolute.
- **Hard pick (parent):** submit `depth_d32_k4_ut_optsched` unless Claude’s Medium beat ~0.2%. Principal approves Hard. Parent done after this run.

### 2026-07-21 — Durable rule: no fixed-T cosine on long clocks
- **Setup:** Documented sawtooth failure; added `learnings/concepts/15-lr-schedules-wallclock.md` + `.cursor/rules/lr-schedule-wallclock.mdc`.
- **Result:** Prefer **inv-sqrt/Noam** (adaptive in step, no T_max). Plateau schedulers unusable (`step()` has no metric). Clamped cosine = acceptable patch.
- **Next:** New submissions default to inv-sqrt unless ablating schedule intentionally.

### 2026-07-21 — EOD organize + Hard standing
- **Setup:** Slimmed docs into `solving/STATUS.md`, `learnings/sessions/2026-07-21.md`, `experiments/OPS.md`; removed PRINCIPAL_NOTES / NOISE / DAILY_QUOTA; submissions README = active vs archive.
- **Result:** Hard LB **#11 at 0.03%** (Claude run). Combined lessons in session note; next = concept 17 (re-quantised recurrence), not more width.
- **Next:** Local grokking / WD / requant — quota only to confirm.






















### 2026-07-21 — Claude Code session: e1 invalidated, LR schedule fixed, Hard groks
- **Budget:** 20 Easy, 3 Medium, 1 Hard (all exhausted). Cards: `solving/submissions/claude_*`.
- **Detail:** `learnings/concepts/16-representation-vs-throughput.md` (do not restate here).
- **Hard H1 result:** `claude_hard_h1` d=2048 / 50.5M / K=4 → **0.03%**, rank 11/18.
  Leaderboard top is 0.40% (az), 0.19% (Frosty40), then ranks 3–16 span 0.05–0.02% —
  a 14-way near-tie we sit inside. **Nobody has solved this task.**

- **Result (facts):**

| finding | evidence |
|---|---|
| **e1 is not a valid ranking signal** | x^(2^T) mod 323 has only **19 distinct answers for all T≥4**; e1's ood majority baseline is **9.94%** and every card ever run scores ood ≈ 9.00% — below trivial. e1 combined trivial ≈ 6.42% vs repo best 6.83% |
| **LR schedule broken past Easy** | `t_max = seconds*8` assumes ~8 steps/s; Medium runs 75–97. Cosine is periodic past T_max → sawtooth for ~40,000 of 45,000 steps. Found independently by both agents |
| **Hard GROKS** | 190,017 steps: ~60k-step plateau at loss 2.17, transition at **~64,000**, then train loss **0.0000** and train exact **100%** — with eval **0.0000%** on all three splits (eval loss 15.8/16.2/16.4) |
| **Medium stopped 6k steps short of the transition** | best Medium run = 58,060 steps; transition begins ~64,000. Every Medium flatline from both agents was **pre-grokking**, not underfitting |
| **place value replicates** | e1 3/3 identical: 5.83% vs anchor 4.67%. (Still below e1's own 6.42% trivial baseline — real effect, invalid benchmark) |
| **noise is per-dataset** | e1 bit-reproducible (4.67×3, 5.83×3); e3 is not (same file: 1.31% vs 0.69%) |
| **depth codes must be distinct, not trained** | zero-init depth emb collapsed best Easy card 6.83% → 2.33%; it trained *best* (9.0%) and generalised *worst* |
| **tied-head init defect** | head tied to token_embedding + default N(0,1) → initial loss 83.4 at d=256, 12.5 at d=32, vs ln(17)=2.83. Affects every card in repo |
| **width→throughput knee** | e5, 60s: d=512 1,981 steps / d=2048 1,765 (−9%, 1580× capacity) / d=4096 1,005 (−48%) |
| **adaptive depth loses** | m1: fixed K=4 → 58,060 steps, loss 2.056, 0.117%; loops=T → 30,249 steps, loss 2.135, 0.050% |
| **H1 has 3 eval splits** | `test`, `ood_t`, `ood_n_t` — Hard explicitly scores unseen T and unseen N+T |

- **Retracted this session:** "C1 doubles the anchor on e3" (replication: 1.31% vs 0.69%,
  same file); "width will pay once the clock stops binding" (it stopped binding and the
  model still learned nothing); "the model is underfitting, the answer is capacity"
  (symptom real, mechanism wrong — steps tipped it, not width); phase/Fourier
  architecture (a task-specific solver; Hard uses a different algorithm by design).

- **Next:** the failure is now located — nothing forces the model to learn a *reusable
  single step*, so with enough steps it memorises. In order: (1) weight decay 0.1 →
  1.0/3.0, one constant, the grokking literature's main memorise→generalise knob;
  (2) force iteration architecturally — state space = output space, one shared step
  applied T times, **re-quantise toward one-hot digits between steps** so error cannot
  accumulate; (3) input injection each loop; (4) label-free entropy aux via the
  `auxiliary` return (reachable, untried); (5) exploit T=1 rows as direct single-step
  supervision. **Do not reach for more capacity.**

- **Process:** run the grokking check LOCALLY. `competition/` generates data and runs the
  real training loop offline with zero quota. The e1 collapse was a seconds-long local
  calculation that would have saved ~15 scored runs; a 60k-step transition is invisible
  in a 600s clock but cheap to find overnight. Iterate locally, spend quota only to
  confirm.

## 2026-07-22 — Ingest protocol + lecture notes; experiment layout

**Question:** Land decision protocol and Path D mechanism notes without mixing evidence and hypotheses.
**What we did:** Copied Downloads `RESEARCH_PROTOCOL.md` → root (+§11 layout); `one-layer-deeper-notes.md` → `learnings/readings/`; added `HYPOTHESES.md`, `concepts/18-…`, `experiments/LAYOUT.md` + `predictions.md`, `scripts/extrapolation_curve.py` stub, `.cursor/rules/research-protocol.mdc`; wired AGENTS/README/STATUS/curriculum.
**Result:** [SOURCED] Files on disk as above. No new scored runs.
**Dead ends:** Did not migrate day-1 `submissions/` into dated experiment dirs (deferred; LAYOUT documents the target).
**Lesson:** Protocol separates PREDICT (human) from implementation (agent); notes stay in readings until runs cite them into concepts.
**Promote?:** Research-protocol rule already added.

## 2026-07-22 — Migrate submissions → dated experiment dirs

**Question:** Split active vs full history on disk.
**What we did:** 39 cards → `solving/experiments/2026-07-21_<name>/` (submission.py, config.json, NOTE.md). `solving/submissions/` now 5 symlinks to active cards only.
**Result:** [SOURCED] Symlink verify OK; metrics remain under `experiments/metrics/`.
**Dead ends:** None.
**Lesson:** Migration was a script (~seconds), not a 30-minute hand move — estimate the automation, not the file count.
**Promote?:** No.

## 2026-07-22 — GPU box ops locked in

**Question:** Document Prime L40S local runner so agents can cold-start without quota.
**What we did:** Expanded `solving/experiments/OPS.md` (connect, cu126 torch, never uv sync, scp+runner, acceptance steps/s); AGENTS Compute points at it.
**Result:** [SOURCED] Box IP ephemeral in OPS; L40S ~145 steps/s vs H100 ~96.8 on d32 K4.
**Dead ends:** uv sync on box → CUDA-13 torch / NCCL break.
**Lesson:** Local runner env ≠ `uv sync` defaults; pin cu126 and invoke `python` directly.
**Promote?:** Already in OPS + AGENTS.

## 2026-07-23 — P2 grokking ladder rung 1; Hard shot #2 fired; harness cross-check

**Question:** Is one step of x² mod N learnable at all for a *fixed* N (P2 ladder
rung 1), given rung 2 (multi-N seen, unseen prompt) already floored at 0.5-0.75%?
**What we did:** Reconciled `claude code fable/` handoff with repo state (the
t1only probes = rung 2, not rung 3 — correction appended to
`2026-07-22_t1only_probe_rope/NOTE.md`). Ran handoff `smoke_test.py` +
the real `validate_submission_source` against `submission_v2.py` on the L40S —
all green, no replica/evaluator discrepancy. Generated fixed-N x-split T=1
datasets (N=323, N=1073, `separate_input_output=true` — the causal_lm default
leaks answers into input_ids and scores a fake 100%). Ran the rope anchor at
wd=0.1 and wd=1.0, 900s each, monitored.
**Result:** [SOURCED — `experiments/metrics/rung1_*monitor.jsonl`, NOTEs in
`2026-07-23_t1only_fixedn_wd01/` and `_wd1/`]

| run | train EM | test EM final (peak) |
|---|---|---|
| N=323 wd0.1 | 100% | **5.17% (6.90%)** |
| N=1073 wd0.1 | 100% | 0.00% (0.99%) |
| N=323 wd1.0 | 61% | 1.72% |
| N=1073 wd1.0 | 31% | 0.00% |

D1 per-position accuracy on wd1 checkpoints ≈ marginal baseline (priors only).
1800s wd0.1 reruns with checkpoints launched (results pending).
**Hard shot:** `submission_v2.py` fired as `99c4d7d3` (use-or-lose, user-approved,
~15 min before UTC day end; queue was blocked by a running Easy E5 job until
~23:5x). Expectation zero per gate; value = monotone shot + returned split vector.
**Dead ends:** causal_lm-format first launch (leak, discarded); wd=1.0 at d=32
(never fits train); pkill self-match killing its own ssh session.
**Lesson:** The ladder's failure point is *below* rung 2: the one-step map is
barely learnable even per-modulus at this scale — cross-N transfer is not yet the
binding constraint. Ladder work should stay at rung 1 (width, budget, digit-pair
primitives) until a fixed-N run clears a real number (>50%), before any further
Hard-architecture iteration.
**Promote?:** separate_input_output gotcha → worth a line in OPS.md.

### 2026-07-23 (addendum) — 1800s rung-1 reruns: the climb is non-monotone
[SOURCED — `experiments/metrics/rung1_n{323,1073}_1800s_monitor.jsonl`]
2x budget raised the peaks (N=323 **8.62%** @ 146k, N=1073 3.47% @ 22k) but both
decayed to 1.5-1.7% final; D1 per-position accuracy at final checkpoints ≈
marginal baseline. Peak models not checkpointed (finals only). Rung-1 d=32
ceiling: ~8.6% peak / ~2-5% final. Hard `99c4d7d3` still running at check time.

### 2026-07-23 (addendum 2) — Hard #2 scored: 0.05%, split vector says "step never exact"
[SOURCED — `experiments/metrics/hard2_99c4d7d3_metrics.jsonl`]
`submission_v2.py` (confidence-gated v2): **0.05%** (new personal best, was 0.03%).
Splits: test 0.1% / ood_t 0.0 / ood_n_t 0.0. Train loss pinned at 2.17 (digit-
marginal floor) for the whole hour — train EM 0.0, i.e. v2 never memorized on H1,
matching the m5 flat pathology. Day-6 tree: test≈ood_t≈0 → the one-step map never
got exact. Confirms the rung-1 diagnosis; no new information beyond it.

## 2026-07-23 — Digit-recurrence diagnostic gates

**Question:** Before attempting the full recurrence `x^(2^T) mod N`, can a
learned model execute the primitive operations needed by one reusable decimal
step rather than memorize a finite table?

| card | frozen variable | data / split | held-out exact match | classification |
|---|---|---|---:|---|
| gate0_copy | target is X | 1–3 digit train; 4-digit held out | 100% | confirmed: routing is sufficient |
| gate1_square | target is decimal X² | same setup | 7% same-length peak; 0% 4-digit | refuted: raw product formation is not learned compositionally |
| gate1_digit_product | one decimal digit pair → product | held-out pairs, four products absent from train | 15% | confounded: output values were also unseen |
| gate1_digit_product_seen_outputs | unchanged Transformer; held pairs with every product value seen in train | 80/20 pairs, 10 repeats | 45% peak; 25% final | pair relation partly learned, then overfit |
| gate1_bilinear_digit_cell | fixed ordinal digit inputs + trainable NALU interaction | same repaired split | 30% peak; 25% final | refuted: numeric multiplicative bias does not beat baseline |
| gate1_carry_scan | continuous shared GRU carry state | 8k train / 2k disjoint test; 1–7 columns | 79.45% at 4k; 98.15% peak at 8k | confirmed: a learned recurrent state can normalize carry |
| gate1_quantized_carry_scan | hard 64-prototype state after every transition | same data / seed / 4k steps | 0.25% peak | refuted: hard projection collapses optimization |
| gate1_soft_prototype_scan | soft mixture instead of hard selection | same data / seed / 4k steps | 38.80% final | partial: differentiability matters, but a prototype bottleneck loses state information |
| gate1_soft_prototype_8k | training budget 4k → 8k; same soft state | same data / seed | 98.75% final | confirmed: prototypes are viable but optimization-delayed |

**Implementation:** The carry prompts are `N` followed by one to seven
three-digit, least-significant-column-first totals. The target is the emitted
digit after each total plus two carry-flush digits. The generator performs
arithmetic only to make labels; the model receives only tokens.

**Measurement:** exact full output sequence match on 2,000 disjoint prompts.
The continuous baseline ran 4,000 steps in 130.2 seconds; hard prototypes ran
176.3 seconds and soft prototypes 155.1 seconds. The 8k soft run reached
98.75% in 328.5 seconds. The matched continuous 8k control reached 98.15%
peak (97.85% final) in 250.5 seconds, but learned much earlier: 69.15% at
3.6k steps versus soft's 16.6%. The soft state had a 0.6-point single-run peak
edge, insufficient to claim better generalization; its established effect is
slower optimization. Its evaluation interval was 200 rather than 100, so
wall-clock LR scheduling makes its early curve not exactly comparable to the
4k run. Remote evidence is retained
on `twoA6000` at `results_local/gate1_{carry,quantized_carry,soft_prototype}_scan/`
and `results_local/gate1_{soft_prototype,continuous_carry}_8k/`.

**Conclusion:** the next gate must attack held-out digit products. Carry is
not impossible for a learned recurrence; forcing a hard discrete state before
the transition is learned is harmful. Future local cards must use step-indexed
rather than wall-clock LR schedules. This is a diagnostic result, not a
competition submission result.

### Addendum — Bilinear Digit Cell

The cell mapped the two digit tokens to fixed ordinal scalars (`d/9`), then
used a trainable neural-arithmetic additive/multiplicative mix and a learned
decimal decoder. It reached 81.6% train exact by step 1,000 while held-out
exact peaked at 20% and ended at 10%; test loss rose from 2.93 to 19.77. The
inherited manifest capped the run at 1,000 despite the source's 4k schedule.
This is a logged configuration mismatch, but the widening train/test gap makes
more steps uninformative. Evidence: `twoA6000:results_local/gate1_bilinear_digit_cell/`.

### Addendum — repaired held-pair split

The original `(a+b) mod 5` test split withheld product values 21, 25, 54, and
56 entirely. A deterministic symmetric 20-pair split was generated instead;
all its product values occur in the 80 training pairs. The unchanged
Transformer now peaked at 45% test exact at step 300 before decaying to 25%
at step 1,000, while the Bilinear Digit Cell peaked at 30% and also ended at
25%. This isolates a real but incomplete learned pair relation and rejects
the NALU-style bias as an improvement. Generator:
`solving/research/generate_digit_product_seen_outputs.py`; evidence:
`twoA6000:results_local/gate1_{digit_product_seen_outputs_transformer,bilinear_digit_cell_seen_outputs}/`.

### Addendum — fixed-step schedule control

The repaired-split Transformer was loaded unchanged through a local-only
wrapper; only its scheduler changed from elapsed-time warmup/cosine to a
1,000-step warmup/cosine. It peaked and finished at 25% test exact, compared
with the wall-clock baseline's 45% peak. It reached 100% train exact by step
200, so the issue is early memorization, not late-clock decay. This refutes
this naive fixed schedule, while giving local work a reproducible step-indexed
reference. Source: `solving/research/digit_product_step_schedule.py`; evidence:
`twoA6000:results_local/gate1_digit_product_step_schedule/`.

### Addendum — slow fixed warmup

Increasing only fixed warmup from 100 to 400 steps recovered the old baseline's
early regime (40% test exact at step 200), then fell to 25% by step 1,000 after
train exact reached 100%. This confirms that wall-clock scheduling was not the
source of the relation failure: a calibrated step schedule can reproduce the
temporary signal, but not retain it. Evidence:
`twoA6000:results_local/gate1_digit_product_step_schedule_w400/`.

### Addendum — pairwise product-and-carry scan: routing is insufficient

**Question.** Can correct schoolbook geometry let a learned local digit-product
and carry rule compose from small to longer operands? This is a local
diagnostic only: random initialization, synthetic labels, and no weights or
code path promoted to a competition submission.

**One changed mechanism.** Each of nine LSD-first digit pairs passed through a
single shared learned MLP; its feature was added to the fixed schoolbook output
column `i + j`. A shared GRU scanned those five columns LSD to MSD and emitted
a sixth flush digit. Geometry was fixed; pair values, carry state, and decimal
decoding were learned. No product, modulo, division, lookup table, or
arithmetic oracle occurs in the model.

**Data / metric.** Training had all 10,000 pairs `a,b in 0..99`, zero-padded
to three digits. The 2,000 held-out examples had both operands in `100..999`.
The target was all six LSD-first decimal product digits, leading zeros retained;
the metric was full six-digit exact sequence match.

**Result.** A 64-wide model, seed 74, and 8,000 fixed optimizer steps (246.7 s
on `twoA6000`) reached **0.25%** held-out exact at its peak (step 2,000) and
**0.05%** final, while train-batch exact rose to 24–30%. Held-out loss briefly
fell from 2.83 to 1.97, then rose to 2.32. Peak and final weights were saved.
Evidence: `twoA6000:results_local/product_scan_length_ood/`; exact source and
manifest: `solving/research/{pairwise_product_carry_scan.py,generate_product_scan_data.py,product_scan_length_ood_manifest.json}`.

**Classification.** Refuted for this mechanism. Correct fixed column routing
does not identify a reusable digit-product law; learned pair features still
support a short-operand fit without length composition. The next useful gate
is a narrow test of local-law identifiability, not full modular squaring.

### Addendum — learned pair table: primitive addressing is not composition

**One changed mechanism.** The shared pair MLP above was replaced by a random
learned categorical table: each token pair addressed one learned vector, which
was routed and carried exactly as before. It does not encode digit products;
its only bias is a stable, separate parameter slot for each observed pair.
Data, seed, schedule, routing, carry GRU, and exact six-digit length-OOD metric
were unchanged.

**Result.** The table reaches 98.4% train-batch exact by step 2,400 and 100%
thereafter, versus only 24–30% for the MLP pair features. Held-out 3-digit
exact improves from 0.25% to **1.30% peak** (step 1,800), but ends at **0.80%**
after 8,000 steps (182.8 s). Test loss rises monotonically after the early peak,
from 2.03 at step 1,000 to 4.98 final. Peak and final weights are retained at
`twoA6000:results_local/product_table_scan_length_ood/`.

**Classification.** Partial but insufficient. Stable local pair addresses fix
short-operand fitting and produce a fivefold OOD peak lift, showing that the
MLP representation was part of the optimization problem. Perfect short-length
fit still does not imply algorithmic length composition. The remaining problem
is forcing the table/carry system to use its local states compositionally
rather than exploit correlations specific to operands below 100.

### Addendum — one-short-operand curriculum: long columns transfer

**One changed variable.** The learned pair table, fixed `i + j` routing, carry
GRU, model width, seed, schedule, and held-out metric were unchanged. Training
support expanded from both operands `<100` to all `a,b in 0..999` where at
least one operand is `<100` (190,000 pairs). Thus nonzero pair features occur
in columns 0–3 during training; the high-by-high column remains absent. Test
still contains 2,000 pairs with both operands `100..999`.

**Result.** The same 64-wide model reached **11.25% peak** exact at step 7,400
and **11.15% final** at 8,000 steps (161.4 s), versus 1.30% / 0.80% under the
short-only curriculum. The curve was stable from step 5,600 onward rather than
an early overfit spike; final held-out loss was 1.76. Train-batch exact was
96.9% at the endpoint. Peak and final checkpoints are at
`twoA6000:results_local/product_table_one_short_ood/`.

**Classification.** Confirmed, but still insufficient. Exposing nonzero
long-column states yields an 8.7× OOD lift and removes the catastrophic
annealing collapse. This isolates a concrete compositional gap: the model can
reuse a learned carry process across columns it has seen, but it has not learned
the untrained high-by-high interaction purely from lower-column examples.

### Addendum — full-position baseline: learned composition is real

**One changed variable.** The pair-table model and all optimization settings
were fixed. The one-short-operand training support was replaced by 190,000
uniform random operand pairs from `0..999`; 2,000 distinct test pairs with both
operands `100..999` were excluded from training. Thus every local digit pair
and every output-column position occurs in training, while full input pairs
remain held out.

**Result.** Held-complete-pair exact rose steadily to **58.55% final/peak** at
step 8,000 (155.6 s); train-batch exact was 67.6%, and test loss 0.336. This
compares with 11.25% under the one-short curriculum and 1.30% under short-only
training. Peak/final checkpoints are at
`twoA6000:results_local/product_table_full_position_ood/`.

**Classification.** Confirmed: the learned pair-table plus recurrent carry scan
has a real compositional multiplication regime when it receives positional
interaction support. It is not yet an exact algorithm—the 41.45% residual
failure is too large—but the sharp support response makes local-product
coverage, rather than carry recurrence, the active target for the next card.

### Addendum — 32k step-horizon control: optimization closes most residual

**One changed variable.** The full-position pair-table setup was unchanged;
only the fixed optimizer horizon and proportionate warmup/cosine schedule were
stretched from 8,000/400 to 32,000/1,600 steps. The local 600-second wall
budget stopped the job at 31,000 steps.

**Result.** Held-complete-pair exact rose from the 8k card's 58.55% to
**94.85% peak** at step 30,500 (94.5% at the last evaluation, step 31,000).
The train batch was 95.7% exact at the endpoint; held-out loss was 0.0503.
Peak and final weights are retained at
`twoA6000:results_local/product_table_full_position_32k/`.

**Classification.** Confirmed: the residual in the 8k full-position card was
mostly an optimization-horizon limitation, not a hard representational ceiling.
This is strong local evidence for a learned digit-product-and-carry primitive,
but it is still a diagnostic multiplication task and not a competition card.

**Peak-checkpoint audit.** On the saved step-30,500 checkpoint, per-digit
accuracy (LSD to MSD) was `[100.0, 100.0, 98.7, 96.1, 99.6, 100.0]%`.
Of 2,000 examples, first incorrect digit was 2 for 26 rows and digit 3 for 72
rows; no row first failed at digits 0, 1, or 5. The two middle columns have the
largest fixed fan-in (three and two pair contributions), making aggregate
column representation—not the carry flush—the evidenced next target.

### Addendum — fan-in tag: explicit count metadata is harmful

**One changed mechanism.** A learned vector keyed by the fixed column fan-in
`[1, 2, 3, 2, 1]` was added to the otherwise unchanged pair-table columns.
It gave no arithmetic outputs or answers; it only declared how many learned
pair terms had entered each column.

**Result.** Under the identical full-position data and 600-second horizon, the
tagged model peaked at **84.75%** held-complete-pair exact (step 31,500), versus
94.85% for the untagged control. Final test loss was 0.1282 compared with the
control's 0.0503. Evidence and retained checkpoints:
`twoA6000:results_local/product_table_fanin_32k/`.

**Classification.** Refuted. The model already encodes enough column position
through its recurrent scan; explicitly injecting count categories slows the
learned mapping and does not repair central-column errors. Future cards should
alter the representation of the *set of pair contributions*, not add metadata
about its cardinality.

### Addendum — learned intra-column fold: preserve pair interactions

**One changed mechanism.** The pair table, carry GRU, data, seed, and 32k-step
schedule were fixed. Only the within-column aggregator changed: instead of
summing pair vectors, a shared GRU folded the one, two, or three pair vectors
in each fixed schoolbook column into one learned column state before the carry
scan. No arithmetic operation, target, or oracle was added.

**Result.** The fold reached **97.60% peak** exact at step 23,000 (97.4% final
at the 600-second cap), exceeding the sum control's 94.85% peak. It learned
faster as well: 70.7% at step 4,500 versus the control's 56.9%, and 91.95% at
step 11,000 versus 82.55%. Evidence and peak/final checkpoints:
`twoA6000:results_local/product_table_fold_32k/`.

**Peak-checkpoint audit.** Per-digit accuracy (LSD to MSD) was
`[100.0, 100.0, 99.9, 97.75, 99.9, 100.0]%`. First-error counts across 2,000
rows were `[0, 0, 2, 45, 1, 0, 1952]`; the sum control had `[0, 0, 26, 72, 5,
0, 1897]`. Thus the fold removes most, though not all, of the central
multi-contribution column failure.

**Classification.** Confirmed. Learned aggregation of pair contributions is
strictly better than raw summation on this diagnostic, producing the first
near-exact legal learned digit-product primitive. This clears the local product
gate sufficiently to test reuse of the primitive under a multi-step learned
composition, still locally and without promoting weights to a submission.

### 2026-07-23 — Soft digit-state squaring recurrence, held T
- **Hypothesis:** A learned digit-pair table, learned within-column fold, and
  learned carry scan can be reused as one shared squaring cell. Keeping each
  digit as a soft ten-way distribution should let gradients pass through the
  repeated cell and generalize from T=1,2 to held-out T=3.
- **Setup:** Train on bases 0..9 at T=1 and T=2. The target is the four
  least-significant decimal digits of x^(2^T), least-significant first. The
  shared cell was unrolled three times according to T; no checkpoint,
  arithmetic operation, or target-derived feature enters the model. The
  checkpoint with the best narrow held-T score was audited on every base 0..9
  at T=3.
- **Result:** The narrow T=3 split (bases 0..3) reached 100% exact by step 600
  (59.3 seconds) and stayed saturated through step 3,000. The saved step-600
  checkpoint reached only **40.0% (400/1,000)** when evaluated on all ten bases
  at held T=3; precisely bases 0..3 passed. The widened audit originally
  exposed a generator bug: values above four digits were not truncated even
  though the task was defined as a four-digit state. The final audit labels are
  explicitly x^(2^T) mod 10^4.
- **Next:** Hold the cell, data, and all-ten-base held-T metric fixed; replace
  only the soft inter-step digit state with a straight-through one-hot state.
  This tests whether accumulated fractional state, rather than the learned
  local transition, causes the third-step collapse.

### 2026-07-23 — STE-discrete digit recurrence control
- **Hypothesis:** The 40% held-T failure comes from fractional digit mixtures
  drifting under reuse. Passing a one-hot digit state between applications,
  with a straight-through gradient estimator, should recover the missing bases.
- **Setup:** The prior card's generator, seed, shared pair-table/fold/carry
  cell, optimizer, and all-base T=3 test were fixed. Only the inter-step state
  changed from the soft ten-way distribution to its argmax one-hot value. The
  output loss used the corresponding pre-projection logits so gradients still
  reach the learned cell.
- **Result:** Train exact reached 100% by step 600, but held-T exact was
  **40.0%** at every evaluation from steps 400 through 1,200. The loss on the
  held split worsened from 1.91 to 4.29 while accuracy remained the same four
  bases. The run was stopped after the plateau; peak weights are
  `twoA6000:results_local/ste_digit_recurrence/monitor_peak.pt`.
- **Next:** Do not add recurrence depth. First establish whether this learned
  cell can square a four-digit decimal state in one application under a
  held-input split. That is the exact unseen transition T=3 requires here.

### 2026-07-23 — One-step four-digit squaring gate
- **Hypothesis:** The pair-table / learned-fold / learned-carry cell can learn
  a genuine four-digit decimal square transition before it is asked to compose
  through time.
- **Setup:** All architecture and optimizer settings were fixed. A deterministic
  shuffle of x in [0, 9,999] supplied 8,000 train and 2,000 disjoint test
  values at T=1. The target was x² mod 10^4 as four LSD-first digits. The
  active STE state is not reused at T=1, so this isolates the local cell.
- **Result:** Held-input exact rose 21.9% (step 500), 47.55% (1,000), 67.8%
  (1,500), 79.95% (2,000), then peaked at **85.35%** (3,500) before falling to
  83.5% (4,000) and 83.55% (4,500). Train exact was 100% from step 2,000.
  The peak checkpoint is
  `twoA6000:results_local/one_step_four_digit_square/monitor_peak.pt`.
- **Classification:** Confirmed partial. The local learned operator strongly
  generalizes across held four-digit inputs, which explains why it was the
  real missing transition in the T=3 card. Its remaining 14.65% error makes a
  recurrence test premature.
- **Next:** Audit errors by decimal output position and change one local-cell
  mechanism only; preserve this split and peak-checkpoint convention.

### 2026-07-23 — Arity-conditioned fold initialization
- **Hypothesis:** The only remaining one-step error is decimal output digit 3,
  whose schoolbook column has four pair terms. Giving the shared fold a learned
  initial state for each arity (rather than one shared zero state) should make
  that four-term aggregation identifiable.
- **Setup:** The 8k/2k shuffled one-step split, seed, pair table, fold order,
  carry scan, and optimizer were fixed. The sole change was
  `pair_fold_initial[arity]` for arities 1..4.
- **Result:** The card was ahead early (71.2% versus 67.8% control at step
  1,500; 84.55% versus 83.4% at step 3,000), but reached only **84.55% peak**
  and was 83.8% at the matched step 3,500, below the control's 85.35% peak.
  It was stopped after that matched negative comparison. Checkpoint/logs:
  `twoA6000:results_local/arity_fold_four_digit_square/`.
- **Classification:** Refuted. The fold can infer its arity from recurrence
  length; a learned arity-specific starting vector improves early optimization
  but harms the final operator law.
- **Next:** Retain the shared fold initialization. The controlled peak audit
  still says that only output digit 3 fails, so the next card must alter the
  representation used by the carry transition at that output column.

### 2026-07-23 — Balanced-tree fold control
- **Hypothesis:** The sequential fold loses information across the four pair
  terms of output column 3. Reusing the same learned fold cell in a balanced
  binary tree should shorten that path and improve the held-input law.
- **Setup:** The 8k/2k one-step split, pair table, carry scan, fold parameters,
  seed, and optimizer were fixed. Only the fold topology changed: leaves were
  initialized as before, then combined pairwise until one column state remained.
- **Result:** The tree lagged at every matched point: 17.3% vs 21.9% at step
  500, 42.05% vs 47.55% at 1,000, 53.15% vs 67.8% at 1,500, and 61.7% vs
  79.95% at 2,000. It was stopped without spending the remaining schedule.
- **Classification:** Refuted. The ordered serial fold is a useful learned
  computation, not merely a lossy way to pool terms. Restore it for the next
  card. Logs: `twoA6000:results_local/tree_fold_four_digit_square/`.

### 2026-07-23 — Soft carry-prototype control
- **Hypothesis:** The fourth-column residual is carry-state drift. Mapping each
  learned carry transition to a soft mixture of 64 learned prototypes should
  supply the finite-state bias that worked on the carry-only gate.
- **Setup:** The serial fold control, 8k/2k split, seed, pair table, and output
  head were fixed. Only each carry GRU transition changed from its continuous
  candidate state to `softmax(selector(candidate)) @ codebook`.
- **Result:** The prototype card was far behind at every matched point: 3.9%
  (step 500), 17.15% (1,000), and **26.4%** (1,500), against the continuous
  control's 21.9%, 47.55%, and 67.8%. It was stopped at the comparison point.
- **Classification:** Refuted for this operator. A prototype state can solve
  carry normalization in isolation but prevents the learned cell from jointly
  representing local pair products and carry. Restore continuous carry.

### 2026-07-23 — Unweighted later-column auxiliary supervision
- **Hypothesis:** The fourth output digit is under-supervised. Training the
  same cell to predict later square columns as additional labels should force
  its carry state after column 3 to remain useful, improving the unchanged
  four-LSD held test.
- **Setup:** The original serial continuous cell and 8k/2k x split were
  restored. Train rows contained digits 3..6 followed by the usual digits
  0..3; test rows contained only digits 0..3, leaving the reported metric
  unchanged. The first two launch attempts ended before training due to
  state/logit shape mismatches; after correction, the actual run trained.
- **Result:** Equal-weight auxiliary labels overwhelmed the primary task:
  test exact was 0.1% at step 500 and **0%** at steps 1,000 and 1,500, versus
  21.9%, 47.55%, and 67.8% for the four-label control. The run was stopped.
- **Classification:** Refuted as configured. Extra future-column labels are
  not a free improvement; any follow-up needs an explicit primary/auxiliary
  loss weighting, which would be a distinct card.

### 2026-07-23 — Primary-weighted later-column supervision
- **Hypothesis:** The auxiliary columns need not be harmful if their loss is
  downweighted. A 0.25 weight on each later-column token and 1.0 on each of
  the normal four output tokens should regularize the carry state without
  displacing the primary task.
- **Setup:** The same auxiliary-label data and model outputs were retained.
  A custom loss weighted the four auxiliary tokens at 0.25 and primary tokens
  at 1.0. Two initial launches exposed data-format issues (equal input/label
  lengths were treated as causal LM); the final run used a one-token-longer
  prompt, producing genuine separate input/output rows. The held test retained
  only the original four digits.
- **Result:** The valid weighted run reached 24.1% at step 500 but then lagged
  the primary-only control: 45.8% vs 47.55% (1,000), 57.8% vs 67.8% (1,500),
  and **64.8% vs 79.95%** (2,000). It was stopped at that matched point.
- **Classification:** Refuted. Downweighting prevents collapse but still pulls
  capacity away from the local four-digit square law. Restore the primary-only
  serial continuous baseline before further operator work.

### 2026-07-23 — Fourth-digit weighted primary loss
- **Hypothesis:** The baseline's only held-input error is output digit 3.
  Giving that token four times the cross-entropy weight, while retaining the
  original four-label data and exact metric, should repair the local column.
- **Setup:** The primary-only serial continuous baseline and 8k/2k split were
  restored. The only experimental change was per-token loss weights
  `[1, 1, 1, 4]` during training; evaluation used ordinary unweighted loss.
- **Result:** The card trailed control throughout: 19.8% vs 21.9% (500),
  43.25% vs 47.55% (1,000), 64.05% vs 67.8% (1,500), and **74.55% vs
  79.95%** (2,000). It was stopped at the matched point.
- **Classification:** Refuted. The digit-3 residual is not caused by too little
  direct loss signal; forcing more gradient into that token impairs the shared
  transition. Restore equal primary weights.

### 2026-07-23 — Symmetry-tied digit-pair table
- **Hypothesis:** Decimal digit multiplication is commutative. Enforcing a
  symmetric learned pair table should reduce local-law degrees of freedom and
  improve held four-digit squaring.
- **Setup:** The serial continuous primary-only baseline and 8k/2k split were
  fixed. The only change was using `(table + table.T) / 2` for every pair-table
  lookup in the squaring cell.
- **Result:** The symmetry tie severely underfit held inputs: 18.0% vs 21.9%
  (500), 37.5% vs 47.55% (1,000), 44.2% vs 67.8% (1,500), and **50.55% vs
  79.95%** (2,000). It was stopped at the matched point.
- **Classification:** Refuted. Although the numeric relation is symmetric, the
  ordered serial fold uses orientation-specific learned features. Do not tie
  the table for this mechanism.

## 2026-07-24 — PR #1 STE token bottleneck (Easy e5 + Medium m5)

**Question:** Does forcing each UT loop (except last) through a discrete vocab STE snap+re-embed beat continuous residual UT (`depth_d32_k4_ut_optsched`)?
**What we did:** Reviewed/validated PR #1 `chatgpt/ste-token-bottleneck`; L40S smoke; hosted e5 + m5.
**Result:** [SOURCED] e5 mean **0.50%** (test 0.70 / ood 0.30, 2521 steps) vs UT K4 e5 **1.00%**. m5 mean **0.17%** (test 0.10 / ood 0.20, 58525 steps) ≈ optsched m5 0.17–0.20%. Jobs `92e064ea` / `fac54972`.
**Dead ends:** Mid-loop full-sequence token snap does not improve e5 or m5 under this recipe.
**Lesson:** Discrete bottleneck between tied loops is legal and trains, but this form is not an Easy/Medium win vs continuous UT.
**Promote?:** No — keep as negative card; do not merge as active shortlist without a different bottleneck design.

### 2026-07-24 — Upstream competition sync `2c56499` → `79f0a09`
- **Hypothesis:** Specs and handoff packet still describe the pre-change evaluator (mean-exact Hard rank, old rule numbering).
- **Setup:** `git pull --ff-only` in gitignored `competition/` clone; diffed README / `benchmark/runner.py` / `data/squaring_mod.py` / `service/competition.py`.
- **Result:** Four upstream commits: (1) submission deadline Aug 31 10pm PT; (2) Rules expanded to 1–16 (trainable-param ceiling; bans on hard-coded weights/algorithms, broken autograd, CPU offload; solvers/data inspection now Rule 14); (3) scoring based on T-extrapolation — Hard ranks certified Max T then OOD N Max T on ladder {1,2,4,8,16,32,64}; Easy/Medium keep mean exact % and gain Max T diagnostics; (4) depth / ood_n depth profile generation in `squaring_mod.py` + runner certification.
- **Next:** Treat historical Hard mean% as legacy; prefer Max T when re-submitting Hard. Local/GPU clones must pull the same pin before smoke. Specs refreshed: `solving/handoff/PRIMARY_SOURCES.md`, `learnings/concepts/{01,03,07,09,14}`, `RESEARCH_PROTOCOL.md` §6, `SCOREBOARD.md`, `STATUS.md`.


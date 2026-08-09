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
## 2026-07-25 — Held-u, held-N one-step campaign — Codex

- **Question:** Can a legal, non-recurrent model clear `u² mod N` before
  returning to T-extrapolation?
- **Pair/N interaction:** Four distinct Transformer blocks with learned
  all-pairs u-digit features attending to N. e5 job `8caf7fa8`: train 94.5%,
  test 0.4%, OOD 1.0%, mean 0.71%. **Memorization collapse.**
- **Per-column carry auxiliary:** Same model plus first-square carry-in/out
  supervision. e5 job `faaa00f3`: train 92.6%, test 0.4%, OOD 0.7%, mean
  0.54%. **Refuted at Easy budget.**
- **T=1-only objective:** Same pair/N model, gradients only from e5 T=1 rows.
  Job `5860d424`: aggregate train 29.1%, test 0.5%, OOD 0.7%, mean 0.58%.
  **Refuted: isolated one-step training still memorizes.**
- **Multi-block supervision:** Average shared-head logits after every distinct
  block. Job `dc7c0746`: train 91.8%, test 1.1%, OOD 1.0%, mean 1.04%.
  **Provisional positive, not confirmed** because it lies inside known e5
  noise. Exact replication was pre-registered but the service rejected it
  while Hard was running.
- **Hard:** Exact validated multi-block file launched at 23:55 UTC, job
  `de4c3c51`; predicted certified Max T=0, with possible T=1 partial gain.

### 2026-07-30 — Task B N-broadcast semantic ablation
- **Hypothesis:** The two-N held-out-u collapse reflects an inadequate learned route from N digits to output slots; broadcasting pooled N states after every Transformer layer should help, while a wrong-N broadcast should not.
- **Setup:** Fixed unpaired N={1349,1357} Task-B data (8k train / 2k held-out u), 4L d128 standard Transformer, seeds 0–2. Compared baseline (799,498 params) to correct and shuffled input-N broadcast (865,546 params); all new variants passed direct-forward and 32-row 100% smoke checks. Counterfactual evaluation used the same 2,000 u under both N, excluded from either train set.
- **Result:** **Refuted (Case C).** Final held-out-u EM: baseline 11.27±0.63%, correct broadcast 11.55±0.30%, shuffled control 11.35±0.39%. Counterfactual predictions change with N 92.55%/95.10%/95.02%, respectively, but correct modulus-specific pairs are 0% in every condition; ~92–95% respond yet are wrong under both. N broadcast is slower (95.5 vs 112.0 steps/s) and does not provide a material gain.
- **Next:** Diagnose fixed-N=1349 generalization. Its final held-out-u EM is 36.60%, q≥10 is 33.9%, and a crude nearest-multiple baseline reaches 39.7%; test a quotient-estimation intervention only after comparing it to an equally sized non-quotient auxiliary control.

### 2026-07-30 — Task B fixed-N quotient auxiliary control
- **Hypothesis:** Fixed-N unseen-u failure is primarily missing quotient estimation; a four-digit quotient auxiliary head should improve held-out-u reduction, unlike an equal-size u-copy target.
- **Setup:** N=1349, 8k/2k disjoint-u data, same 4L d128 Transformer and budget, seeds 0–2. Baseline vs 1,290-parameter quotient-digit auxiliary (λ=.25) vs same-head u-copy control. All new variants passed manual q checks, direct autograd, and 32-row 100% smoke.
- **Result:** **Refuted.** Final held-out-u EM: baseline **33.65±2.95%**, quotient aux **29.38±6.91%**, u-copy control **22.35±0.40%**. Fixed-N error diagnostic: 1/2/3/4 quotient-digit EM=94.4/39.2/38.0/24.6%; 82.6% of wrong rows have a contiguous error run. Copy/interpolation baselines do not explain predictions; nearest-multiple heuristic reaches 39.7% but model agreement is only 11.0%.
- **Next:** Stop the parallel standard-Transformer branch. The remaining supported hypothesis is a missing learned serial remainder/borrow state; any next Task-B run must introduce a learned serial workspace plus a capacity-matched non-serial control.

### 2026-07-30 — Task B serial workspace vs depth control
- **Hypothesis:** A learned serial workspace, reused eight times while cross-attending to immutable N/u context, will improve fixed-N held-out-u reduction—especially at large quotients—over both the existing baseline and parameter-matched deeper Transformer.
- **Setup:** N=1349, 8k/2k disjoint-u data, three seeds. Baseline 4L d128 (799,498 params); deep control 5L d120 (877,810); recurrent workspace K=8 d144, two context layers, eight registers and a tied self/cross-attention transition (845,434). Same optimizer/budget; direct-autograd and 32-row smoke passed.
- **Result:** **Refuted for this workspace formulation.** Peak/final held-out-u EM: baseline 34.75±2.35 / 33.65±2.95; deep 38.23±4.67 / 35.65±5.56; recurrent 29.23±5.84 / 23.47±7.08. Recurrent fits train (~97.20% final) and improves monotonically as its evaluated depth rises to 8, but remains worse on q>=10 (20.2% vs baseline 30.8%) and four-digit quotient (15.3% vs 22.5%).
- **Next:** Do not K-sweep: recurrence failed its gate. The targeted next test is input-conditioned versus shuffled-context workspace initialization with K=8 held fixed, to separate fixed-state representation from tied-transition capacity; no auxiliary labels.


### 2026-08-03 — Task B input-conditioned workspace initialization
- **Hypothesis:** The K=8 tied workspace failed because its registers began as
  a fixed learned state. One input-derived cross-attention read should make
  that state semantically useful; a matched read from a different, fixed
  dataset row should not.
- **Setup:** Fixed N=1349, 8,000 train / 2,000 disjoint held-out-u examples,
  three seeds per condition, d=144, two context layers, eight registers, K=8,
  batch size 512, and the same 50k-step/early-stop budget. Both conditions
  use two ContextEncoder calls and the tied transition's existing
  cross-attention parameters; only the source row of the initialization read
  differs. Source commit `3cde93d`; six runs completed on oneL40.
- **Result:** **Confirmed.** Ordered input context reached **39.22±3.33%**
  held-out-u exact (35.45%, 40.45%, 41.75%), compared with 33.65±2.95% for
  fixed K=8 registers and 35.65±5.56% for the deep non-recurrent control.
  The matched row-stable shuffled control reached only **14.67±4.43%**
  (18.65%, 15.45%, 9.90%). All six training and evaluation commands exited 0.
  Evidence: `diagnostics/analysis_out/task_b_workspace_init_phase1/`.
- **Interpretation:** This is semantic information use, not a generic extra
  attention/encoder-pass benefit: correct context beats both controls while
  the same mechanism with incorrect context is much worse. The registered
  `q >= 10` subprediction remains untested, because the saved evaluator uses
  relative small/mid/large quotient buckets rather than that absolute slice.
- **Next:** Add the absolute quotient slice to evaluator-only analysis of these
  retained checkpoints. Keep K, transition, and optimizer fixed until that
  analysis decides whether the gain is where predicted.

### 2026-08-03 — Upstream competition sync `79f0a09` → `8a3c78d`
- **Hypothesis:** Specs still describe the Max-T-only Hard tie-break and flattened-only custom loss from `79f0a09`.
- **Setup:** `git pull --ff-only` in gitignored `competition/` (was `79f0a09`); GPU box `oneL40` already at tip. Diffed README / `benchmark/{api,runner,validation}.py` / `service/{db,views}.py` / `client/cli.py`.
- **Result:** One upstream commit. Adds `TokenLossBatch` + mutually exclusive `token_training_loss` (legacy `training_loss` still valid). Hard ranking becomes Max T → OOD N Max T → next-rung exact accuracy (`seen_tiebreak_accuracy_percent` / `ood_n_tiebreak_accuracy_percent`) → earlier time. Rule 4 clarifies outer loop vs in-model recurrence/TRM/optimizer curvature. Rule 14 bans data augmentation. README notes closed beta Jul 31–Aug 2 and Monday Aug 31 10pm PT deadline.
- **Next:** No forced submission rewrites (active cards keep `training_loss`). Prefer `token_training_loss` only when sequence-level loss is the one variable. Specs refreshed: `solving/handoff/PRIMARY_SOURCES.md`, `learnings/concepts/{01,03,09,14}`, `RESEARCH_PROTOCOL.md` §6, `SCOREBOARD.md`, `STATUS.md`, `ASSUMPTIONS.md`.

## 2026-08-04 — Session open (Author: Codex)

- Current branch/commit: `main` / `fab088c5c5219500d31851234f8628170f3dd8ac`.
- Competition upstream commit or live-rule timestamp: pending read-only audit.
- Question being investigated: where learned fixed-`N=1349` decimal reduction first fails across the B0–B4 capability ladder.
- Approved experiment: initial action was a rule/upstream audit only; the
  bounded Task B capability ladder follows under the executor-prediction rule.
- Expected result: executor-authored in `solving/experiments/predictions.md`.
- Maximum compute: none during audit.
- Files permitted to change: factual audit/session records only; no competition-facing code.

### 2026-08-04 — Rule audit result (Author: Codex)

- **Setup:** Compared the unchanged local checkout
  `8a3c78d6eae4047b07cd8c617c1b311f544a0e9f` to fetched upstream
  `e32c2f985f8ed4107c96d00271448777954ecc0c` (`2026-08-03T23:44:35Z`). Read
  the live problem page on 2026-08-04. No competition checkout or submission
  source was modified.
- **Result:** Upstream adds bounded evaluator-owned multi-backward-pass and
  same-batch-reuse APIs, and forbids participant-initiated derivative entry
  points/nested training; scoring and data generation are unchanged. Static
  scan found no newly forbidden calls in local submission sources. Existing
  validation artifacts are stale against the new pin. Full file/API/rule diff:
  `solving/RULE_AUDIT_2026-08-04.md`.
- **Status correction:** Completed Hard Fable v2 is #19/19 on the live Hard
  board, with no T=1 certification; earlier `#11 at 0.03%` text was historical.
- **Next:** Run the executor-predicted fixed-`N` ladder; retain the audit as
  a scope/legality record.

### 2026-08-04 — Prediction ownership change (Author: Codex)

- **Decision:** User authorized the experiment executor, rather than the
  human, to write pre-registered predictions and continue the bounded task
  sequence autonomously.
- **Protocol change:** `RESEARCH_PROTOCOL.md` §1 now requires an
  executor-authored prediction before a run; it records the mechanism and
  falsifier and can be amended by the human. The one-variable and result rules
  remain unchanged.

### 2026-08-04 — Task B retained-checkpoint depth audit (Author: Codex)

- **Question:** Does the ordered-input K=8 workspace use additional tied steps
  for difficult reduction cases, or is its reported gain only ordinary depth?
- **Setup:** Evaluated the same fixed-`N=1349` held-out-u rows at override
  `K={1,2,4,8}` for the three saved ordered-context **peak** checkpoints.
  Weights, initialization context, rows, and output decoding were unchanged.
  Reports: `diagnostics/analysis_out/task_b_workspace_depth_audit/seed{0,1,2}.json`.
- **Result:** Exact match increased at every depth: K=1 **0.63±0.38%**, K=2
  **5.37±3.07%**, K=4 **36.93±1.78%**, K=8 **55.98±4.23%**. The `q>=4` bucket
  likewise climbed **0.53±0.41% → 4.29±2.49% → 34.35±1.43% → 53.87±4.41%**.
  Aggregated repaired/broken counts were 295/11 (1→2), 1,934/40 (2→4), and
  1,375/232 (4→8).
- **Classification:** Confirmed, bounded. Extra tied steps help difficult rows
  and performance does not peak before K=8. This is not evidence that the
  transition has learned the mathematical remainder recurrence: every tested
  K is inside the training architecture's finite range.

### 2026-08-04 — Task B capability-ladder budget amendment (Author: Codex)

- **Setup:** The planned 15-cell, 20,000-update seed-0 screen was started on
  the L40 and stopped before its first logged metric: it had not reached step
  200 after more than two minutes. No result was recorded or interpreted.
- **Decision:** All ladder cells now use a uniform **2,000-update** screening
  horizon with no early stopping. This preserves comparisons within the ladder
  while avoiding an all-day wall-clock confound; batch size, optimizer, data
  sizes, architectures, and seed remain fixed.

### 2026-08-04 — Task B L40 execution record (Author: Codex)

- **Instance:** Prime Intellect pod `025c2d0974444e979138da8ed4627d23`,
  name `somn-l40`, one L40 48 GB, 14 vCPU, 72 GB RAM, 100 GB disk, created
  2026-08-04 11:20:52 UTC.
- **Remote paths:** source `~/somn-taskb/diagnostics`; virtual environment
  `~/somn-venv`; ladder artifacts `~/somn-taskb/runs/reduction_ladder`;
  queue log `~/somn-taskb/reduction_ladder_screen.log`.
- **Durability plan:** sync every completed run directory to
  `diagnostics/artifacts/somn-l40-2026-08-04/` before provider termination;
  checkpoints remain outside Git.

### 2026-08-04 — Task B capability ladder, seed-0 screen (Author: Codex)

- **Data:** fixed `N=1349`, 800 train / 256 validation / 256 held-out-u rows,
  split-disjoint operands; B0=`q=0`, B1=`q=1`, B2=`q=2-3`, B3=`u=x^2` for
  disjoint `x`, B4=the existing broad sampler. Every cell used 2,000 updates.
- **Models:** matched standard Transformer, five-layer deep control, and
  ordered input-context K=8 recurrent workspace. Exact-match held-out-u is
  the decision metric; token accuracy and quotient buckets are diagnostic.
- **Result:** B0: all 100.00%; B1: 98.83/99.61/98.83%; B2:
  96.88/96.09/92.58%; B3: 0.39/0.00/0.39%; B4: 20.70/16.80/18.36%
  (standard/deep/recurrent order). In B4, q=0 constitutes 41/256 held-out
  rows and is 95.12/92.68/97.56% correct, while q>=4 is only
  6.57/2.35/3.29% correct.
- **Classification:** Refuted the pre-registered B3-easier-than-B4 prediction.
  The direct gate is not copying or a one-to-three-subtraction primitive; it
  is general high-quotient reduction. B4's aggregate exact-match is not a
  valid high-q success because its composition is q=0-heavy.
- **Artifacts:** all 15 reports/metrics are synced under
  `diagnostics/artifacts/somn-l40-2026-08-04/reduction_ladder/`; resumable
  checkpoint copying is separate from the lightweight report backup.

### 2026-08-04 — Quotient-balanced broad reduction, seed 0 (Author: Codex)

- **Question:** Is B4's apparent broad-u gain a real high-quotient reduction
  capability, or an aggregate artifact of a q=0-heavy split?
- **Change:** Only the operand distribution: B5 has 200/800 training and
  64/256 held-out examples in each q bucket `0`, `1`, `2-3`, and `>=4`.
  Fixed `N=1349`, disjoint operands, standard baseline, and 2,000 updates are
  otherwise identical to the ladder screen.
- **Result:** 47.27% held-out exact (train 82.37%). Bucket exact is q=0
  87.50%, q=1 59.38%, q=2-3 42.19%, and q>=4 **0.00%**.
- **Replication result:** Seeds 0/1/2 have 47.27/48.05/51.56% held-out exact.
  The q>=4 bucket is exactly **0.00% in all three**; q=0 is
  87.50/89.06/92.19%, q=1 59.38/57.81/67.19%, and q=2-3
  42.19/45.31/46.88%.
- **Classification:** Confirmed across three seeds. It narrows the unsolved
  gate to reduction beyond three modulus subtractions, not a seed-sensitive
  aggregate or B4 composition artifact.

### 2026-08-04 — Prior learned-reduction-cell forensic gate (Author: Codex)

- **Candidate located:** `solving/research/pure_reduction_cell_v2.py` and
  `generate_pure_reduction_v2.py`, introduced together in commit
  `60e87717403e46ff57fb5ea66f3dacd5152ea8ef` (2026-07-25 02:37:02 +0100).
  The historical 78.45% claim refers to this *named* v2 card, rather than the
  earlier v1: the source difference is weight decay `0.01 -> 1.0`, while the
  v2 generator changes the direct 8-digit operand distribution to
  reciprocal/log-uniform sampling at fixed `N=323`.
- **Mechanism reconstructed from source:** four learned pieces only: digit
  embeddings, an 8-step GRU state sweep over MSB-first operand digits,
  attention from state to the three supplied `N` digits, and a soft learned
  quotient embedding followed by a learned GRU update. The final continuous
  state feeds three digit heads. No modulo, comparison, handwritten quotient,
  subtraction loop, or lookup table occurs in `forward`; the fixed loop count
  is input-digit length (eight), not a quotient-dependent control flow.
- **Historical setup claimed in prose:** 8,000 reciprocal-sampled direct
  8-digit operands / 2,000 disjoint held-out operands; fixed `N=323`; final
  three-digit `P mod N` supervision; reported 78.45% peak at step 26,200 and
  a 69–84% late window, with a prose-only 75.5% `P>=323` check.
- **Blocking reconstruction result:** the only committed candidate has
  `TOTAL_STEPS=20,000`, while the claim requires 80,000 steps. There is no
  committed 80,000-step source/config, command, generated split, raw metric,
  checkpoint, or checkpoint hash (`find`/repository-wide search found none).
  `diagnostics/analysis_out/task_b_canonical_matrix.md` independently labels
  these raw artifacts missing. Therefore the 78.45% number is not reproduced
  evidence and cannot identify an exact runnable mechanism/config.
- **Decision:** Phase 2 reproduction and every N=1349 port/control are
  stopped by the required gate. Re-running a guessed 80,000-step variant would
  be a new, bundled experiment, not a reproduction. Historical rows remain
  unchanged; this is a labeled correction to their evidentiary status.

### 2026-08-04 — Learned reduction-cell new reimplementation, N=1349 (Author: Codex)

- **Classification:** **NEW REIMPLEMENTATION — NOT A REPRODUCTION.** This run
  tests the architecture form only; it neither confirms nor revises the
  unverified historical 78.45% claim.
- **Interface-only port:** `diagnostics/models/learned_reduction.py` preserves
  learned decimal digit embeddings, an eight-transition GRU sweep over the
  eight MSB-first `u` digits, learned attention over supplied `N` digits, soft
  learned quotient embedding, learned state update, and four learned output
  heads. The necessary interface changes are 3→4 `N` digits, 3→4 output
  digits, and the current 19-token Task B prompt. Width remains 128; no
  Abacus, auxiliary target, state discretization, or streaming variation was
  added.
- **What is structural versus learned:** decimal prompt layout and exactly
  eight state transitions are structural. Every state transformation,
  attention, quotient representation, and output mapping is learned. The
  eight transitions are tied to input-digit count—not quotient magnitude or
  an explicit repeated-subtraction count—so the architecture description alone
  does not imply scaling with q.
- **Data and command:** B5 quotient-balanced fixed `N=1349`, 800 train / 256
  validation / 256 independent held-out operands, 64 rows in each q bucket.
  L40 command: `python train.py configs/mod_fixed_n_learned_reduction.yaml
  --override out_dir=/home/ubuntu/somn-taskb/runs/learned_reduction_b5/seed0`.
  Seed 0; AdamW settings, batch 512, and 2,000 updates match B5 baseline.
- **Result supported by this run:** 242,098 parameters; 45–51 steps/s;
  65.87% train exact and 3.12% held-out exact. Held-out exact by quotient is
  q=0 7.81%, q=1 1.56%, q=2-3 3.12%, q=4-9 0.00%, q=10-99 0.00%, and q>=4
  0.00%. Output-position accuracy is 79.30/30.08/32.42/24.22%. Error runs
  average 2.16 digits. Run bundle (raw metrics, report, source snapshot via
  config, and checkpoints) is durable at
  `diagnostics/artifacts/somn-l40-2026-08-04/learned_reduction_b5/`.
- **Comparison:** B5 standard baseline is 47.27% held-out with q>=4 0.00%
  across three seeds and has 799,498 parameters; its q=0/1/2-3 are materially
  higher. Existing deep (877,810 parameters) and input-context workspace
  (845,434) controls were not rerun on B5 because D failed the registered
  nonzero-q>=4 screening gate; their B4 values are not used as a comparison.
- **Classification:** Refuted. The port fails the promotion rule, so no extra
  seeds, digit-order control, state-init control, or parameter-matched control
  is justified. Those controls would diagnose an already-killed formulation,
  not change the q>=4 decision.

### 2026-08-04 — Scalable iterative-reducer hypothesis (Author: Codex)

- **User-supplied working theory:** the controlled B5 results support a fixed
  effective-computation failure, not a general inability to learn decimal
  modular operations: q=0 through q=3 can be learned, while q>=4 is zero.
- **Legal reformulation:** the literal program `s <- s-kN until s<N` cannot
  appear in a model forward because it would be a handwritten reduction loop.
  Phase 1 is therefore a diagnostic-only learned transition, trained on
  generator-produced trace targets. The evaluator supplies q as an unroll
  depth only to test composition; this is not valid submission inference.
- **Pre-registered next card:** one tied learned `(state,N)->next-state` cell,
  tested at q=0,1,2,5,10,50,100. It separates reusable-step accuracy from
  adaptive halting. Learned halting, digit-order, state-init, and
  parameter-matched controls remain separate cards, contingent on a signal.

### 2026-08-04 — Teacher-depth iterative reducer, seed 0 (Author: Codex)

- **Classification:** **NEW DIAGNOSTIC — NOT SUBMISSION-RELEVANT.** The
  evaluator supplies q only as an unroll count, so this cannot be used for
  competition inference.
- **Mechanism:** one 400,906-parameter tied Transformer transition maps the
  decimal digits of `(current state,N)` to the next-state digits. Training
  rows are generated trace targets at depths 0,1,2,5,10,50,100 for 800
  remainders; 256 remainders are disjoint for self-fed terminal evaluation.
  The forward contains no arithmetic reduction operation.
- **Result:** at 2,000 updates, teacher-forced loss is 0.000128 and throughput
  139.8 steps/s. Greedy terminal exact is q=0 100.00%, q=1 100.00%, q=2
  100.00%, q=5 72.27%, and q=10/50/100 **0.00%**.
- **Classification:** Refuted the primitive-before-halting formulation at
  q>=10. This supports the updated theory only in part: adding tied depth can
  reach beyond q=3, but high one-step accuracy is not enough for scalable
  execution because self-generated digit errors compound. Learned halting is
  not promoted: it would choose when to stop an already-collapsed trajectory,
  not repair state fidelity.
- **Artifacts:** metrics, report, config, and checkpoint are durable at
  `diagnostics/artifacts/somn-l40-2026-08-04/teacher_depth_reducer/`.

### 2026-08-04 — Working-theory correction: stable recurrent execution (Author: Codex)

- **Supported by new run:** fixed effective depth is necessary but not
  sufficient. A tied learned transition reaches q=5, yet fails at q>=10 while
  its teacher-forced loss is near zero. The next bottleneck is state fidelity
  under self-fed rollout: local correctness does not compose.
- **Not supported:** learned halting as the next intervention. Halting selects
  a time on a trajectory; it does not restore an invalid state. It remains
  deferred until a state representation/correction mechanism survives a long
  rollout.
- **Next isolation:** a no-training rollout-drift audit records true-state
  and self-fed fidelity at every step. Representation, quantization,
  correction, chunking, and hybrid-memory proposals remain hypotheses until
  that audit distinguishes their target failure shape.

### 2026-08-04 — Correction: anchor-depth reducer did not isolate drift (Author: Codex)

- **Audit result:** the saved q=10 rollout has teacher-forced exact 100.00%,
  98.05%, 58.20%, then 0.00% at steps 1–4; free rollout is 100.00%, 98.05%,
  56.25%, then 0.00%. The free path diverges slightly at step 3, but the
  decisive collapse is also present when the cell receives the true state.
- **Cause:** Phase-1 generation trained only anchor quotient depths
  `{0,1,2,5,10,50,100}`, not all intermediate depths on their traces. Thus
  states with, for example, q=7 were held-out distribution values, despite
  being necessary during a q=10 rollout. Near-zero training loss did not imply
  a uniformly correct one-step transition.
- **Labeled correction:** the earlier conclusion that q>=10 failure was
  *caused solely* by self-fed drift is unsupported. The supported result is
  only that anchor-depth support reaches q=5 but not q=10. The next card holds
  architecture and optimization fixed, fills trace-depth support q=0..100,
  and reruns the same teacher/free audit to isolate drift cleanly.

### 2026-08-04 — Full-trace iterative reducer: stable rollout correction (Author: Codex)

- **One changed variable:** identical 400,906-parameter tied transition,
  optimizer, seed, 800/256 disjoint remainders, batch size, and 2,000 updates;
  training now includes every true trace depth q=0..100 rather than anchor
  depths only.
- **Result:** free terminal exact is q=0 100.00%, q=1 99.61%, q=2 99.61%,
  q=5 99.22%, q=10 99.22%, q=50 97.27%, and q=100 95.70%. The paired audit
  has final teacher/free exact of 99.61/99.22% at q=5 and q=10,
  99.61/97.27% at q=50, and 99.61/95.70% at q=100.
- **Supported conclusion:** a tied learned decimal transition can execute a
  stable 100-step self-fed reduction trajectory when it is trained on the
  intermediate-state distribution it will encounter. Fixed depth and an
  inherently unusable decimal state are not the dominant explanation for the
  earlier q>=4 failure.
- **Still unsupported:** competition relevance. Phase 1 receives q from the
  evaluator as an unroll count, so it is diagnostic-only. The new bottleneck
  is selecting/learning the stop condition without q; that is a separate
  learned-halting card, now scientifically justified but not yet implemented.
- **Labeled correction to the preceding working theory:** state drift exists
  (3.91 percentage points at q=100), but it is not the primary q>=10 collapse
  observed in the anchor-depth card; missing intermediate-state support was.
- **Artifacts:** full model, raw metrics, and paired rollout audit are durable
  at `diagnostics/artifacts/somn-l40-2026-08-04/teacher_depth_full_trace/`.

### 2026-08-04 — Updated reduction conclusion and Phase-2 target (Author: Codex)

- **Supported by commit `9e00277`:** the tied learned reduction primitive is
  capable of stable long-horizon execution when trained on the intermediate
  state distribution it encounters. The q=100 teacher/free gap is 3.91 points
  (99.61%/95.70%): degradation, not collapse.
- **Updated decomposition:** fixed-depth networks do not scale computation
  with quotient depth; tied recurrence does. Full rollout-state support is a
  first-order training requirement. Residual state drift is secondary to that
  support issue, not the primary blocker.
- **Remaining gap:** Phase 1 receives q as evaluator-supplied unroll depth,
  which is diagnostic-valid but not competition-valid. The next experiment is
  learned canonicality detection/halting: a learned stop head sees `(state,N)`
  and decides when the tied reducer should stop, with no q supplied at
  inference. It must report exact remainder, stop accuracy, iteration count,
  early stops, non-stops, and wrong-remainder stops.

### 2026-08-04 — Learned canonicality/halting, seed 0 (Author: Codex)

- **Classification:** **NEW DIAGNOSTIC — NOT SUBMISSION-RELEVANT PENDING RULE
  REVIEW.** One change from the full-trace reducer: a learned binary stop head
  over the same `(state,N)` representation, with stop supervision only at
  canonical q=0 states. The reducer, q=0..100 trace support, optimizer, seed,
  batch size, and 2,000-update horizon are unchanged.
- **Autonomous result on 256 independent remainders:** exact remainder and
  halting accuracy are each 100.00% in q=0, q=1, q=2-3, q=4-9, q=10-99, and
  q=100. Mean iterations are 0, 1, 2.5, 6.5, 54.5, and 100 respectively.
  Early stops, late stops, failures to stop, and wrong remainders after a
  correct-depth stop are all 0.00%.
- **Supported conclusion:** within the trained depth range, the model can
  autonomously stop a stable learned reduction rollout without externally
  supplied q. The remaining research question is depth extrapolation beyond
  q=100 and eventual rule-compliant integration—not state correction.
- **Unsupported conclusion:** full Task B/B5 or competition success. B5 has
  quotient depths beyond 100, and learned control flow has not received a
  fresh submission-rule review. Artifacts are durable at
  `diagnostics/artifacts/somn-l40-2026-08-04/learned_canonicality_halting/`.

### 2026-08-04 — Generalization plan after learned halting (Author: Codex)

- **Stage 1, active:** quotient extrapolation. Hold the q=0..100 reducer and
  stop head fixed; evaluate independent remainders at every q=101..500,
  reporting teacher-forced local fidelity separately from autonomous exact,
  halting accuracy, rollout length, and classified stop failures.
- **Stage 2, gated on Stage 1:** train the unchanged reducer on multiple N
  values and test unseen N, with the same autonomous metrics. This separates
  learned reduction from modulus memorization.
- **Stage 3, gated on Stages 1–2:** evaluate the competition-shaped quotient
  distribution. No architecture change is bundled into any stage.

### 2026-08-04 — Quotient extrapolation q=101..500, Stage 1 (Author: Codex)

- **Setup:** evaluation-only on the committed learned reducer/stop-head
  checkpoint trained at q=0..100; same 256 independent remainders; every
  unseen integer depth q=101..500. Teacher-forced one-step accuracy at each
  unseen starting state is reported separately from autonomous execution.
- **Result:** aggregate teacher one-step exact is 11.82%, autonomous remainder
  exact 11.81%, and halting accuracy 11.81%; 88.19% stop early. The model is
  100% through q=147, 25.78% at q=148, and 0% from q=149 through q=500.
  q=101-150 averages 94.57% teacher and 94.52% autonomous exact; every later
  50-depth group is 0% and stops early on every row.
- **Classification:** Refuted the reusable-transition extrapolation prediction.
  The first failure is local transition/canonicality quality on unseen large
  states, not accumulated rollout error: teacher and free metrics fall
  together. Stage 2 (unseen-N) and Stage 3 (competition distribution) are
  gated off; they cannot establish algorithmic generalization while the
  fixed-N depth extrapolation has this cliff.
- **Artifacts:** `quotient_extrapolation_101_500.json` is synced under
  `diagnostics/artifacts/somn-l40-2026-08-04/learned_canonicality_halting/`.

### 2026-08-04 — Revised quotient-depth plan after Stage 1 (Author: Codex)

- **Interpretation constraint:** the q=148 cliff does not yet prove a finite
  recurrent horizon. Teacher-forced and autonomous metrics collapse together,
  but the learned stop head may still be responsible for the observed early
  stops.
- **Next ordered tests:** (1) reducer-only q-known rollout at q=101..500;
  (2) a no-training boundary map at q=101,110,120,130,140,145..150 derived
  from the same curve; (3) only after component separation, curriculum
  extensions q=0..200→201..500 and q=0..500→501..1000. Architecture remains
  fixed. Unseen-N and competition-distribution tests remain gated off.

### 2026-08-04 — Reducer-only separation and boundary map (Author: Codex)

- **Reducer-only result:** externally supplying q and applying the unchanged
  reducer exactly q times yields 11.81% terminal exact across q=101..500—the
  same as autonomous learned-halting exact. The stop head is not causal.
- **Requested boundary map:** q=101,110,120,130,140,145,146,147 are 100%; q=148
  is 25.78%; q=149 has 2.73% teacher one-step but 0% q-known terminal; q=150
  is 0%. The learned transition encounters an unsupported-state cliff.
- **Next card:** one-variable depth-curriculum extension q=0..200, tested at
  q=201..500. If the cliff moves with support, it favors coverage over a fixed
  architectural horizon; unseen-N and competition-shaped evaluation remain
  gated off until this is known.

### 2026-08-04 — Curriculum q=0..200, extrapolation q=201..500 (Author: Codex)

- **Result:** q=201..221 have 100% teacher-one-step, q-known terminal,
  autonomous remainder, and halting exact; q=222 is 37.89% on all four;
  q=223..500 are 0% and stop early. Aggregate q=201..500 q-known/autonomous
  exact is 7.13%.
- **Classification:** Confirmed the support-coverage hypothesis. Expanding
  support q<=100→q<=200 moves the first imperfect depth 148→222 while the
  reducer architecture remains fixed. The next pre-registered stage expands
  only trace support to q<=500 and tests q=501..1000.

### 2026-08-04 — Curriculum q=0..500, extrapolation q=501..1000 (Author: Codex)

- **Result:** q=500 and q=501..518 are 98.05% q-known/autonomous terminal
  exact; q=519 is 90.62%; q=650 is 80.47%; q=677 is 4.69%; q=678..1000 is 0%.
  Aggregate q=501..1000 q-known/autonomous exact is 29.79%, halting is 30.48%,
  teacher-forced one-step exact is 41.40%, and 68.90% stop early.
- **Interpretation:** This is qualitatively less abrupt than the 148 and 222
  boundaries and it remains accurate 18 depths beyond support. It is not a
  clean proof of extrapolation: direct q=500 evaluation is already 98.05%, so
  its rollout error can accumulate before the held-out range starts.
- **Next card:** Keep the entire q=0..500 setup fixed and increase updates
  2,000→6,200 to match q=0..200's transition-row exposure. First require a
  100% q=500 floor; only then interpret q=501..1000 as quotient extrapolation.

### 2026-08-04 — Exposure-matched q=0..500, extrapolation q=501..1000 (Author: Codex)

- **In-range gate:** independent q=500 rollout is 100% teacher-one-step,
  q-known terminal, autonomous terminal, and halting exact after 6,200 updates.
- **Held-out result:** q=501..666 are also 100% on all four measures. q=667 is
  69.92% terminal, q=741 is 16.80%, and q=742..1000 is 0%. Aggregate q=501..1000
  teacher-one-step is 47.98%, terminal is 43.12%, and halting is 43.65%.
- **Classification:** Confirmed the narrow scientific claim: a learned tied
  reducer can execute an exact reusable transition for 166 quotient depths
  beyond its trained horizon. The later cliff still shows finite distribution
  support/representation limits; it does not refute extrapolation.
- **Next stage:** per the staged plan, hold the architecture fixed and train
  on N={1081,1349,1763}, then test unseen N={1189,1517}. This distinguishes a
  modular reduction primitive from a fixed-N=1349 state mapping.

### 2026-08-04 — Multi-N unseen-modulus screen (Author: Codex)

- **Training:** fixed reducer, q=0..500 traces, N={1081,1349,1763}, 15,000
  updates (approximately matched transition-row exposure). Held-out remainders
  on seen N reach 95.31–97.66% terminal exact at q=100, so this is a screening
  result rather than a clean perfect-in-range reproduction.
- **Held-out-modulus gate:** at q=1, N=1189 and N=1517 have 0% exact next-state
  and terminal remainder accuracy. Token accuracy is 66.06% / 56.74%, which is
  insufficient: the decimal state has at least one wrong digit on every case.
- **Classification:** Refuted the modulus-generalization hypothesis for this
  representation/training formulation. Do not spend compute on deeper unseen-N
  rollouts or competition-distribution evaluation of this checkpoint.

### 2026-08-04 — Fixed-N depth-frontier confirmation plan (Author: Codex)

- **Observation, not law:** full-collapse occurs at q=149 after q<=100 training
  and q=742 after exposure-matched q<=500 training. Their ratios are 1.49 and
  1.484. This two-point alignment motivates one intermediate confirmation; it
  is not evidence of a general scaling law.
- **Registered card:** train q=0..300 at matched per-transition exposure
  (3,000 updates) and evaluate every independent q=301..550. The prediction is
  complete collapse near q=447. It must not be tuned after observing the curve.
- **Required report:** final perfect q, first degraded q, first zero-exact q,
  teacher-forced one-step exact, autonomous remainder exact, halting failure
  types, and a dense per-q curve around the first boundary.
- **Confound diagnostic:** at fixed N, s=qN+r conflates quotient depth with raw
  decimal magnitude. After the confirmation, separate (a) teacher-forced rows
  at matched q but low/high remainder bands, and (b) matched state-magnitude
  bands produced under different *seen* N values and thus different q. A
  teacher-forced magnitude failure identifies unsupported decimal state
  geometry before rollout; a free-only failure identifies recurrence/rollout.
  A raw-decimal versus N-relative representation comparison is deferred to the
  unseen-N branch because it changes the representation, not this confirmation.
- **Priority:** freeze fixed-N depth work after this one confirmation unless the
  competition's quotient range specifically demands more. The primary blocker
  remains held-out-N q=1 exact: the present multi-N reducer is 0% there, so a
  modulus-compositional, digit-significance-aware subtraction representation
  must pass unseen-N q=1 before any unseen-N depth experiment.

### 2026-08-04 — q<=300 intermediate frontier confirmation (Author: Codex)

- **Qualification:** the originally registered 3,000-update card was invalid
  because q=300 autonomous exact was 85.55%. A separately registered
  exposure-matched 3,800-update gate restored q=300 to 100% exact/halting.
- **Result:** held-out q=301..302 are 100%; q=303 is 94.53% teacher one-step,
  q-known terminal, autonomous terminal, and halting; q=371..550 is 0% exact
  and stops early. Thus last perfect q=302, first degradation q=303, and first
  zero q=371.
- **Classification:** Refuted the pre-registered q≈447 collapse prediction and
  the tentative 1.49×q_train frontier relation. Three conditions are not a
  scaling law: the boundary depends on details beyond maximum q support. Since
  teacher and autonomous metrics fail together at q=303, this run supports the
  unsupported decimal-state transition diagnosis over a pure rollout-length
  diagnosis. Freeze fixed-N frontier fitting as planned.

### 2026-08-04 — Primary branch: serial unseen-N subtractor (Author: Codex)

- **Architecture:** learned digit embeddings, shared GRU, and categorical heads
  scan aligned `(u digit, N digit)` pairs LSD-to-MSD; no handwritten arithmetic,
  borrow, comparison, quotient, or lookup occurs in the forward.
- **q=1 result:** training on 48 seen four-digit semiprimes and evaluating 16
  unseen semiprimes × 128 independent remainders yields 100% seen-N exact,
  100% unseen-N exact, and 100% at every digit position. This supersedes the
  0% held-out-N q=1 parallel-decimal result.
- **Composition:** q=1-only rollout on unseen N is 100.00%, 94.53%, 88.67%,
  83.35%, and 77.15% at q=1..5; raw q>1 teacher-one-step fails faster. The
  remaining issue is unsupported higher-q transition states.
- **Next card:** keep the serial architecture and split fixed; add balanced
  q=1..5 transition traces and test unseen-N q=1..5, then q=6..10. Report
  teacher one-step, autonomous terminal, per-digit exact, failure type, and
  three-seed stability. Add learned canonicality only after this gate passes.

### 2026-08-04 — Serial q=1..5 support, three-seed audit (Author: Codex)

- **Support gate:** Seed 0 is 100% autonomous rollout q=1..10. Seed 1 is 100%
  q=1..6 and 96.58%, 95.17%, 93.90%, 92.72% at q=7..10. Seed 2 is 100% q=1..4
  and 99.76% q=5.
- **Width confound:** for one seed-2 held-out modulus, q=10 produces a six-digit
  qN+r state. The five-digit input representation rejects it, so higher-q seed-2
  interpretation is invalid. This is deterministic representation capacity, not
  arithmetic failure.
- **Next registered audit:** change only state width 5→6 and repeat all three
  seeds through q=10. Report teacher one-step, autonomous terminal, per-digit,
  representable count, arithmetic errors, width failures, mean/min seed accuracy,
  first q<100%, first q<95%, and curve shape. Promote canonicality only if q=1..5
  remains near-perfect, q=6..10 remains strong without collapse, and all remaining
  failures are arithmetic rather than width failures.

### 2026-08-04 — Serial six-digit width audit, seed 0 (Author: Codex)

- **Change:** decimal state width only, 5→6; leading-zero padding and LSD-relative
  alignment retained. Every generated qN+r and target state was asserted to fit;
  no examples were truncated or dropped.
- **Result:** unseen-N q=1 remains 100% exact at every one of six digit positions.
  Autonomous rollout is 100% through q=10 (2,048 examples per q). Teacher one-step
  is 100% q=1..7 and 95.26%, 91.65%, 87.50% q=8..10, while rollout remains exact.
- **Reading:** the prior seed-2 q=10 failure was representation width, not a
  subtraction failure. This requested one-seed width screen contains no width
  failures; residual raw-state teacher degradation is gradual, not cliff-like.

### 2026-08-04 — Frozen serial-subtractor learned canonicality, seed 0 (Author: Codex)

- **Setup:** froze width-six checkpoint
  `serial_subtractor_width6/seed0/final.pt`; trained only a 129-parameter
  linear stop readout of its final LSD-to-MSD GRU state. Training has true
  q=0..5 trace states from 48 seen moduli, with q=0 states oversampled so stop
  and continue labels are exactly 1:1. The stop head receives only current
  padded state and N: no quotient, remaining depth, or oracle at inference.
- **Evaluation:** 16 unseen four-digit semiprimes × 128 independent remainders
  per q, q=0..10. Starting from u=qN+r, execution repeatedly queries learned
  canonicality and, only when it says continue, applies the frozen learned
  subtractor; cap is 16 learned reductions. Artifact (not committed, including
  checkpoint): `diagnostics/artifacts/somn-l40-2026-08-04/serial_stop_head_width6_seed0_balanced_state_labels/`.
- **Result:** q=0..7 are 100% final remainder, halting correctness, exact stop
  step, and every output digit. q=8=95.26%, q=9=91.65%, q=10=87.50% on both
  remainder and correct stop step. There are zero late stops, non-stops,
  width/truncation errors, false-positive stops on noncanonical generated
  states, and false-negative continues on canonical generated states.
  Average executed steps are exactly q through q=7, then 7.823/8.560/9.093 at
  q=8/9/10 due to early terminal states.
- **Interpretation:** the canonicality head passes the promotion gate. Its
  high-q residual exactly matches the frozen subtractor's q=8..10 one-step
  exact curve; the apparent early stops follow a prior erroneous subtractor
  state that is already below N, rather than a false positive on a correct true
  trace. Generated-state mixture training is therefore not indicated yet.

### 2026-08-04 — Serial q=1..10 transition support and autonomous check (Author: Codex)

- **One changed training variable:** serial subtractor transitions were expanded
  from balanced q=1..5 to balanced q=1..10 across the same 48 seen moduli.
  Width six, LSD-to-MSD GRU, optimizer family, 4,000 updates, batch size, and
  16 held-out moduli × 128 remainders remained fixed. Every q=1..20 state is
  representable (40,960 total examples; zero width failures).
- **Subtractor-only result:** teacher-forced one-step exact is 100% q=1..14,
  96.88% q=15, 93.75% q=16, and 94.53/89.50/87.50/88.38% q=17..20. In contrast,
  q-supplied fixed-depth terminal rollout is 100% at every q=1..20 and every
  digit position. First one-step <100%=q15; first <95%=q16; no zero-exact q.
  Artifact (not committed, including checkpoint/logs):
  `diagnostics/artifacts/somn-l40-2026-08-04/serial_subtractor_width6_q10_support_seed0/`.
- **Old-head control:** the original q=1..5 stop head is 0% even at q=0 when
  attached to this independently retrained GRU (99.24% generated-state false
  positives and 100% canonical false negatives). This is a latent-coordinate
  mismatch control, not evidence that canonicality itself ceased to be learned.
- **Conditionally permitted rebound head:** a fresh 129-parameter readout was
  trained on the unchanged balanced q=0..5 stop states while the new q=1..10
  subtractor stayed frozen. Autonomous unseen-N remainder/stop-step exact is
  100% q=0..13, 96.88% q=14, and 93.75% q=15..16. The learned 16-step cap makes
  q=17..20 primarily non-stops (89.50/87.50/87.50/83.94%), as expected.
- **New boundary diagnosis:** q=14 has 3.12% early stops from accumulated
  stop-head false positives. At q=15 and q=16, 64 and 128 examples respectively
  reach an *incorrect but canonical* generated decimal state. The readout is
  correct to stop there, but fixed-q rollout would later self-correct it. Thus
  100% q-supplied terminal rollout does not yet establish a true reduction
  primitive with a correct state invariant. Across generated states, stop-head
  false-positive rate is 0.0826% and false-negative rate is 0%.

### 2026-08-04 — Stability gate and absorbing/recovery support (Author: Codex)

- **Frozen stability diagnostic:** with the q=1..10 subtractor and rebound stop
  head unchanged, a state could halt only when learned-canonical and unchanged
  by one further learned subtraction; the cap was raised 16→24. This is
  diagnostic-only code, never a submission mechanism. On every q=0..20 bin,
  all 2,048 true remainders have `F(r,N) != r`; hence every example is a
  non-stop at the cap and final exact is 0%. Wrong canonical events never stay
  unchanged or directly repair to r; all transition to another wrong state.
- **Required next condition:** held q=1..10 transitions, width-six GRU,
  optimizer family, batch, and 4,000 updates fixed; added 6,144 canonical
  identity rows r→r plus 2,299 unique wrong-canonical recovery rows generated
  by the frozen prior checkpoint on seen moduli. The new subtractor was then
  given its necessary fresh 129-parameter stop readout solely to bind to its
  independently trained latent coordinates.
- **Result:** refuted on unseen N. Direct true-remainder stability remains 0%
  in every q bin (2,048/2,048 non-absorbing each). Stability-gated q=0..20 is
  consequently 0% final exact and 100% non-stop. Fixed-depth terminal rollout,
  formerly 100% q=1..20, is now 100% q=1..2, 99.66% q=3, 98.44% q=4, 94.43%
  q=5, 74.56% q=10, and 66.26% q=20. Artifact (not committed):
  `diagnostics/artifacts/somn-l40-2026-08-04/serial_subtractor_width6_q10_absorbing_recovery_seed0/`.
- **Interpretation:** naive identity and frozen-trajectory recovery examples do
  not generalize their invariant across unseen moduli. They interfere with the
  compositional q-transition map, rather than converting it into a learned
  reduction primitive. Do not scale this data mixture or add q support/seeds.

### 2026-08-04 — Clean piecewise identity/subtraction transition (Author: Codex)

- **Question:** can the unchanged serial GRU learn the actual piecewise local
  map `F(u,N)=u-N` for u≥N and `F(u,N)=u` for u<N, without frozen-model recovery
  labels? Training used exactly balanced q=0..20 buckets: 6,144 identity rows
  q=0 and 6,144 subtraction rows at each q=1..20 (129,024 total), same width,
  optimizer family, batch, and 4,000 updates. There were zero recovery rows.
- **Transition result:** unseen-N q=1/5/10 one-step exact is 100%; q=20 is
  96.44%. But direct fixed-point q=0 is 0% on both the 6,144 seen training
  remainders and 2,048 unseen remainders. Thus identity did not merely fail to
  generalize: it was not fitted under this equally balanced training budget.
- **Prescribed-depth evaluation:** no state is truncated through q=100 (206,848
  held-out examples). q-supplied rollout is 100% through q=13, first below
  100% at q=14, first below 95% at q=17, 92.58% q=20, and 89.84% q=100. The
  q=0 rollout row is trivially 100% because zero transitions are applied; it
  must not be mistaken for fixed-point evidence.
- **Decision:** do not train/evaluate a stop head, because the required q=0
  fixed-point gate fails. This experiment identifies optimization/data-family
  interference at the present exposure, rather than a held-out-N representation
  failure. It motivates—but does not establish—the need for a future learned
  comparator/subtractor decomposition. Artifact (not committed):
  `diagnostics/artifacts/somn-l40-2026-08-04/serial_subtractor_width6_piecewise_q0_q20_seed0/`.

### 2026-08-04 — Comparator-controlled serial reducer (Author: Codex)

- **Why this architecture:** the monolithic serial GRU learned unseen-N
  subtraction but could not fit the q=0 branch `F(r,N)=r`. This card isolates
  the discontinuous decision: a learned LSD-first GRU comparator predicts
  `x≥N`; its probability selects learned subtractor digit probabilities or an
  identity residual. No handwritten comparison or arithmetic occurs in either
  learned module's forward; the residual is diagnostic-only and not a
  submission design.
- **Stage 1 comparator:** balanced seen-N data has 12,480 rows (half `<N`, half
  `≥N`) and explicit N−1/N/N+1 cases. The learned comparator is 100% seen-N,
  99.9279% unseen-N (4,160 rows), and 100% on all 64 held-out boundary rows.
- **Stage 2 gated reducer:** initialized from Stage 1 and the q=1..10 serial
  subtractor, then jointly trained on q=0..20 identity/reduction traces. On
  unseen N, transition exact is 100% q=0..28 (including q=0/1/5/10/20) and
  true-remainder fixed-point exact is 100%. Autonomous remainder and exact
  halt are 100% q=0..28 with zero early, late, and non-stops.
- **Frontier:** q=29 is first degradation (94.04%, all error is early stop);
  q=30 is 93.75%; q=100 is 37.50%, again entirely early stops after a learned
  subtraction produces a state below N too early. Thus comparison repairs the
  established fixed-point failure but does not make high-q learned subtraction
  exact. Artifact (not committed):
  `diagnostics/artifacts/somn-l40-2026-08-04/serial_comparator_controlled_reducer_seed0/`.
- **Classification:** supported: the previous missing capability was a learned
  comparison/branch condition, not digit order. Not established: a complete
  modular reducer, competition solution, or unlimited quotient extrapolation.

### 2026-08-04 — Frozen transition versus rollout audit (Author: Codex)

- **Question:** is the comparator-controlled reducer's q≥29 frontier caused by
  an inaccurate local learned transition, or by otherwise-good transitions
  accumulating errors in self-fed rollout? The q=0..20 checkpoint was frozen;
  no weights, representation, or loop changed. We tested 2,048 unseen-N
  examples each at q=1, 5, 10, 20, 30, 50, and 100.
- **Result:** comparator accuracy is 100% in every bucket. Raw subtractor and
  composed teacher-forced transition are each 100% through q=20, then 93.75%
  at q=30, 85.06% at q=50, and 86.04% at q=100. Autonomous final exact is
  respectively 100%, 62.50%, and 37.50% at those last three depths. The first
  recorded failure is a subtractor transition, not a comparator branch error.
- **Decision:** the primary regime is unsupported-state transition failure;
  rollout accumulation is secondary at q=50/100. The registered next control is
  q=0..100 intermediate-state curriculum support, with architecture unchanged.
  Artifact (not committed):
  `diagnostics/artifacts/somn-l40-2026-08-04/serial_comparator_controlled_reducer_seed0/audits/`.

### 2026-08-04 — q=0..100 intermediate-state curriculum (Author: Codex)

- **Question:** can the already-qualified comparator-controlled reducer learn
  its local piecewise transition on the high-q states that caused the frozen
  q=30 frontier, without an architecture change? We resumed that q=0..20
  checkpoint and changed only balanced transition support to q=0..100, retaining
  q=0 identity, width-six LSD-first scan, optimizer, batch size, and 4,000
  updates. The 48/16 seen/unseen modulus split and 128 remainders/modulus are
  unchanged. All q≤100 states fit six digits; none were truncated or dropped.
- **Result:** on unseen N, all 206,848 teacher-forced transition cases q=0..100
  are exact; learned continue/stop is also 100% in every q bucket. True q=0
  remainders are fixed points at 100%. Autonomous q=0..100 terminal remainder
  and exact halt are 100%, with zero early stops, late stops, or non-stops.
  The requested independent frozen table at q=1, 5, 10, 20, 30, 50, and 100
  is 100% for comparator, raw subtractor, composition, and final rollout in
  every row (2,048 examples/row).
- **Interpretation:** the former high-q failure was primarily missing
  algorithmically relevant transition-state support, not an intrinsic finite
  horizon of this serial comparator/subtractor mechanism. This establishes
  in-range q≤100 execution only; it does not establish quotient extrapolation,
  unlimited reduction, or a competition solution. Artifact (not committed):
  `diagnostics/artifacts/somn-l40-2026-08-04/serial_comparator_controlled_reducer_seed0/stage3_q100/`.

### 2026-08-04 — Frozen q=101..140 horizon probe (Author: Codex)

- **Question:** after q=0..100 trace support, does the learned local reducer
  execute beyond that observed quotient horizon? The q≤100 checkpoint was
  frozen and evaluated at q=101, 110, 120, 130, and 140. This is the largest
  shared extrapolation range that fits all held-out-modulus states in the fixed
  six-decimal-digit representation; q=145 would overflow for some examples.
- **Result:** every selected bucket has 100% comparator accuracy, raw subtractor
  next-state exactness, composed transition exactness, and autonomous final
  remainder exactness (2,048 examples/bucket). The first attempt displayed 0%
  q≥120 rollout because the audit itself retained an obsolete 110-step cap;
  after making the cap `max(requested_q)+10`, the frozen rerun is perfect.
- **Interpretation:** support through q=100 produces at least 40% quotient-depth
  extrapolation in this controlled width-six range. It remains a bounded
  extrapolation result, not evidence for unlimited execution; representation
  width blocks a clean all-example test at q≥145. Artifact (not committed):
  `diagnostics/artifacts/somn-l40-2026-08-04/serial_comparator_controlled_reducer_seed0/stage3_q100/q101_140_horizon_probe.json`.

### 2026-08-04 — Public competition-scale envelope audit (Author: Codex)

- **Question:** is the controlled width-six, one-subtraction-per-recurrence
  reducer near submission-ready on the public task scales? This is a source
  audit, not a dataset inspection or training run. Sources are the public
  catalog, manifests, README rules, and `competition/data/squaring_mod.py`.
- **Established public envelope:** generated inputs are units `1≤x<N`; for a
  one-step square, `u=x²` and `q=floor(u/N)` obey `0≤q≤N−2`. Thus the current
  learned reducer requires O(q) applications for one modular square. Public
  Medium includes m4 with 14/18/22-bit N at T=8. A 22-bit N can require a
  14-decimal-digit raw-square state and up to 4,194,302 unit reductions; its
  typical q is also O(N), not O(100). Width six covers neither its state nor
  its iteration count. The public Hard h1 distribution is hidden, so its exact
  N/q envelope cannot be determined without prohibited data inspection.
- **Decision:** do not claim that q≤140 establishes competition-scale
  reduction, and do not spend a run on width seven alone as a submission
  solution. The next controlled branch must preserve the serial learned path
  while testing a learned multi-unit/quotient-chunk reduction mechanism, after
  first setting a public-scale width target. No hidden data was read.

### 2026-08-04 — Width-six to width-fourteen serial control (Author: Codex)

- **Question:** does leading-zero expansion from six to fourteen decimal
  positions itself break the validated serial comparator/reducer? Fresh W=14
  weights repeated the seed-0 q=1..10 subtractor, boundary-balanced comparator,
  q=0..20 gated reducer, and q=0..100 support stages. LSD-first order, four-
  digit 48/16 seen/unseen split, optimizer, batch 512, and stage update counts
  were otherwise unchanged.
- **Result:** held-out q=1 subtractor exact is 100%. Comparator held-out-N
  accuracy is 98.4615% (4,160 examples) and N−1/N/N+1 boundary accuracy is
  100% (64 examples). At the final q≤100 stage, composed teacher transitions
  are 100% for q=1..100 and 99.9512% q=0; true fixed points and autonomous
  remainder/exact-halt are 99.9512% in each audited q bucket (2,047/2,048).
  Frozen q=1,5,10,20,50,100 checks give 100% initial comparator, raw
  subtractor, and composed transition, with the same 99.9512% terminal result.
- **Interpretation:** W=14 padding does not damage learned subtraction or
  boundary comparison, but one canonical state false-continues and so exact
  autonomous behavior is not preserved. This is a narrow comparator/fixed-point
  defect, not evidence that width alone yields public-scale reduction. Do not
  tune it inside this card; proceed to the separately registered chunk test.
  Artifact (not committed):
  `diagnostics/artifacts/somn-l40-2026-08-04/serial_comparator_width14_control/`.

### 2026-08-04 — Learned chunk reducer k∈{0,1,2,4,8} (Author: Codex)

- **Question:** can the same LSD-first serial GRU learn a multi-unit reduction
  action and its next digits, avoiding O(q) unit steps? The W=14 model adds a
  learned five-class action head to the serial digit decoder. It is trained
  from random initialization on q=0..100 targets `qN+r→(q−k)N+r`, where the
  largest allowed k not exceeding q is the synthetic label. The forward emits
  only learned action and digit logits; it contains no implemented comparison,
  subtraction, multiplication, or quotient calculation. Split, seed, batch,
  optimizer, and 4,000 updates match the preceding control.
- **Result:** refuted. Action accuracy is 84.91% q=0, 93.31% q=1, and 100%
  q≥20, but next-state exact is 0% q=0/q=1, 55.86% q=8, 51.86% q=100, and
  9.03% q=1000. Autonomous remainder exact is 84.91% q=0, 0% q=1, 0.146%
  q=100 (43.45 mean steps, versus its k≤8 lower-bound target of 13), and
  0.293% q=1000. The local learned chunk transition—not merely halting or
  long rollout—fails.
- **Decision:** kill this fresh joint action-and-digit formulation. It does not
  preserve the validated unit subtraction law, so do not proceed to public
  14/18/22-bit data or claim sublinear reduction. A future card may isolate
  initialization from the qualified unit primitive, but must not be confused
  with this refuted from-scratch chunk architecture. Artifact (not committed):
  `diagnostics/artifacts/somn-l40-2026-08-04/serial_chunk_reducer_0248/`.

### 2026-08-04 — Unit-preserving chunk controls (Author: Codex)

- **Option A — initialized direct chunk decoder:** loaded the qualified W=14
  subtractor weights into the fresh chunk model, leaving only its five-way
  action head random. All chunk targets, loss, seed, optimizer, batch, and
  4,000-update budget match the refuted fresh card. **Result: refuted.** q=0
  action accuracy rises to 99.32%, but next-state exact remains 0% q=0/q=1,
  43.07% q=8, 39.16% q=100, and 42.33% q=1000; terminal q=100 exact is 0%.
  Direct `x−kN` digit generation is not obtained merely by initializing from
  the learned `x−N` primitive.
- **Option B — frozen unit plus learned controller:** froze the complete W=14
  comparator-controlled unit reducer and trained only a 645-parameter action
  head. The predicted action schedules repeated frozen learned unit updates;
  no arithmetic is added to the forward. **Result: refuted as configured.**
  Frozen macro transition is 99.95% q=0/q=1, but action accuracy is 0% q=0..5
  and 100% q≥8; action 0 is never selected, causing 100% non-stops. q-balanced
  q=0..100 traces are action-imbalanced: 93 quotient buckets label k=8.
- **Decision:** do not re-open fresh or initialized direct chunk-digit decoders.
  The frozen controller is a valid diagnostic of scheduling, but needs one
  action-class-balanced exposure control before concluding that learned chunk
  selection cannot bind to a preserved unit primitive. Artifacts (not
  committed): `diagnostics/artifacts/somn-l40-2026-08-04/serial_chunk_reducer_unit_init/`
  and `diagnostics/artifacts/somn-l40-2026-08-04/frozen_unit_chunk_controller/`.

### 2026-08-04 — Action-class-balanced frozen chunk controller (Author: Codex)

- **Question:** did the Option-B controller fail only because q-balanced traces
  place 93 of 101 q values in action k=8? Change only the sampler so all five
  action labels are equally represented per batch; retain frozen W=14 unit
  reducer, controller head, q=0..100 rows, targets, loss, seed, optimizer,
  batch 512, and 4,000 updates.
- **Result:** refuted as a chunk solution. q=0 action, macro transition, and
  terminal exact are all 100%; q=1 terminal exact is 93.75%. But action
  accuracy is 0% q=2, 11.67% q=4, 56.35% q=8, 41.85% q=100, and 57.18%
  q=1000. q=100 terminal exact is 0%, despite 13.11 mean outer actions versus
  a 13-action target. The frozen arithmetic remains intact; the controller
  cannot choose useful multi-unit actions from its frozen serial features.
- **Decision:** the required controls are complete. Direct chunk-digit learning
  is refuted both fresh and unit-initialized; frozen scheduling is refuted both
  q-balanced and action-balanced. A new controller representation/architecture
  would be a new research branch and requires an explicit card choice. Artifact
  (not committed):
  `diagnostics/artifacts/somn-l40-2026-08-04/frozen_unit_chunk_controller_balanced_actions/`.

### 2026-08-04 — Frozen unit threshold-bank controller (Author: Codex)

- **Question:** does decomposing a multi-unit chunk into four learned binary
  magnitude bits fix the frozen controller's action-selection failure? The
  complete qualified W=14 comparator-controlled unit reducer remains frozen.
  A 516-parameter head emits bits for a safe greedy code `k=min(q,15)` and
  schedules that many repeated learned unit transitions. This uses a code, not
  literal independent “can subtract” labels: the latter would choose 15 at
  q=8 and overshoot. Training balances the sixteen codes on the same q≤100,
  48-seen/16-unseen-modulus setup, then evaluates 2,048 unseen-N examples per
  quotient.
- **Result:** q=0 is fully preserved (1.0 fixed-point/remainder exact). At
  q=1, selected-code/macro/remainder exact are 15.72%/77.25%/77.25%. At
  q=100, individual-bit accuracies are 94.43%, 80.32%, 74.07%, and 92.68%;
  selected-code/macro/remainder exact are 73.29%/73.29%/30.18%, with 4.22 mean
  outer actions against a seven-action greedy target. q=1000 remainder exact
  is 0%, with 13.33% non-stops under the 70-step audit cap.
- **Decision:** refute threshold coding as a promotion candidate: it improves
  high-q terminal accuracy relative to the 0% five-way balanced control but
  violates the q=1 preservation gate and remains unstable under rollout. The
  next permitted diagnostic is a representation audit comparing the frozen
  GRU’s final state with its per-position states; do not add another controller
  mechanism first. Artifact (not committed):
  `diagnostics/artifacts/somn-l40-2026-08-04/frozen_unit_threshold_bank/seed0/eval_report.json`.

### 2026-08-04 — Final-state versus per-position controller audit (Author: Codex)

- **Question:** is the failed threshold-bank controller missing quotient
  information because the frozen serial GRU compresses all digit positions into
  one final state? Change only its controller input from that final 128-vector
  to the concatenation of all fourteen frozen GRU position states. The same
  W=14 reducer, safe four-bit chunk code, q≤100 rows, sixteen-code balancing,
  seed, batch 512, AdamW, 4,000 updates, and unseen 16-modulus evaluation are
  retained. The per-position linear head has 7,172 parameters versus 516.
- **Result:** support the compression hypothesis. q=0 stays 100% fixed-point
  and terminal exact. q=1 selected-code/macro/remainder exact are
  89.06%/99.95%/99.95%; q=8 remainder exact is 98.93%; q=32 is 99.27%; and
  q=100 threshold/selected-code/macro exact are all 100%, with 99.51% terminal
  exact across 2,048 held-out-N examples. At q=100 the controller takes 7.50
  mean outer actions (seven greedy target) but 102.05 mean inner learned unit
  updates. q=1000 terminal exact is 0% because the controller has no
  intermediate-state support above q=100 and stops early, not because its raw
  q=1000 action prediction is inaccurate.
- **Decision:** final-state compression is the action-selection bottleneck;
  the binary code itself is viable when per-position serial features are
  exposed. This is not a public-scale solution: repeated unit execution keeps
  inner computation O(q). Close the final-state controller formulation. Any
  next chunk card must preserve per-position control while changing the *state
  transition* cost, and requires a new explicit decision. Artifact (not
  committed):
  `diagnostics/artifacts/somn-l40-2026-08-04/frozen_unit_threshold_bank_per_position/seed0/eval_report.json`.

### 2026-08-04 — Track A Easy serial recurrent candidate (Author: Codex)

- **Question:** can the validated representational mechanisms—LSD-relative
  position, tied recurrence, and learned discrete state—yield a smallest legal
  end-to-end Easy model from prompt `(N,x,T)` alone? The candidate adds a shared
  bidirectional-attention/right-to-left-GRU cell and applies it up to four
  times selected by the prompt T field. It receives no precomputed square,
  quotient, reduction state, factor, or intermediate target. Public Easy
  exposes final labels only, so trace supervision is not permitted.
- **Result:** legal but locally refuted as an Easy improvement. Static source
  validation and evaluator CPU smoke pass. A tier-faithful public e1 run on an
  L40 trains 500 updates in 60.00 seconds with 21,152 model-state elements;
  final test exact is 1.33% (2/150), OOD exact 0% (0/100), and mean exact
  0.67%. This is below the historical e1 reference (4.7% test / 9.0% OOD /
  6.8% mean), whose leaderboard meaning is itself documented as weak.
- **Decision:** keep the source as a legal, audited submission attempt only;
  do not claim performance improvement or tune it before the requested single
  submission completes. Evidence: `EASY_SUBMISSION_REPORT.md` and remote
  `~/somn-taskb/easy_serial_recurrent/e1/run.log`.

### 2026-08-04 — Action-conditioned learned macro transition (Author: Codex)

- **Question:** did direct learned chunk transitions fail because their digit
  decoder never received the selected action? Freeze the qualified W=14 unit
  comparator/reducer and the successful per-position four-bit controller. A
  new 136,970-parameter LSD-first serial decoder is initialized from unit
  subtractor weights and receives the frozen learned chunk bits at every digit
  update. It alone is trained on q≤100 seen-modulus targets; unseen-N evaluates
  direct one-call macro transition and autonomous rollout.
- **Result:** refuted. At q=100 the controller selects the correct chunk on
  100% of 2,048 unseen-N inputs, but direct macro-transition exact is 0% and
  autonomous terminal exact 0.10%. q=1 macro/terminal exact are both 47.07%;
  q=2 macro exact is 0.59%, then q≥8 is 0%. The correct action plus a learned
  action-conditioned decoder does not preserve exact multi-unit state update.
- **Decision:** action selection and final-state compression are now ruled out
  as the main reason multi-unit reduction fails. The blocker is exact learned
  generation of `F^k(state,N)` itself. Do not retry the same conditioned
  decoder or merely tune its optimizer; a later branch must introduce a
  compositional transition representation, not another chunk selector.
  Artifact (not committed):
  `diagnostics/artifacts/somn-l40-2026-08-04/action_conditioned_macro_transition/seed0/eval_report.json`.

### 2026-08-04 — Hierarchical accelerator reclassification and VDF audit (Author: Codex)

- **Hierarchical accelerator:** the requested architecture already exists in
  `train_frozen_unit_threshold_controller.py --features positions`: frozen
  state → learned per-position four-bit controller → selected chunk k → k
  repeated frozen learned unit updates. Its held-out-N q=0 fixed point is 100%;
  q=1 terminal exact 99.95%; q=100 selected-k/macro exact 100% and terminal
  exact 99.51%. It takes 7.50 mean external decisions at q=100.
- **Critical cost result:** q=100 also takes 102.05 mean *unit* transitions.
  Thus it is a correct hierarchical scheduler, but not an arithmetic
  accelerator: repeated unit execution remains O(q). The failed
  action-conditioned macro decoder confirms that replacing those k calls with
  one learned digit decoder loses exact state generation.
- **VDF-pipeline audit:** a legal end-to-end solution must learn
  `state -> Square(state,N) -> Reduce(raw_square,N)` and repeat it T times.
  Current evidence begins only at an externally supplied raw reduction state;
  no learned square-to-reducer bridge is established. Public Medium m4 can
  require up to 4,194,302 unit reductions per square, so the current hierarchy
  is not eligible for Medium integration. Full source-backed audit:
  `VDF_PIPELINE_AUDIT.md`.

### 2026-08-04 — Clean recurrent learned VDF cell, small-N gate (Author: Codex)

- **Architecture:** a single reusable VDF transition applies an LSD-first
  learned raw-square GRU, then repeatedly applies learned comparator and
  learned subtractor GRUs until the comparator says canonical. The same modules
  have no phase-specific weights and are rolled out for T=1…8. Synthetic
  intermediate labels train the three primitives; the forward contains no
  handwritten multiplication, comparison, subtraction, modulo, quotient, or
  fixed-T answer path. The complete regime uses every residue of 18 seen and 8
  held-out two-digit semiprime moduli, so raw squares fit W=4 and reducer q
  support is complete for every seen modulus.
- **Run recovery:** the first remote artifact never trained because its q=0
  identity target was accidentally generated as a negative number. Repairing
  only that label to `r→r` produced the reported clean run; no architecture or
  optimization setting changed.
- **Seen-N result (1,147 examples):** Squareθ and its raw square representation
  are 100%; subtractor teacher exact 97.63%; comparator 99.98%; composed
  T=1 exact 93.72%. Rollout T=1…8 is 93.72, 91.37, 89.89, 88.14, 88.14, 88.06,
  88.06, 88.14%.
- **Held-out-N result (428 examples):** Squareθ/raw-square representation are
  100%; comparator is 99.97%; subtractor teacher exact is 80.36%; reduction
  given true raw square is 46.96%; reduction given model raw square is the
  identical 46.96%. Thus there is no square/reducer interface mismatch. Final
  rollout T=1…8 is 46.96, 37.62, 34.11, 33.88, 34.58, 34.81, 34.81, 33.88%.
- **Failure localization:** held-out reduction exact by true quotient is q=0
  100% (62), q=1 90.91% (22), q=2–3 54.29% (35), q=4–9 41.54% (65), and q≥10
  29.92% (244). Training includes the same numerical q range, so the failure
  is unseen-modulus serial subtraction/reduction generalization, magnified by
  repeated transitions, rather than missing quotient-depth support or a square
  representation distribution shift. Do not integrate this cell into a
  submission. Artifact (not committed):
  `diagnostics/artifacts/somn-l40-2026-08-04/recurrent_vdf_square_reduce_smalln/seed0/localization/eval_report.json`.

### 2026-08-05 — VDF-square-trace reducer support (Author: Codex)

- **Question:** did the full VDF cell fail because the reducer was trained on
  uniform algebraic `qN+r` rows rather than the raw square and intermediate
  states it sees inside the VDF transition? Change only comparator/subtractor
  rows: for every seen-modulus residue s, train on every trace state from
  `s²` down to `s² mod N`. The exact learned Squareθ checkpoint, serial
  architecture, 3,000-update budget, 18/8 split, and all-residue evaluation
  are retained.
- **Held-out-N result (428 examples):** raw Squareθ remains 100%; comparator
  is 99.91%; generic subtractor teacher exact is 86.17%; but reduction given
  the true or model raw square is **95.56%**, versus 46.96% under uniform
  `qN+r` support. Reduction exact by quotient: q=0 100% (62), q=1 100% (22),
  q=2–3 88.57% (35), q=4–9 96.92% (65), q≥10 **94.67%** (244).
- **Tied rollout:** unseen-N VDF T=1…8 exact is 95.56, 92.99, 90.65, 89.95,
  89.02, 87.85, 87.85, 89.02%. The corresponding original uniform-support
  values were 46.96% at T=1 and 33.88% at T=8.
- **Interpretation:** confirmed state-distribution mismatch. The existing
  comparator/subtractor representation can compose with learned square in this
  complete small-N regime when trained on algorithmically relevant trace
  states. Residual local reduction error remains, so this is evidence for a
  clean recurrent transition—not a submission-ready exact solver. Artifact
  (not committed):
  `diagnostics/artifacts/somn-l40-2026-08-05/recurrent_vdf_reducer_square_trace_support/seed0/eval_report.json`.

### 2026-08-05 — Final-label VDF execution controls (Author: Codex)

- **Scope:** these are legal final-label-only e1 submission-model controls,
  not evidence that the competition VDF problem is solved. All use the same
  learned tied Square→Reduce cells, AdamW schedule, batch size, seed 74, and a
  60-second L40 budget. They measure whether implementation changes preserve
  exact-match behavior while increasing updates.
- **Results:** dynamic-depth execution, which is the source submitted for
  Hard at commit `3bd03d9`, completed 434 updates and obtained 3.33% test / 0%
  OOD. Fusing the valid digit prefix with `nn.GRU` completed 490 updates
  (+13%) but fell to 2.00% / 0%. Active-row compaction gave 463 updates and
  4.00% / 0%. Restricting interim logits to learned register positions gave
  494 updates and returned to 3.33% / 0%. Tensorizing prompt-T parsing was
  refuted: 453 updates and 1.33% / 0%; the sequential parser was restored.
- **Interpretation:** valid-prefix scan and register-only logits are measured
  throughput improvements; none removes the decisive OOD-generalization
  failure. Muon controls also overfit (99.8% train exact, 0% OOD). Do not
  infer an accuracy ordering from these single-seed runs.
- **Artifacts:** excluded from Git and retained on the GPU mirror as
  `~/somn-taskb/runs/competition/vdf_square_reduce_{dynamic_depth,fused_valid_gru,active_row_compaction,register_only_logits,tensorized_t_parse}_e1/`.

### 2026-08-05 — Integrated final-label VDF on public Medium m1 (partial, Author: Codex)

- **Registered engineering integration:** fused valid-prefix scan, active-row
  compaction, and register-only recurrent logits were combined, while AdamW,
  width, final-label loss, tied Square→Reduce cells, and sequential T parsing
  remained fixed. The exact public m1 manifest was generated from the pinned
  repository command after the L40 was found to have only e1 data.
- **Observed before intentional stop:** the run reached 3,400 updates in
  407.4 seconds (about 8.35 steps/s); initial loss fell 28.351→2.249, but
  recorded batch exact was 0–0.39%. It was stopped at the user's direction
  before the 600-second training budget and before final evaluator-owned test,
  OOD, and depth evaluation. It must **not** be cited as a Medium score.
- **Interpretation:** this is evidence of early optimization/credit-assignment
  failure, not an execution bottleneck: the model has ample updates but has not
  learned even its in-batch exact output. It is compatible with, but does not
  prove, the hypothesis that final-only loss cannot identify the diagnostic
  Square/Reduce primitives at m1 scale. Artifact (Git-ignored):
  `runs/vdf_square_reduce_integrated_medium_m1_partial/train.log`.

### 2026-08-05 — Final-label transition-identifiability controls (Author: Codex)

- **Legal e1 curriculum:** existing public final-label rows were staged
  T=1 → T≤2 → T≤3, while each model forward always executed that row's actual
  input T. It completed 461 L40 updates in 60.1s: test 3.33%, OOD 0%, seen-N
  rung T=1 5.2632% (2/38), and zero at every seen rung T≥2. This is a small
  one-step signal but no composable transition; it does not promote.
  Artifact: `runs/vdf_final_label_t_curriculum_e1/competition_report.md`.
- **Trace-supervision ablation (diagnostic only):** the identical VDF model and
  e1 prompts received equal-weight generated intermediate-state register loss.
  It reached 598 updates/60.1s and 24.22% batch final exact at step 500, but
  final held-out exact was 0% at test T=1/2/3 and OOD T=6. Intermediate labels
  improve fitting yet do not make this prompt-tail register representation
  generalize in the one-minute envelope. Do not call this evidence that trace
  supervision is competition-legal or that the serial diagnostic architecture
  is refuted; it only closes this specific final-label VDF cell formulation.
  Artifact: `diagnostics/artifacts/vdf_trace_supervision_ablation_e1/report.json`.
- **Architecture audit:** on the same e1/L40/60-second control, upstream direct
  Transformer A makes 697 updates and gets 2.00% test/1.00% OOD; tied VDF B
  makes 434 updates and gets 3.33%/0%; tied VDF curriculum C makes 461 updates
  and gets 3.33%/0%. No model certifies any T. A's 1% OOD and isolated 10.53%
  seen T=32 are non-monotone, so they are not evidence of extrapolation.
  C has a 5.26% seen T=1 bump but zero seen T≥2. This comparison does not
  support the current prompt-tail-register tied-VDF architecture over direct
  output modeling under a final-label-only one-minute objective.

### 2026-08-05 — True final-label depth curriculum (research only, Author: Codex)

- **Question:** can a model learn a reusable tied transition from final labels
  if it first masters composition depth one? This removes evaluator/submission
  constraints but retains final labels only: fixed N=323, held-out x split,
  phase 1 T=1, phase 2 T<=2, phase 3 T<=4, then held-out x ladder T=1…64.
  No intermediate labels appear in the loss.
- **Result:** 3,095 updates in 180.0 L40 seconds. Training exact reached 100%
  in the T=1 phase by step 500, returned to 100% in T<=2, and reached 97.66%
  at step 3,000 in the T<=4 phase. Nevertheless held-out-x final exact was
  **0%** at T=1/2/4, 1.54% at T=8 (1/65), and 0% at T=16/32/64. Unseen three-
  digit semiprime N was at most 0.66%.
- **Conclusion:** this is strong negative evidence against final-label depth
  curriculum as the missing ingredient. It fits every phase but does not learn
  a transferable one-step VDF law. The next branch must alter the state/input
  representation or use an explicitly supervised/otherwise identifiable local
  transition; more phase tuning is closed. Artifact:
  `diagnostics/artifacts/vdf_final_label_true_depth_curriculum/report.json`.

### 2026-08-05 — L40 archive and teardown (Author: Codex)

- **Archive:** the complete remote experiment mirror `~/somn-taskb` was copied
  before teardown to Git-ignored
  `diagnostics/artifacts/somn-l40-2026-08-05-final-backup/somn-taskb`.
  Verification matched **437 files** and **18,718,435 bytes** exactly; the
  manifest is `PRIME_BACKUP_VERIFIED.txt` in that archive. A source snapshot
  excluding recreatable virtualenv/generated data is retained beside it.
- **Instance:** verified target is Prime pod `somn-l40`
  (`bfa4b7489af34ba78a2dec326c8b9bf2`), L40 48GB. Teardown is user-requested
  after no active GPU process remained.
- **Completion:** remote `~/somn-taskb`, `~/one-layer-deeper`, and recreatable
  uv caches were wiped only after archive verification. Prime CLI terminated
  the exact pod; provider listing then returned zero pods and SSH timed out.

### 2026-08-07 — Five distinct legal submission cards (packaged, not yet scored)
- **Hypothesis:** Distinct bets (exact-match loss, T=1 specialist, arithmetic tape, modulus memory, dyadic composition) should be controlled separately rather than freestyle-merged into Fable.
- **Setup:** Implemented five standalone `solving/submissions/*/submission.py` cards under evaluator pin `e32c2f9` (TokenLossBatch, multi-backward, batch reuse). All five pass `one-layer validate`. No hosted submit in this packaging step.
- **Cards:** `exact_match_optimizer` (Fable + sequence CE/worst-digit/margin/dual-head + 2-pass SAM + reuse); `t1_assassin` (pairwise x-grid × N cross-attn + refine + T router); `gated_arithmetic_tape` (local conv/tape microsteps); `learned_modulus_memory` (product-key memory + gate); `dyadic_semigroup` (level adapters + soft compose KL).
- **Next:** Hosted Easy e1 for cards 1→2→3 first; kill per design kill-conditions. Do not start Hard on any until T=1 local/hosted signal.

### 2026-08-07 — GPT-5 Pro five-card Easy e5 screen
- **Cards (labeled GPT-5 Pro):** exact_match_optimizer, t1_assassin, gated_arithmetic_tape, learned_modulus_memory, dyadic_semigroup.
- **Result (Easy e5 mean exact, Max T none on all):** exact_match **1.04
### 2026-08-07 — GPT-5 Pro five-card Easy e5 screen
- **Cards (labeled GPT-5 Pro):** exact_match_optimizer, t1_assassin, gated_arithmetic_tape, learned_modulus_memory, dyadic_semigroup.
- **Result (Easy e5 mean exact; Max T none on all):**
  - exact_match_optimizer **1.04%** (`aca0613e`)
  - dyadic_semigroup 0.75% (`a67a65f4`)
  - learned_modulus_memory 0.58% (`85dd2faa`)
  - t1_assassin 0.42% (`2cdf18a8`)
  - gated_arithmetic_tape 0.13% (`61694c82`)
- **Next:** Human picks Hard candidate; agent does not auto-submit Hard.

### 2026-08-08 — T=1 state-topology tournament and discrete refinement (Author: Codex)

- **Question:** before any new recurrence-depth work, does a structured
  LSD-aligned per-position state solve the T=1 transfer bottleneck, and can a
  discrete iterative refiner improve exactness cheaply?
- **Control:** the same synthetic small-N VDF harness, 18 seen moduli, 8
  unseen moduli, 80/20 held-out-x split on seen moduli, all unseen-N residues,
  AdamW, one seed, and 120 seconds per arm on the L40. Only T=1 rows were
  trained. Existing T>1 diagnostics were not extended.
- **Result:** register baseline was 4.62% held-out-x / 8.64% unseen-N; global
  latent was 10.50% / 16.36%; structured LSD tape was 12.18% / **17.06%**.
  Structured state is a narrow positive (+1.68/+0.70 points over global) but
  does not clear the promotion gate. It is 311,626 parameters versus 262,346
  for controls and runs fewer updates in the same wall-clock budget.
- **Discrete refiner:** the masked-token shared refinement model scored
  unseen-N 0.47%, 7.94%, 8.18%, and 8.88% at K=1,2,4,8 respectively, with
  increasing latency. This is refuted for promotion; it does not justify a
  competition screen.
- **Decision:** no Easy/Medium/Hard transfer. Freeze the structured result as
  research evidence and register a binary/limb T=1 representation comparison
  next. Do not run zero-shot T>1 recurrence until a T=1 model reaches a
  qualitatively stronger regime.
- **Artifact:** `diagnostics/artifacts/t1_tournament_2026-08-08/summary.md`.

### 2026-08-08 — T=1 numerical representation comparison (Author: Codex)

- **Question:** is the current low T=1 transfer ceiling primarily caused by
  decimal digit representation? This is a matched, final-label-only,
  single-step comparison; no T>1 training or recurrence evaluation was run.
- **Controls:** same 18 seen / 8 unseen modulus split, 80/20 held-out-x split,
  one seed, structured LSD-aligned local tape, AdamW, batch 512, 120-second
  L40 budget, and the same decoder/topology. The fixed limb choice was made
  before training: two little-endian 4-bit limbs (8 representable bits), with
  no post-hoc width tuning.
- **Results (full-train exact / held-out-x exact / unseen-N exact):** decimal
  100.00% / **11.76%** / **18.69%**; binary (7 little-endian bits) 7.81% /
  2.10% / 4.21%; 4-bit limbs 100.00% / 11.34% / 14.49%. Parameter counts
  were 261,962 / 260,770 / 262,864 respectively. Updates in 120 seconds were
  24,524 / 21,006 / 27,056. Held-out inference latency was 0.0091 / 0.0120 /
  0.0077 ms per example (local batch measurement).
- **Convergence:** decimal and limbs fit their seen training rows rapidly;
  binary remained near 8% full-train exact and never reached a fitting regime.
  The complete curves and raw reports are in
  `diagnostics/artifacts/t1_representation_2026-08-08/{decimal2,binary2,limb42}`.
- **Conclusion:** **refuted as a representation-only explanation** under this
  matched budget. Neither binary nor fixed 4-bit limbs produces a qualitative
  generalization improvement; both are worse than decimal on held-out-x and
  unseen-N. The representation branch is closed for now. This does not prove
  binary arithmetic can never work with a different objective or architecture;
  it says representation micro-tuning did not break the present final-label
  ceiling. Do not launch T>1 or competition runs from these arms before review.

### 2026-08-08 — T=1-weighted Hard execution (Author: Codex)

- **Change:** weight T=1 rows by 8x, normalized to mean-one row weight, inside
  the existing GPT-5 Pro exact-match/SAM card. No architecture, optimizer,
  recurrence, batch-reuse, or inference change.
- **Screen:** local L40 e5 reached 0.4583% mean and T=1 5/512 seen / 0/512
  OOD-N. Hosted Easy `cb98f944-9b21-4869-af5a-c924845ca89e` reached 0.3750%
  mean and T=1 3/512 seen / 0/512 OOD-N. The parent hosted Easy profile was
  2/512 seen / 1/512 OOD-N, so total T=1 hits did not improve.
- **Hard:** exact SHA-1 `8c796bf39f3b0d2f90043b08430be26c23f0f180` was
  accepted as `9e7404cb-b0c9-480a-aa64-8d90cc853d67`; daily Hard quota is
  exhausted. This is a weak first-profile bet, not a promoted mechanism.

### 2026-08-08 — T=1 phase information-flow control (Author: Codex)

- **Question:** does preventing the square phase from seeing `N` force an
  `N`-independent square representation under final-label-only training?
- **Control:** matched 443,594-parameter four-step square/four-step reduction
  tapes. The factored arm sees a learned null during square; the entangled arm
  sees `N`. Data, optimizer, wall time, depth, and all other computation match.
- **Result:** all six runs reached 100% train exact. Across seeds 0/1/2,
  factored median held-out-x/unseen-N exact was **11.76%/17.29%** versus
  **13.03%/17.29%** entangled.
- **Conclusion:** refuted. Information withholding alone does not identify
  squaring and slightly hurts held-out-x. Artifacts:
  `diagnostics/artifacts/t1_phase_square_reduce_2026-08-08/`.

### 2026-08-08 — T=1 pair-fold square tape (Author: Codex)

- **Question:** can the strongest structural multiplication bias recover
  upstream square learning from final modular labels: learned digit-pair
  categories, pair-to-column routing, within-column fold, and LSD carry scan?
- **Result:** all three 435,530-parameter runs reached 100% train exact, but
  median held-out-x/unseen-N exact was **10.50%/16.36%**, worse than the
  generic factored tape's 11.76%/17.29%.
- **Conclusion:** refuted. Directly supervised raw-square success does not
  transfer when its only gradient comes through learned reduction. Stop
  rearranging T=1 topology; the next card must change legal identifiability or
  credit assignment. Artifacts:
  `diagnostics/artifacts/t1_pairfold_square_reduce_2026-08-08/`.

### 2026-08-08 — Exact Hard-source local e5 replication (Author: Codex)

- **Identity:** local and L40 source SHA-1 both
  `8c796bf39f3b0d2f90043b08430be26c23f0f180`, exactly matching Hard job
  `9e7404cb-b0c9-480a-aa64-8d90cc853d67`; evaluator pin `e32c2f9`.
- **Result:** public Easy e5 on the idle L40 completed 1,192 updates in 60.03
  seconds. Test was 7/1,200 (0.5833%), OOD 4/600 (0.6667%), mean **0.6250%**.
  T=1 profiles were 4/512 seen-N and 1/512 OOD-N; no rung certified.
- **Interpretation:** no-change replication confirmed source/runtime integrity
  and the registered prior-local neighborhood. This card descends directly
  from `fable_tcap_adamw`; exact-match heads/loss/SAM/reuse were added next,
  and normalized 8x T=1 row weighting is the sole Hard-attempt change. It is
  unrelated to `t1_assassin` and the research-only pair-fold tape.
- **Artifact:**
  `diagnostics/artifacts/hard_exact_source_e5_replication_2026-08-08/stdout.log`.

### 2026-08-08 — T=1-weighted Hard final and L40 retirement (Author: Codex)

- **Hosted result:** job `9e7404cb-b0c9-480a-aa64-8d90cc853d67` succeeded
  after 60,915 updates / 3,600.05 training seconds. Overall exact was
  **0.02333%**: test 3/9,999, OOD-T 2/10,002, OOD-N 2/10,002.
- **Certification:** no seen-N or OOD-N rung certified. Both T=1 profiles were
  exactly **0/768**. The 8x T=1 weighting therefore fails its actual Hard goal
  despite nonzero public e5 T=1 hits.
- **Training behavior:** final loss was 2.4779 after plateauing near 2.47 for
  most of the hour. The longer budget did not uncover a reusable transition.
- **Hard artifact:** structured hosted metrics and full status are preserved
  under ignored `diagnostics/artifacts/hard_9e7404cb_result_2026-08-08/`.
- **GPU backup:** Prime pod `8b90919e9f4541f197a57f3493896542`
  (`somn-l40-2026-08-07`, IP `216.81.248.102`, L40 UUID
  `GPU-150b876a-ce24-da40-63a8-0580474bb736`) was idle. A cache-excluded
  backup plus three custom evaluator submission sources was verified at
  `diagnostics/artifacts/prime_backup_2026-08-08T0056Z_hard_complete/somn-taskb/`:
  **49 files and 838,421 bytes matched exactly**. Manifest verification time:
  `2026-08-08T00:58:51Z`.
- **Termination:** the verified manifest unlocked guarded termination of only
  pod `8b90919e9f4541f197a57f3493896542`. Prime then listed zero running pods;
  target status had no IP/SSH, and `ssh oneL40` timed out. Billing compute is
  retired. Recreatable virtual environments and Python caches were excluded.

### 2026-08-08 — Canonical register sprint and forced Hard selection (Author: Codex)

- **Mechanism:** initialize a mutable LSD-first register from `x` once; expose
  only that state plus immutable `N` to the tied cell; use `T` only as loop
  count; route the same state logits to the answer. Explicit small embedding
  initialization repaired initial CE from ~42 to 2.76. Final labels only.
- **Matched e5 ablation:** prompt-reinject curriculum was 1/512 seen + 1/512
  OOD-N T=1; canonical plain was 1/512 + 1/512; canonical curriculum was
  5/512 + 2/512. None passed the registered promotion gate.
- **Downward-transfer refutation:** full m1 (T=4/8/16 labels) stayed at CE
  2.2930 for 9,815 updates / 600.02 seconds and produced 0/192 seen plus 0/512
  OOD-N T=1. Canonical state alone does not identify a one-step root.
- **Compact selection:** dynamic active slots yielded exact-source hosted e5
  jobs `b99d4e4b-95f3-420c-ba08-282e4060d4d0` at 2/512 + 3/512 and
  `e0542460-87d9-49ef-aaa7-a691ac378414` at 5/512 + 4/512 (seen + OOD-N T=1).
  Width 128, batch 256, and LR 6e-3 each zeroed one first-rung profile.
- **Decision:** freeze SHA-1 `5b622f06680600f4b346e34b635b839dde18471c`
  for the owner's explicitly requested Hard attempt. This is a reproducible
  first-rung lottery, not a promoted mechanism or solved T=1 submission.
- **Submission:** exact frozen source passed validation and completed as Hard
  h1 job `7714d650-78a4-4d4a-8fc1-a384914d7658`. It scored 0.0500% mean exact
  (8/9,999 test, 2/10,002 OOD-T, 5/10,002 OOD-N), certified no rung, and was
  0/768 at T=1 on both seen-N and OOD-N profiles. Training completed 163,274
  updates / 3,600.01 seconds at final train loss 2.17846. The two hosted Easy
  profiles were chance-scale and did not transfer.
- **GPU retirement:** active L40S pod `aaa91aae061a42efb488ded82707752d`
  (GPU `GPU-ce5e12f8-947e-a813-9811-ba731e8defbf`) was idle. Backup verified
  21 files / 178,089 bytes under ignored
  `diagnostics/artifacts/prime_backup_2026-08-08T2227Z_canonical_sprint/`
  before guarded termination. A separate original L40 pod
  `f9df27bd9eac40be95525d58448521e2` remained stuck in provisioning with no
  IP/SSH and was not terminated because no real backup could be verified.

### 2026-08-09 — Interface noise does not repair T=1 credit assignment (Author: Codex)

- **Question:** does training-time noise at the square/reduce boundary suppress
  a brittle example-specific hidden code and favor a reusable square state?
- **Control:** relative to the public-E5 factored seed-0 anchor, change only a
  Gaussian perturbation with std 0.1 after square and before reduction. Data,
  parameters, optimizer, seed, final-label loss, evaluation, and clock match.
- **Result:** 16,677 updates retained **1,599/1,600 train exact**, but seen-N
  T=1 fell to **2/512** and OOD-N T=1 to **0/512**, versus anchor 7/512 and
  1/512. The registered kill threshold fired.
- **Conclusion:** refuted. Smooth interface robustness is insufficient; do not
  sweep noise strength. Any next credit intervention must be qualitatively
  discrete or impose a cross-example constraint. Evidence:
  `experiments/2026-08-09_t1_factored_e5_interface_noise/NOTE.md`.

### 2026-08-09 — Public-E5 support does not identify T=1 modular square (Author: Codex)

- **Question:** was the generic factored tape's weak unseen-N result mainly a
  consequence of training on only 18 tiny two-digit moduli?
- **Control:** keep its 443,594 parameters, four square/four reduction steps,
  optimizer, seed, final-label-only loss, and 180-second clock fixed; replace
  only training/evaluation support with public Easy e5 T=1 rows.
- **Result:** 18,019 L40 updates fit all 1,600 training rows exactly, but
  public seen-N T=1 was **7/512 (1.3672%)** and OOD-N T=1 was **1/512
  (0.1953%)**. Both registered kill thresholds fired.
- **Conclusion:** strongly refuted. More modulus diversity strengthens exact
  memorization without identifying a reusable modular-square rule. Do not run
  more seeds or another support-only expansion. Evidence is linked from
  `experiments/2026-08-09_t1_factored_e5_support/NOTE.md`.

### 2026-08-09 — Complete hosted Easy/Medium dataset ladder (Author: Codex)

- **Control:** exact Fable T-cap/AdamW source SHA-1
  `aa75819a878fab6c03c6a23d979f6234560f6e3d` across every public dataset.
  Existing anchors were e1/e5/m1/m5; today filled e2/e3/e4/m2/m3/m4.
- **New results:** e2 **1.2083%**, e3 **0.5000%**, e4 **0.2708%**, m2
  **0.1500%**, m3 **0.2667%**, and m4 **0.0778%** mean exact. All jobs
  succeeded and none certified a rung. m3 completed 19,424 updates without
  exceeding Easy-scale chance accuracy.
- **Conclusion:** fixed N is insufficient, fixed T=2 is insufficient, and a
  10x longer clock is insufficient. The score degrades with modulus variation
  and size before deep recurrence becomes the central issue. Evidence and plot:
  `experiments/2026-08-09_easy_medium_completion_suite/NOTE.md`.

# Predictions (append-only)

Human writes PREDICT before the run. Agent never writes the prediction.
Format: [`../../RESEARCH_PROTOCOL.md`](../../RESEARCH_PROTOCOL.md) §1.

---

<!-- template

### YYYY-MM-DD

```
CARD:
CHANGE:
PREDICT:
RESULT:     confirmed | refuted | unclear   <one sentence>
```

-->

### 2026-07-22

```
CARD:       claude_std_rope_e1
CHANGE:     standard non-recurrent 4-layer Transformer, plain token embedding,
            RoPE instead of absolute+depth embeddings (vs depth_d32_k4_ut anchor)
PREDICT:    With enough width and training time, a pure (non-recurrent)
            Transformer should be able to memorize the in-context digit
            mod/squaring operation itself as a fixed NxN mapping — it is not
            recurrent, so it cannot learn the general iterated map. On the
            T-extrapolation curve (train T in {1,2,3}, eval T=4/5/8/16) we
            should therefore expect exact-match to go to 0 at unseen T:
            no mechanism exists in the architecture to apply "one more step"
            beyond what depth/width let it memorize for the trained T's.
RESULT:     unclear — ood (T=4, the only unseen-T split e1 has) scored 7.00%,
            below the 9.94% majority-class baseline, so no real
            generalization signal (consistent with "expect ~0"). But train
            hit 100% while held-out same-T test scored only 2.67% — severe
            overfitting to exact training examples, not the in-context
            digit-mod memorization the prediction described. See NOTE.md.
```

### 2026-07-22 (b)

```
CARD:       claude_abacus_e1 / claude_fire_e1 / claude_fireabacus_e1
CHANGE:     swap RoPE (claude_std_rope_e1's position scheme) for, in turn:
            (1) Abacus place-value embedding alone, (2) FIRE relative
            attention bias alone, (3) both together. Same base: 4 independent
            layers, d=32, heads=4, plain token embedding otherwise.
PREDICT:    "my hypothesis is that it allows us to learn addition and
            multiplication easily and we see generalization in this same
            regime" — i.e. abacus/FIRE embeddings should let the model learn
            the digit-mod/squaring operation itself (not just memorize
            examples), producing real generalization in the same regime
            claude_std_rope_e1 failed at (test 2.67%, ood 7.00%).
RESULT:     refuted — all three underperformed the plain RoPE anchor on
            both splits. abacus test 1.33%/ood 6.00% (never fully
            memorized train, unlike the other three); fire test 0.67%/
            ood 3.00%; fire+abacus test 0.67%/ood 2.00% (worst ood of all
            four cards, despite being the paper's own strongest combo).
            See NOTE.md in each 2026-07-22_claude_{abacus,fire,fireabacus}_e1/.
```

### 2026-07-22 (c)

```
CARD:       claude_std_rope_e1 (unmodified anchor), on a new T=1-only,
            varying-N probe manifest filtered from e5
CHANGE:     no architecture/embedding change — isolates whether the block
            can do ONE step of modular squaring for held-out (u, N) at all,
            separate from composition (T>1) and from embedding choice.
            e3 excluded: fixed_time_steps=2, has zero T=1 rows. Filtered
            e5's train/test to time_steps==1 only (1600 train / 400 test,
            N varying) — no ood split included (e5's ood is T=6, irrelevant
            to this probe).
PREDICT:    "if the block has any workable mechanism for reduction mod N,
            T=1 exact-match on e3/e5 test split should be high (>90%); if
            the failure is representational, it will be low regardless of
            embedding."
RESULT:     confirmed (LOW branch) — 0.75% test exact-match at T=1 on
            held-out (u,N), vs train 100% (memorized). The block cannot
            represent one modular-squaring step for unseen N even with no
            composition and no embedding ambiguity (every row is T=1).
            See NOTE.md in 2026-07-22_t1only_probe_rope/.
```

### 2026-07-22 (d)

```
CARD:       depth_d32_k4_ut and depth_d32_k8_ut (weight-tied UT loop,
            absolute + depth embedding — yesterday's champion recipe,
            unmodified) on the T=1-only probe manifest
CHANGE:     swap the T=1 probe's model from 4 untied layers (RoPE) to a
            weight-tied K-loop (UT depth embedding, input re-injected each
            iteration implicitly via the shared block), K=4 and K=8.
            Everything else unchanged from the t1only_probe_rope run.
PREDICT:    if Newton-Raphson-depth (more rounds of the same
            multiply-capable operation) is the real gap, exact-match should
            jump well above 0.75% at K>=4. If it stays near-zero even with
            looping, that's evidence for the harder conclusion — a
            qualitatively different mechanism is needed, not just more
            rounds.
RESULT:     unclear — confounded, not a clean test. K=4: test 0.25%, K=8:
            test 0.50% (both still worse than flat's 0.75%, but neither
            card had finished fitting train in 60s — K=4 train loss 0.674 /
            ~42.5% exact-match still climbing at cutoff, K=8 train loss
            0.851 / ~35-50% still climbing. Flat anchor fully converged
            (loss 7e-6, 100% train) by step 1700 in the same window. A
            weight-tied K-loop does K forward passes per optimizer step, so
            it gets proportionally fewer effective updates per second —
            this test needs matched wall-clock (or matched step count) to
            be interpretable, not matched seconds. See NOTE.md in both
            2026-07-22_t1only_probe_ut_k4/ and _k8/.
```

### 2026-07-22 (e)

```
CARD:       depth_d32_k4_ut and depth_d32_k8_ut, wall-clock scheduler bug
            found and fixed, rerun on the T=1-only probe
CHANGE:     patched _build_scheduler in both cards — replaced
            `t_max = training_time_seconds * 8` (CosineAnnealingLR, uncapped
            past T_max, cycles every ~7600 steps at this box's ~48.7 steps/s)
            with the validated wall-clock scheduler. Same bug as
            15-lr-schedules-wallclock.md, reappeared because this file was
            never patched. Reran K=4 (300s) and K=8 (500s) to full train
            convergence with monitor_train.py.
PREDICT:    (carried over from (d)) if Newton-Raphson-depth is the real gap,
            exact-match should jump well above 0.75% at K>=4 once training
            actually converges. If it stays near-zero, that's evidence for
            the harder conclusion.
RESULT:     refuted, cleanly this time. K=4 fixed: 14,500 steps, train 100%
            (loss 1.5e-4), test 0.50% final / 1.25% max ever seen. K=8
            fixed: 21,750 steps, train 100% (loss 1.7e-5), test 0.75% final
            / 1.50% max ever seen. Both land in the same noise band as the
            flat anchor (0.75%). Looping — K=4 or K=8, weight-tied,
            depth-embedded — does not close the T=1 gap even with full
            convergence and a correctly-annealing schedule. See NOTE.md in
            both 2026-07-22_t1only_probe_ut_k4/ and _k8/, and the published
            chart (schedule sawtooth vs. fixed curves, both cards).
```

### 2026-07-22 (f)

```
CARD:       fable_hard_h1_adamw and fable_hard_h1_muon — same architecture
            as fable_hard_h1, optimizer swapped
CHANGE:     fable_hard_h1's flat lr=3e-4 WarmupSchedule (never decays,
            loss stuck flat on both e5-60s and m5-600s local runs) replaced
            with (a) our validated AdamW lr=3e-3/wd=0.1 + wall-clock
            schedule, and (b) Muon (hidden weight matrices) + AdamW
            (embeddings/norms) hybrid, wall-clock schedule.
PREDICT:    "use our optimizer or try muon :eyes: or both lmao" — if the
            original flat loss was purely an optimizer problem, either fix
            should get the model actually learning again.
RESULT:     confirmed, and sharper than expected. AdamW-fixed: loss moves
            (2.9->1.82 on e5/60s) but doesn't converge in the window. Muon:
            full train convergence in 3,616 steps on e5 (loss ~1e-5, 100%
            exact-match) — dramatically faster than AdamW-fixed — and
            ood 2.0%, the best number on e5 all session. But m5 (600s,
            10x): Muon goes completely flat again, same pathology as the
            ORIGINAL broken run (loss stuck ~2.1-2.2, train never above
            ~1%, test 0.12%/ood 0.07%) — Muon didn't just fail to help at
            m5, it failed as badly as no fix at all. Unconfirmed hypothesis:
            m5's T range (up to 8) roughly doubles effective unroll depth
            vs. e5 (up to 3), interacting badly with the harness's global
            grad_clip=1 and Muon's untested-at-that-depth lr=0.02. See
            NOTE.md in both 2026-07-22_fable_hard_h1_adamw/ and _muon/.
```

### 2026-07-22 (g)

```
CARD:       fable_max — new Hard-tier architecture (T-proportional
            weight-tied FiLM loop, register re-quantization)
CHANGE:     new card, audited against ban list + harness contract before
            running (checked for the two bugs already found this session:
            step-count-vs-wall-clock scheduler, interface mismatches).
            Tested on e5 only so far.
PREDICT:    "verifies this works, matches our spec and previous errors then
            try it" — if the audit passes, it should run clean; whether it
            learns is the open question.
RESULT:     unclear. Audit passed (contract clean, no crash). Real learning
            signal: loss 3.13->0.29, train accuracy ->80% by step 1100 —
            fastest clean learning curve of the day. Then collapses hard at
            step 1200-1300 (loss back to ~2.18, accuracy ~0%, stays
            collapsed). Collapse timing lines up almost exactly with
            progress crossing 0.55, where forward() switches quantization
            from soft to hard straight-through — looks like a schedule
            artifact (abrupt regime change destroying a good soft solution)
            rather than an architectural dead end. See NOTE.md in
            2026-07-22_fable_max/.
```

### 2026-07-22 (h)

```
CARD:       fable_max_nohardst and fable_max_notheta — clean single-variable
            ablations off the ORIGINAL fable_max (not fable_max_smooth, to
            avoid its compute-cost confound)
CHANGE:     nohardst: hard_st forced False the whole run (pure soft
            quantization throughout), theta anneal unchanged. notheta: theta
            frozen at 1.0, hard_st's discrete prog>0.55 jump unchanged.
            First nohardst run was GPU-contended (concurrent m5 background
            job) and discarded; both final runs GPU-uncontended.
PREDICT:    "try removing whichever of the two was causing issues and using
            it only as a test as well" — isolating hard_st vs theta should
            show one variant collapsing and one not, identifying the cause.
RESULT:     refuted the premise that it's either of the two named suspects.
            Both ablations still collapse, at the same progress point
            (~0.74-0.76) as the unmodified original and as each other:
            nohardst collapses at step 1300 (loss 0.72->2.15, had reached
            47.9% train accuracy); notheta collapses at step 1300 too (loss
            1.99->2.15), with weaker peak performance throughout (never
            below loss 1.89). Neither hard_st nor theta is the sole cause.
            The one schedule variable common to all three collapsing runs
            and not yet tested: alpha (register-blend strength), which
            hits its ceiling (1.0, full state replacement instead of a
            blend) at exactly this progress point in every run. See
            NOTE.md in 2026-07-22_fable_max_nohardst/ and _notheta/, and
            the updated 2026-07-22_fable_max/NOTE.md (m5 result: same flat
            pathology over 600s, does not recover with more time).
```

### 2026-07-22 (i)

```
CARD:       fable_max_wd1 — weight_decay 0.1 -> 1.0, monitored, e5
            overtrained to 500s
CHANGE:     first monitored (periodic held-out eval) run of the fable_max
            family. Also first weight-decay test all session (note 17
            priority #1, previously untested).
PREDICT:    "try 1 for now. or try overtraining or both!" — raising wd
            should suppress the overfitting seen in the unmodified card's
            monitored run (train down, test up); overtraining tests whether
            more time helps regardless.
RESULT:     confirmed a real dynamics change, refuted a better ceiling.
            wd=1.0 overfits DEEPER before intervening (test loss peak 2.98
            vs wd=0.1's 2.52), then shows genuine slow generalization
            (test loss 2.9->2.27 over steps ~1800-9600 while train stays
            fit, the grokking shape, first time seen outside the Hard run)
            — then a second schedule-triggered collapse at step 9800
            (progress~0.708, same ~0.7 threshold as every 60s run's single
            collapse) settles test loss at 2.171, the best of the day, but
            only ~0.01 better than every other variant's floor (~2.17-2.18).
            Every fable_max variant tested today — different wd, different
            optimizer, different quantization ablations — converges to the
            same ~2.17-2.18 test loss once past the schedule's ~0.7
            threshold. Matches the T=1 probe's finding (block can't
            represent one step of mod reduction for held-out N) — this is
            an architecture ceiling, not an optimizer/schedule one. See
            NOTE.md in 2026-07-22_fable_max_wd1/.
```

### 2026-07-23 (a)

```
CARD:       claude_std_rope_e1 (unmodified, wd=0.1) on new fixed-N x-split
            T=1 datasets — P2 ladder rung 1 (N=323 and N=1073)
CHANGE:     dataset only: split_group=x, fixed semiprime, all units mod N,
            80/20 by x, T=1, separate_input_output=true. (N=323 variant
            uses batch_size 115 so drop_last keeps a batch; nothing else.)
PREDICT:    [carried from FULL_TRANSCRIPT.md P2 per START_PROMPT — human to
            countersign] rung 1 should be learnable (grokking literature,
            Power et al. on modular arithmetic); the transcript put the
            rung1→rung2 transition as the likely failure (P≈0.7 fail at
            rung 2), implying rung 1 itself well above floor.
RESULT:     unclear, leaning refuted for this width/budget — N=323: train
            100% at 86.5k steps, test 5.17% final / 6.90% peak, a real but
            tiny grokking-shaped climb. N=1073: train 100%, test 0.00%.
            Even one fixed modulus at T=1 barely moves off floor. See
            2026-07-23_t1only_fixedn_wd01/NOTE.md.
```

### 2026-07-23 (b)

```
CARD:       claude_std_rope_e1_wd1 (+ _b115_wd1) — same rung-1 datasets,
            weight_decay 0.1 → 1.0
CHANGE:     one variable: AdamW weight_decay to 1.0 (L4: the grokking knob).
PREDICT:    [agent-proposed, human to countersign] wd=1.0 should suppress
            pure memorization and raise held-out EM above the wd=0.1 runs.
RESULT:     refuted — wd=1.0 crushes weight norms (~15 vs ~30-35) before
            train ever fits (61% / 31% train EM); test 1.72% / 0.00%. D1
            per-position accuracy ≈ train-marginal baseline everywhere.
            See 2026-07-23_t1only_fixedn_wd1/NOTE.md.
```

### 2026-07-23 (c)

```
CARD:       claude_std_rope_e1 (+_b115), rung-1 datasets, budget 900s → 1800s
CHANGE:     one variable: total_training_time_seconds doubled (wallclock
            scheduler stretches the anneal horizon accordingly).
PREDICT:    [agent-proposed, human to countersign] the N=323 slow climb was
            still rising when the 900s LR annealed out, so 2x budget should
            let it continue past 7%.
RESULT:     refuted in an informative way — peaks rose (N=323: 8.62% @ 146k
            steps, N=1073: 3.47% @ 22k) but both runs then DECAYED back to
            1.5-1.7% by end of anneal; D1 on final checkpoints = marginal
            baseline at every position. The climb is real but non-monotone
            and is destroyed late in training rather than consolidated.
            See 2026-07-23_t1only_fixedn_wd01/NOTE.md addendum.
```

### 2026-07-23 (d)

```
CARD:       pathd_digit_microscan
CHANGE:     parallel register update → learned LSD-to-MSD discrete-carry scan
PREDICT:    no improvement in T
RESULT:     confirmed
```

### 2026-07-23 (h)

```
CARD:       gate1_aligned_products
CHANGE:     one digit-pair product → repeated fixed-width local products without cross-position carry
PREDICT:    collapse
RESULT:     confirmed
```

### 2026-07-23 (g)

```
CARD:       gate1_digit_product
CHANGE:     exact multi-digit squaring → one decimal digit pair product without carries
PREDICT:    fails held out pairs. ngl multiplication might be a memorization task? thats how i feel
RESULT:     confirmed
```

### 2026-07-23 (e)

```
CARD:       gate0_copy
CHANGE:     modular-squaring target → exact copy of X
PREDICT:    fail on the first digit. pass on the last three. or first three you get what i mean
RESULT:     refuted
```

### 2026-07-23 (f)

```
CARD:       gate1_square
CHANGE:     copy X → exact x² without modular reduction
PREDICT:    didgit multiplication is hard iirc for normal transformers. everything can be learnt well but im sure we arent generalizaing
RESULT:     confirmed
```

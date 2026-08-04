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
CARD:       solving/research/soft_digit_squaring_recurrence.py — one-step
            four-digit squaring gate (Codex's card, `9c9cbc7`), rerun with a
            wall-clock LR schedule instead of the original step-count one.
CHANGE:     one variable: build_optimizer's LambdaLR was step-count-based
            (TOTAL_STEPS=10_000, WARMUP_STEPS=500) against a manifest whose
            real budget is total_training_time_seconds=600 — the schedule
            assumed ~10k steps would fit in 600s. Replaced with the
            time.monotonic()-based wall-clock scheduler already validated
            elsewhere in this project (WARMUP_FRACTION=0.05,
            FINAL_LR_FRACTION=0.05). Model, data, seed, batch size, and
            budget (600s) all unchanged. Also completed Codex's own
            in-progress revert of the refuted symmetric pair table (was
            already correct on disk, uncommitted) as part of the same diff.
PREDICT:    [agent-proposed, human to countersign] Codex's own log shows this
            exact baseline peaking at 85.35% (step 3,500) then decaying to
            83.5-83.55% (steps 4,000-4,500) — right where the original
            schedule still had ~65% of peak LR remaining (10k-step cosine,
            only ~4-4.5k steps actually fit in 600s). Every one of Codex's 5
            follow-ups changed the model and was refuted; none touched the
            schedule. If the decay is schedule-driven (LR still too high
            post-peak, not a representational ceiling), annealing to the
            same FINAL_LR_FRACTION floor by the time the peak historically
            occurred should let the run consolidate near 85%+ instead of
            drifting past it.
RESULT:     partially confirmed, usefully refuted. The decay is schedule-
            driven and IS eliminated — the wall-clock run is monotone: peak
            82.55% @ step 6,700 == final 82.55% @ step 7,500 (no overshoot,
            no drift). BUT the peak is ~3 pts BELOW the original's transient
            85.35% @ step 3,500. Reading: the original schedule's high-LR
            "bug" let it briefly bounce to 85.35% before knocking itself
            back to 83.55% — that spike was an unstable transient the model
            could not hold, not a recoverable solution. The fix trades the
            unstable 85% spike for a stable 82.55% floor. Net: no peak-accuracy
            gain, but a trustworthy checkpoint (peak == final, so the saved
            weights ARE the good ones). The real blocker — ~15-17% one-step
            error concentrated on decimal output digit 3 — is untouched, same
            as all 5 of Codex's refuted model-side follow-ups. Schedule was
            not the missing piece. Metrics: twoA6000:/tmp/soft_digit_wallclock_
            monitor.jsonl; peak ckpt twoA6000:/tmp/soft_digit_wallclock_monitor_peak.pt.
```

### 2026-07-24

```
CARD:       learned_reduction_cell — fixed N=323, held-out x, T=1
            (solving/DESIGN_NEXT.md OPT 1, rung 1)
CHANGE:     new mechanism: learned schoolbook long division (recurrent
            remainder + attention over N's digits + learned quotient head +
            learned subtract), fed by the already-validated multiply cell's
            UNTRUNCATED product (previously discarded above digit 4). Only
            the final remainder is supervised; every quotient guess along
            the sweep is unsupervised. Normal (step-count) schedule, wall-
            clock deferred per user instruction. 453,298 params.
PREDICT:    good generalization but poor train/test — i.e. expect a small
            train-test gap (whatever it learns should transfer to held-out
            x cleanly, no memorization gap) even if both numbers end up low,
            because the credit-assignment problem (division-shaped, only
            final-answer supervision) is much harder than the multiply
            cell's direct supervision.
RESULT:     confirmed the "poor train/test" half, refuted the "good
            generalization" half. Train 100%, test PEAKED 5.17% at step
            4,800 then decayed to 1.72% (exactly the digit-marginal
            baseline) by step 25,000 — not a small clean gap, a full
            decay to floor. dump_predictions.py on the peak checkpoint:
            80% of wrong predictions were still valid quadratic residues
            mod 323 (plausible but misassigned), avg |true-pred|~120 (not
            close numeric misses). Composed supervision (squaring+
            reduction, final remainder only) does not clearly separate
            which stage fails. See isolation tests below, same day.
```

### 2026-07-24 (b)

```
CARD:       pure_squaring_cell — multiply-and-carry cell ONLY, no modulus
            at all (competition token format, N present but unused, label
            = plain x², 8 digits, x uniform 0-9999, held-out x)
CHANGE:     isolates squaring from reduction entirely, to find the true
            ceiling of the validated multiply/carry mechanism at full
            output width (previous "97.8%" figure was on a truncated
            mod-10^4/4-digit sub-problem only).
PREDICT:    [agent-proposed under delegated authority] full 8-digit
            squaring should land well below 97.8% given the truncated
            figure was on an easier sub-problem, but should still clear
            the digit-marginal floor by a wide margin if the mechanism
            generalizes at all.
RESULT:     confirmed — train 100%, test STABLE at 18.65% peak (step
            13,500) / 18.25% final (step 18,800), small gap, no decay.
            Implied per-digit accuracy ≈ 0.18^(1/8) ≈ 81%. This is the
            corrected ceiling for full-width unmoded squaring: the
            mechanism generalizes for real, but 97.8% was never the
            right number for the actual competition-scale problem.
```

### 2026-07-24 (c)

```
CARD:       pure_reduction_cell v1 — reduction mechanism ONLY (learned
            quotient+subtract over attention to N's digits), P fed
            directly from tokens (arbitrary 8-digit int, NOT x² of
            anything), N=323 fixed, label = P mod N, fully supervised,
            P sampled uniformly, held-out P.
CHANGE:     isolates reduction from squaring entirely and from the
            composed test's weak (final-remainder-only) supervision —
            P is a direct, fully-observed input this time.
PREDICT:    [agent-proposed under delegated authority] decoupled from
            squaring's unreliability and given full direct supervision,
            reduction should generalize better than the composed test's
            5.17% peak.
RESULT:     refuted, and worse than the composed test — peak only 0.60%
            (steps 700-2,000) on 2,000 held-out P, BELOW the 1.72%
            digit-marginal baseline. Train reached ~100%. Clean, isolated
            negative result: uniform-P reduction, even fully decoupled
            and fully supervised, shows essentially no generalization.
```

### 2026-07-24 (d)

```
CARD:       pure_reduction_cell v2 — same task as v1, three changes
            bundled per explicit user delegation ("Option 3"): P sampled
            from the reciprocal/log-uniform distribution (arXiv
            2506.23679 appendix A.1, derived and verified this session)
            instead of uniformly, weight_decay 0.01 -> 1.0, budget 600s
            -> 1800s (max_steps 80,000).
CHANGE:     three variables at once (bundled deliberately for speed per
            user instruction, not the usual one-variable rule).
PREDICT:    [agent-proposed under delegated authority] the reciprocal
            distribution's small-P skew plus a real grokking-scale wd
            and budget should meaningfully beat v1's ≤0.60% floor; unclear
            whether the improvement would be a genuine reduction skill or
            an artifact of oversampling trivially-small P (P<N needs no
            real reduction).
RESULT:     confirmed, decisively, and NOT a trivial-P artifact. Full
            80,000-step run completed in 1,251.9s (well under the 1,800s
            budget, ran to convergence on its own schedule). Peak 78.45%
            at step 26,200. Unlike every other experiment this session,
            it did NOT decay afterward — final window (steps 78,000-
            80,000) fluctuates 69-84%, holding a stable noisy plateau
            through step 80,000 (60k more steps with no collapse).
            Confound-checked at an earlier checkpoint (step 12,000, 64.5%
            aggregate): only 65/2,000 (3.2%) of test P values are <323
            (trivial, because the generator deduplicates sampled values,
            capping small-P representation near the true 323-value
            ceiling) — split-by-difficulty gave 95.4% on the trivial
            P<323 subset and 75.5% on the genuine P>=323 subset. This is
            the best and most stable positive result of the session: pure
            modular reduction (no squaring) generalizes for real, given
            (a) an operand distribution that emphasizes small values the
            way real x²-derived P's would concentrate near multiples of N,
            and (b) grokking-scale weight decay/budget. See
            solving/research/pure_reduction_cell_v2.py and
            generate_pure_reduction_v2.py.

DATE:       2026-07-25
CARD:       fable_tcap_adamw
CHANGE:     Merge last Fable Hard arch with timeout-safe T handling:
            parse T from prompt; TRAIN_LOOP_CAP=16 in train(); full
            min(T,64) in eval(); swap Muon→AdamW+wallclock;
            eval_batch_size=1024. File:
            solving/submissions/fable_tcap_adamw/submission.py
PREDICT:    Medium m5 completes without wall-clock death; Hard h1 at least
            accepts and finishes (no timeout like muon 5b363135). Exact% /
            Max T TBD — primary check is completion under new depth profile.
RESULT:     Medium m5 0.25% mean (aa699c3f; test 0.20 / ood 0.30). Hard h1
            succeeded — no timeout — score 0.03% mean exact (f4246e70).
            Timeout hypothesis confirmed; learning still at floor.
```
DATE:       2026-07-25
CARD:       claude_ut_k4_carry_aux_e5
CHANGE:     Add a learned two-scalar auxiliary head to the validated UT-K4
            STE anchor. It predicts aggregate schoolbook-squaring carry count
            and maximum carry from X-token hidden states; primary logits and
            recurrence are unchanged.
PREDICT:    Carry supervision should beat the STE e5 anchor's 0.50% if the
            Task-A mechanism transfers within Easy's short budget, but may
            remain below the continuous UT champion because the diagnostic
            benefit emerged over tens of thousands of steps.
RESULT:     confirmed relative to the STE anchor but not promoted as-is:
            e5 mean 0.75% (test 0.80%, OOD 0.70%), job 8e5457ad. Python
            input_ids.tolist() synchronization reduced throughput to 1,349
            steps versus the STE anchor's 2,521, confounding the mechanism
            with a severe optimization-budget loss.

DATE:       2026-07-25
CARD:       claude_ut_k4_carry_aux_tensorized_e5
CHANGE:     Implementation-only control: replace CPU/Python decoding and
            carry-target generation with tensorized GPU operations. Auxiliary
            targets, weight, model, optimizer, data, and metric stay fixed.
PREDICT:    The e5 curve should retain or improve the 0.75% score while
            recovering much of the STE anchor's ~2.5k-step throughput. If it
            does, Medium's 10-minute budget should be a fairer test of whether
            carry supervision improves held-out X/N learning.
RESULT:     refuted — tensorization recovered throughput (2,179 steps versus
            1,349) but e5 mean fell to 0.58% (test 0.80%, OOD 0.30%), job
            4eff4824. Aggregate carry supervision does not rescue the STE
            bottleneck.

DATE:       2026-07-25
CARD:       claude_ut_k4_continuous_carry_aux_e5
CHANGE:     Remove only the refuted STE token snap between UT loops. The
            tensorized carry auxiliary head, loss, data, optimizer, and four
            learned processing blocks stay fixed.
PREDICT:    Continuous state should recover the stronger UT-K4 baseline while
            retaining any benefit from carry supervision, beating 0.58% and
            plausibly matching or exceeding the 1.00% e5 champion. Failure
            below 1.00% means the aggregate auxiliary target is not useful
            enough for hosted variable-N learning.
RESULT:     refuted — e5 mean 0.38%, job 16ebb553. Aggregate carry
            supervision is rejected for hosted use; it does not transfer the
            per-column offline mechanism.

DATE:       2026-07-25
CARD:       pair_n_interaction_e5
CHANGE:     Replace recurrent/weight-tied UT with a non-recurrent four-block
            Transformer containing one learned pairwise-X interaction stage
            and content-dependent attention from that pair representation to
            N-digit states. No arithmetic operation or routing is hard-coded.
PREDICT:    If one-step held-X/N failure is caused by a plain Transformer's
            inability to expose the two required operands, this learned
            pair/N interface should beat the 1.00% e5 champion. A floor result
            means final-answer supervision still cannot identify reduction.
DATE 2026-07-25
CARD pair_n_carry_aux_e5
CHANGE add per-column first-square carry-in/out supervision to the otherwise identical non-recurrent pair/N model.
PREDICT exceed the 0.71% pair/N baseline and 1.00% e5 reference if carry-state identification transfers to held u,N; floor means first-square supervision does not reach modular reduction within the Easy budget.
DATE 2026-07-25
CARD pair_n_t1_objective_e5
CHANGE keep the non-recurrent pair/N model fixed but stop gradients from T=2,3 examples, scaling T=1 gradients to preserve magnitude.
PREDICT improve the latent T=1 slice enough to beat the 0.71% pair/N aggregate; failure means even the isolated one-step map remains memorization-bound.
DATE 2026-07-25
CARD pair_n_multiblock_supervision_e5
CHANGE replace final-block-only logits with the mean of shared-head logits after every distinct block; architecture, data, and optimizer otherwise fixed.
PREDICT beat the 0.71% pair/N baseline if shorter gradient paths make the arithmetic rule identifiable; floor means answer-only deep supervision is insufficient.
DATE 2026-07-25
CARD pair_n_multiblock_supervision_h1
CHANGE promote the exact e5-validated multi-block-supervision file to Hard.
PREDICT certified Max T remains 0, but T=1 partial exact accuracy may exceed the prior 0.03% Hard reference if the held-N one-step mechanism transfers; no T>1 extrapolation is expected without recurrence.
DATE 2026-07-25
CARD pair_n_multiblock_supervision_e5_rep1
CHANGE exact hosted replication; no code or configuration change.
PREDICT remain at or above 0.71% if the gain is directional; below 0.5% classifies the 1.04% run as e5 noise.

DATE:       2026-07-23
CARD:       gate1_quantized_carry_scan
CHANGE:     continuous shared carry state → 64 learned categorical state prototypes after every transition
PREDICT:    [agent-authored per human override] held-out exact match exceeds 95% and c6/c7 improve materially because hard prototype projection prevents continuous state drift from compounding
RESULT:     refuted — the hard state projection collapsed optimization (0.25% peak exact).

DATE:       2026-07-23
CARD:       gate1_aligned_products
CHANGE:     one digit-pair product → repeated fixed-width local products without cross-position carry
PREDICT:    collapse
RESULT:     confirmed

DATE:       2026-07-23
CARD:       gate1_digit_product
CHANGE:     exact multi-digit squaring → one decimal digit pair product without carries
PREDICT:    fails held-out pairs; multiplication may be a memorization task
RESULT:     confirmed

DATE:       2026-07-23
CARD:       gate0_copy
CHANGE:     modular-squaring target → exact copy of X
PREDICT:    fail on the first digit; pass on the other three
RESULT:     refuted

DATE:       2026-07-23
CARD:       gate1_square
CHANGE:     copy X → exact x² without modular reduction
PREDICT:    digit multiplication will not generalize
RESULT:     confirmed

CARD:       taskb_input_conditioned_workspace
CHANGE:     Fixed learned K=8 workspace initialization becomes one learned
            ordered-input cross-attention read using the existing transition
            attention; shuffled-context uses the same read from a deterministic
            non-identity row-level derangement.
PREDICT:    Correct context will improve held-out-u exact match, especially
            q>=10, while shuffled context remains near the fixed-workspace
            result; otherwise the fixed initializer is not the primary
            bottleneck.

DATE:       2026-08-04
CARD:       taskb_workspace_depth_audit
CHANGE:     Evaluation-only override of the retained ordered-context K=8 peak
            checkpoints at K={1,2,4,8}; model weights, data rows, and
            initialization context remain fixed.
PREDICT:    If the tied workspace is carrying out useful iterative reduction,
            K→2K will selectively repair q>=4 rows more often than it breaks
            them. If it is merely a depth effect, exact match will peak early
            or aggregate gains will not concentrate in difficult q buckets.
RESULT:     confirmed, bounded — K=8 is best on all three retained peak
            checkpoints (55.98±4.23% exact vs 0.63±0.38% at K=1); q>=4 rises
            0.53±0.41%→53.87±4.41%, and every K→2K pair has net repairs.
            This demonstrates useful extra computation, not an identified
            general remainder recurrence.

DATE:       2026-08-04
CARD:       taskb_reduction_capability_ladder
CHANGE:     Replace the single broad fixed-N reduction diagnostic with five
            fixed-size, disjoint operand regimes B0–B4; architectures A/B/C
            are matched controls and all optimizer/training settings are fixed.
PREDICT:    All models pass B0; B1/B2 expose a subtraction/borrow or small-q
            boundary; B3 should be easier than B4 if the broad-u diagnostic is
            distribution-mismatched. A recurrent advantage is meaningful only
            if it concentrates in q>=4 rather than q=0.
NOTE:       Screening horizon set to 2,000 fixed updates before any valid
            metric was recorded. The original 20,000-update queue was stopped
            before step 200 after measured throughput made the full 15-cell
            matrix impractical; batch size, optimizer, and all other settings
            remain fixed across cells.
RESULT:     refuted — every architecture solves B0 (100.00%), B1
            (98.83–99.61%), and B2 (92.58–96.88%) held-out, but B3 square
            operands are 0.00–0.39% despite 42.50–64.00% train exact. B4
            reaches 16.80–20.70%, but almost entirely from q=0 (92.68–97.56%)
            while q>=4 remains 2.35–6.57%; standard is the B4 screen winner.

DATE:       2026-08-04
CARD:       taskb_quotient_balanced_broad
CHANGE:     Replace B4's naturally q=0-heavy broad-u sampling with equal-size
            q buckets {0, 1, 2-3, >=4}; keep fixed N=1349, disjoint splits,
            the baseline architecture, optimizer, seed, and 2,000 updates.
PREDICT:    If B4's apparent gain is mostly q=0 composition, held-out exact
            will fall sharply and q>=4 will remain below 10%. A materially
            higher q>=4 score would instead show that broad operand support,
            not a missing long-reduction mechanism, is the main bottleneck.
RESULT:     confirmed at seed 0 — held-out exact is 47.27%, but q>=4 is 0.00%
            (q=0/1/2-3 = 87.50/59.38/42.19%). B4's aggregate was composition-
            confounded; balanced exposure does not yet yield long reduction.

DATE:       2026-08-04
CARD:       taskb_quotient_balanced_replication
CHANGE:     Repeat the quotient-balanced broad diagnostic at data/training
            seeds 1 and 2; architecture, N, optimizer, split sizes, and
            2,000-update horizon are unchanged from seed 0.
PREDICT:    Both replications retain q>=4 exact below 10%, while aggregate
            held-out remains materially above B4 because q=0 through q=3 are
            deliberately represented. A q>=4 recovery in either seed would
            make the apparent long-reduction gate seed-sensitive and block
            promotion to a new mechanism.
RESULT:     confirmed — seeds 0/1/2 score 47.27/48.05/51.56% overall held-out
            exact, but q>=4 is exactly 0.00% in all three. q=0 is
            87.50/89.06/92.19%, q=1 59.38/57.81/67.19%, and q=2-3
            42.19/45.31/46.88%. The current baseline has a reproducible
            long-reduction gate, not a seed-sensitive aggregate artifact.

DATE:       2026-08-04
CARD:       taskb_learned_reduction_cell_reimplementation
CLASS:      NEW REIMPLEMENTATION — NOT A REPRODUCTION
CHANGE:     Replace only the current standard Transformer with a diagnostic
            interface port of the historically named learned reduction cell:
            learned digit embeddings, an 8-step GRU digit sweep, learned
            attention over N digits, soft learned quotient state, learned GRU
            update, and learned output heads. Data, N=1349, seed, optimizer,
            batch size, and 2,000-update horizon remain B5-identical.
PREDICT:    If the digit-serial state supplies a usable long-reduction bias,
            q>=4 held-out exact becomes nonzero and exceeds the established
            0.00% B5 baseline. If it remains zero despite train fit, eight
            input-digit transitions are not sufficient evidence of an
            effective quotient/reduction iteration at N=1349.
RESULT:     refuted — train exact reaches 65.87%, but held-out exact is 3.12%
            and q=4-9, q=10-99, and q>=4 are each 0.00%. It is below every
            B5 baseline bucket and fails the nonzero-q>=4 promotion gate.

DATE:       2026-08-04
CARD:       taskb_teacher_depth_iterative_reducer
CLASS:      NEW DIAGNOSTIC — NOT SUBMISSION-RELEVANT
CHANGE:     Replace one-shot reduction supervision with generated intermediate
            trace supervision for one tied, fully learned `(state,N)->next
            state` cell. The evaluator supplies a known unroll count q only
            for Phase-1 diagnosis; the model forward contains no arithmetic
            subtraction, quotient, comparison, lookup table, or halting rule.
PREDICT:    If fixed effective computation is the blocker and the learned
            primitive can be exact, terminal exact stays high at q=5 and is
            nonzero at q=10 or beyond. If error compounds, terminal exact
            collapses rapidly after q=3 despite high one-transition accuracy,
            falsifying this primitive-before-halting formulation.
RESULT:     refuted at q>=10 — terminal exact is 100.00% at q=0/1/2,
            72.27% at q=5, and 0.00% at q=10/50/100 after 2,000 updates.
            Teacher-forced loss reaches 0.00013, so a learned single step can
            fit while greedy self-fed state errors still compound to collapse.
CORRECTION: The loss covers only anchor depths {0,1,2,5,10,50,100}, not every
            intermediate trace state. The later drift audit finds teacher-
            forced q=10 accuracy itself falls to 0% on unseen true states.
            Therefore q>=10 rollout stability is **unclear**, not a clean
            state-drift falsification; do not use this card as proof that
            error accumulation alone caused the collapse.

DATE:       2026-08-04
CARD:       taskb_teacher_depth_rollout_drift_audit
CHANGE:     Evaluation-only: retain the completed teacher-depth reducer and
            its held-out remainders, but report true-state versus self-fed
            state fidelity at every unroll step for initial q=5/10/50/100.
PREDICT:    Teacher-forced one-step fidelity remains near-perfect, whereas
            self-fed exact state falls before terminal failure; q=10 should
            show the first stable divergence. If both curves remain aligned
            until the final step, terminal collapse has another cause.
RESULT:     refuted as an isolation test — teacher-forced fidelity is not
            near-perfect at all q: for q=10 it falls from 100% at step 1 to
            58.20% at step 3 and 0% at step 4, before free rollout reaches
            0%. The anchor-depth training distribution omitted these true
            intermediate states.

DATE:       2026-08-04
CARD:       taskb_teacher_depth_full_trace
CLASS:      NEW DIAGNOSTIC — NOT SUBMISSION-RELEVANT
CHANGE:     Replace anchor-depth-only trace rows with equal learned-transition
            supervision for every true quotient depth q=0..100; model,
            optimizer, seed, batch size, 2,000 updates, held-out remainders,
            and evaluation audit are unchanged.
PREDICT:    Teacher-forced exact remains high across q=5/10/50/100. If
            self-fed exact then still separates sharply from teacher-forced,
            state drift is isolated. If the curves remain aligned, missing
            intermediate-state coverage—not self-feeding—is the primary
            explanation for the anchor-depth collapse.
RESULT:     confirmed — free terminal exact is q=5 99.22%, q=10 99.22%,
            q=50 97.27%, and q=100 95.70%; corresponding final teacher-forced
            exact is 99.61% at every depth. Full intermediate-state coverage,
            not a new state representation, removes the former q>=10 collapse.

DATE:       2026-08-04
CARD:       taskb_learned_canonicality_halting
CLASS:      NEW DIAGNOSTIC — NOT SUBMISSION-RELEVANT PENDING RULE REVIEW
CHANGE:     Add one learned binary stop head to the stable full-trace tied
            reducer. It is supervised as stop=1 only for canonical trace
            states (q=0), and evaluator-supplied q is removed at inference;
            the reducer, state representation, trace data, optimizer, seed,
            and 2,000-update horizon are otherwise retained.
PREDICT:    If stable reduction is the solved primitive, learned stopping will
            terminate near the true depth and retain high held-out exact at
            q=4-9 and q=10-99. If it stops early or never stops while q-known
            rollout succeeds, canonicality detection—not reduction—is the
            remaining gap.
RESULT:     confirmed within the trained q=0..100 range — independent held-out
            remainders score 100.00% exact and halting accuracy in q=0, q=1,
            q=2-3, q=4-9, q=10-99, and q=100. There are no early/late stops,
            non-stops, or wrong-remainder stops. This does not test q>100 or
            establish competition legality for learned control flow.

DATE:       2026-08-04
CARD:       taskb_reducer_quotient_extrapolation_101_500
CHANGE:     Evaluation-only quotient-depth extrapolation of the fixed learned
            reducer and stop head trained at q=0..100. Test independent
            remainders at every unseen q=101..500; no architecture, weight,
            trace data, or optimizer changes.
PREDICT:    If the tied cell learned the reusable one-N transition and
            canonicality rule rather than a finite horizon, teacher-forced
            local fidelity and free autonomous remainder/halting accuracy stay
            high through q=500. If only the trained horizon was learned,
            free rollout will degrade with q, with early/late/non-stop failure
            categories locating the break.
RESULT:     refuted — held-out q=101-500 averages 11.82% teacher one-step,
            11.81% remainder exact, and 11.81% halting accuracy, with 88.19%
            early stops. It is perfect through q=147, falls to 25.78% at
            q=148, and is 0% from q=149 through q=500. Local unseen-state
            failure and stopping fail together; this is not rollout-only decay.

DATE:       2026-08-04
CARD:       taskb_reducer_only_quotient_extrapolation
CHANGE:     Evaluation-only removal of the learned stop head from the q=0..100
            checkpoint's inference path: externally supply q and apply the
            unchanged reducer exactly q times on independent q=101..500
            states. No weights, architecture, trace data, or optimizer change.
PREDICT:    If the q=148 cliff is primarily canonicality detection, reducer-only
            terminal remainder exact will extend materially past q=148. If it
            shares the teacher-one-step cliff, the learned transition—not
            halting—is the first failed component.
RESULT:     refuted — q-known terminal exact exactly matches learned-halting
            autonomous exact: 11.81% aggregate. Boundary map: q=101,110,120,
            130,140,145,146,147 are 100%; q=148 is 25.78%; q=149 terminal is
            0% (2.73% one-step); q=150 is 0%. The transition is the failure.

DATE:       2026-08-04
CARD:       taskb_reducer_curriculum_200_to_500
CHANGE:     Extend only full-trace training support from q=0..100 to q=0..200
            for the identical tied reducer and learned stop head; optimizer,
            model, seed, independent remainders, batch size, and 2,000 updates
            are unchanged. Test every unseen q=201..500.
PREDICT:    If the q=148 cliff is state-support coverage, the exact transition
            and stop boundary will move beyond q=200, with high q=201..~250
            accuracy. If it remains near q=148, the architecture has a finite
            horizon unrelated to the trained quotient range.
RESULT:     confirmed — q=201..221 are 100%; q=222 is 37.89%; q=223..500 are
            0%. The extrapolation boundary moves from 148 to 222 when training
            support moves from q<=100 to q<=200, favoring state coverage over
            a fixed architectural horizon.

DATE:       2026-08-04
CARD:       taskb_reducer_curriculum_500_to_1000
CHANGE:     Extend only full-trace training support from q=0..200 to q=0..500
            for the identical tied reducer and learned stop head; all other
            settings remain unchanged. Test every unseen q=501..1000.
PREDICT:    If the boundary is controlled by state support, it will move beyond
            q=500. The amount of extrapolation beyond 500 measures whether
            curriculum coverage alone is becoming sufficient for scalable q.
RESULT:     partially supported but not promotable — autonomous/q-known exact
            is 98.05% at q=500 and remains 98.05% through q=518, then declines
            (90.62% at q=519; 80.47% at q=650) and reaches 0% at q=678. The
            q=501..1000 aggregate is 29.79% exact and 30.48% halting. Because
            the endpoint of the training range is already imperfect, this card
            cannot isolate extrapolation from accumulated in-range error.

DATE:       2026-08-04
CARD:       taskb_reducer_curriculum_500_exposure_matched
CHANGE:     Hold the q=0..500 full-trace distribution, tied reducer, learned
            stop head, optimizer, batch size, seed, and evaluation fixed; raise
            training updates from 2,000 to 6,200. This matches the q=0..200
            card's approximately 6.3 passes over its transition-row dataset.
PREDICT:    If the 98.05% q=500 result is insufficient exposure rather than an
            intrinsic q=500 limit, q=500 terminal exact will reach the prior
            card's 100% floor and the first q>500 failure will move materially
            beyond 518. If it remains imperfect, training support alone is not
            sufficient at this scale.
RESULT:     confirmed — q=500 is independently 100% exact/halting, and unseen
            q=501..666 remain 100% for teacher one-step, q-known terminal,
            autonomous terminal, and halting. q=667 falls to 69.92% terminal;
            q=741 is 16.80%; q=742..1000 is 0%. Held-out q=501..1000 averages
            43.12% terminal exact and 43.65% halting. This demonstrates 166
            exact quotient steps beyond the trained range with a fixed reducer.

DATE:       2026-08-04
CARD:       taskb_reducer_unseen_modulus_screen
CHANGE:     Hold the tied reducer architecture, optimizer, batch size, trace
            support q=0..500, update exposure, and evaluation protocol fixed;
            replace fixed N=1349 training with N={1081,1349,1763} and
            evaluate N={1189,1517};
            evaluate full traces on held-out N values. No inference algorithm or
            model capacity change.
PREDICT:    If the transition learned modular reduction rather than a 1349-only
            state map, held-out moduli will retain strong in-range q terminal
            exact and halting. If it memorized fixed-N geometry, performance
            will collapse despite depth support.
RESULT:     refuted — at q=1, held-out N=1189 and N=1517 each have 0% exact
            next-state/remainder accuracy (though token accuracy is 66.06% and
            56.74%). Deeper unseen-N rollout is therefore not informative. The
            fixed architecture learns a shared seen-N mapping, not a reusable
            modulus-general reduction primitive.

DATE:       2026-08-04
CARD:       taskb_reducer_depth_frontier_300
CHANGE:     Hold fixed N=1349, the tied reducer, learned stop head, optimizer,
            batch size, seed, full-trace data construction, and per-transition
            exposure fixed; change the maximum training quotient from 500 to
            300, using 3,000 updates (about 6.4 passes over its trace rows).
            Evaluate every held-out q=301..550.
PREDICT:    Pre-registered two-point interpolation only, not a scaling law:
            based on zero-exact frontiers 149 after q<=100 and 742 after
            q<=500, complete collapse will occur near q=447 (1.49×300).
            Report the last perfect q, first degradation q, first zero q,
            teacher one-step exact, autonomous remainder exact, halting errors,
            and the per-q boundary curve without changing this prediction.
RESULT:     invalidated as a frontier measurement — before held-out-q evaluation,
            independent in-range q=300 is only 85.55% autonomous remainder
            exact (86.33% q-known) despite 100% teacher one-step exact. This
            checkpoint has accumulated in-range rollout error, so q=301..550
            would not isolate a depth-extrapolation boundary.

DATE:       2026-08-04
CARD:       taskb_reducer_depth_frontier_300_exposure_gate
CHANGE:     Hold the q=0..300 data, fixed N, tied reducer, learned stop head,
            seed, batch size, optimizer, and all architecture settings fixed;
            raise updates 3,000→3,800 to match the successful q<=500 card's
            approximately eight passes over its trace rows. First evaluate q=300.
PREDICT:    This is an optimization gate, not a changed frontier prediction:
            q=300 autonomous exact should reach 100%. Only then rerun the
            frozen q=301..550 test with its original q≈447 collapse prediction.
RESULT:     gate confirmed — independent q=300 is 100% exact/halting. The
            frozen extrapolation prediction is refuted: q=301..302 are 100%,
            q=303 falls to 94.53% teacher/q-known/autonomous exact, and
            q=371..550 is 0%. Teacher and autonomous failure begin together;
            complete collapse is 371, not the predicted ≈447.

DATE:       2026-08-04
CARD:       taskb_unseen_n_q1_serial_subtractor
CHANGE:     Replace the parallel decimal-state reducer with a learned
            LSD-to-MSD GRU over aligned (u-digit, N-digit) pairs. Train q=1 on
            48 generated four-digit semiprimes and evaluate 16 unseen moduli;
            all forward computation remains learned categorical digit mapping.
PREDICT:    If digit significance plus a learned serial state makes subtraction
            compositional, held-out-N q=1 exact will be nonzero and materially
            exceed the prior 0%. If it remains zero, this representation still
            does not identify cross-modulus subtraction.
RESULT:     confirmed — 100% exact on all 2,048 q=1 examples from 16 unseen
            four-digit semiprimes (128 independent remainders each), with 100%
            token accuracy at every LSD-to-MSD digit position.

DATE:       2026-08-04
CARD:       taskb_unseen_n_serial_subtractor_rollout
CHANGE:     Evaluation-only: repeatedly apply the unchanged q=1 learned serial
            subtractor to q=1..5 states for the same unseen moduli and held-out
            remainders. No weights, data, architecture, or arithmetic changes.
PREDICT:    If the learned digit-state transition is exact and reusable, exact
            rollout remains 100% through q=5. Any gap between teacher one-step
            and rollout exact identifies accumulated transition error.
RESULT:     partially confirmed — rollout exact is 100%, 94.53%, 88.67%,
            83.35%, and 77.15% for q=1..5. Teacher one-step drops much faster
            on raw q>1 states (89.31%, 57.42%, 41.94%, 33.74%), showing the
            learned q=1 state transition composes better than its unsupported
            high-magnitude input distribution.

DATE:       2026-08-04
CARD:       taskb_unseen_n_serial_subtractor_q5_support
CHANGE:     Extend only serial-subtractor training-state support from q=1 to
            q=1..5 for the same generated seen/unseen modulus split, model,
            optimizer, and digit order; evaluate unseen-N q=1..10.
PREDICT:    If the remaining loss is high-magnitude state support, teacher and
            rollout exact will become near-perfect through q=5 and retain
            substantial exactness beyond q=5. If q=1-only learning was special,
            performance will fall despite the added trace support.
RESULT:     confirmed in-range; promising but width-confounded extrapolation.
            Seed 0 is 100% rollout q=1..10. Seed 1 is 100% q=1..6 and
            96.58%, 95.17%, 93.90%, 92.72% at q=7..10. Seed 2 is 100% q=1..4
            and 99.76% q=5, but q=10 includes six-digit qN+r states which the
            fixed five-digit representation cannot encode.

DATE:       2026-08-04
CARD:       taskb_unseen_n_serial_subtractor_width6_audit
CHANGE:     Change only decimal state width 5→6; keep the LSD-to-MSD learned
            GRU, embeddings, q=1..5 support, seen/unseen-N split, remainder
            sampling, optimizer, 4,000 updates, batch size, seeds, and q=1..10
            evaluation fixed. Use leading-zero padding with LSD-relative digit
            alignment; reject neither six-digit states nor truncated examples.
PREDICT:    q=1..5 remains near-perfect across seeds. q=6..10 remains strongly
            above the q=1-only baseline without width failures; any residual
            degradation is gradual arithmetic error, not a representation cliff.
RESULT:     confirmed for the requested seed-0 screen — all 2,048 unseen-N
            examples are representable at every q=1..10, with zero truncation
            failures. Autonomous rollout is 100% at every q=1..10; teacher
            one-step is 100% q=1..7 and 95.26%, 91.65%, 87.50% q=8..10.

DATE:       2026-08-04
CARD:       taskb_unseen_n_serial_subtractor_frozen_stop_head
CHANGE:     Freeze the qualified width-six serial subtractor checkpoint and
            train only a learned linear canonicality readout from its final
            serial GRU state plus N. Train on balanced true q=0..5 states from
            the 48 seen moduli; run no-q, no-depth autonomous halting on the
            16 held-out moduli at q=0..10, capped at sixteen reductions.
PREDICT:    The frozen serial state already supports exact learned subtraction,
            so a canonicality readout should keep q=0..5 near-perfect with
            negligible early stops, and preserve strong q=6..10 autonomous
            accuracy. If it fails while the fixed-q rollout passes, the likely
            cause is a stop-head distribution shift rather than a need to
            redesign the subtractor.
RESULT:     confirmed in the requested seed-0 screen. On 16 unseen moduli ×
            128 remainders per q, q=0..7 has 100% final-remainder and exact
            stop-step accuracy. q=8/9/10 is 95.26%/91.65%/87.50%, exactly the
            frozen subtractor's raw teacher-transition curve; there are no
            late stops, non-stops, representation failures, false-positive
            stops on noncanonical generated states, or false-negative
            continues on canonical generated states. The residual early
            terminal states arise only after an existing subtractor arithmetic
            error.

DATE:       2026-08-04
CARD:       taskb_unseen_n_serial_subtractor_q10_support
CHANGE:     Extend only learned serial-subtractor transition-state support from
            balanced q=1..5 to balanced q=1..10 traces, retaining width six,
            the LSD-to-MSD GRU, seen/unseen-modulus split, optimizer, batch,
            update count, and autonomous inference. Evaluate the new frozen
            subtractor q=1..20, then attach the existing frozen stop head
            without retraining it.
PREDICT:    Direct q=6..10 support moves the first subtractor degradation past
            q=10: q=1..10 should be near-perfect and q=11..20 should decline
            gradually rather than at the former q=8 boundary. The existing
            canonicality head should remain correct on generated states; if
            complete-system errors track subtractor errors, no head retraining
            is warranted. Failure in the directly supported q=8..10 range
            would instead identify insufficient per-q optimization exposure.
RESULT:     partly confirmed — q=1..20 fixed-depth terminal rollout is 100%
            (40,960 representable unseen-N examples, zero width errors), and
            one-step degradation moves to q=15 (96.88%; first <95% is q=16).
            The prediction that the old stop head transfers is refuted: its
            q=0 accuracy is 0% because a separately trained GRU has arbitrary
            latent coordinates, so it is not a valid canonicality readout.

DATE:       2026-08-04
CARD:       taskb_unseen_n_serial_subtractor_q10_rebound_stop_head
CHANGE:     After the q=1..10-trained serial GRU replaces the prior frozen
            subtractor, train only a fresh 129-parameter canonicality readout
            against that newly frozen GRU. Keep the width-six representation,
            q=0..5 balanced stop data, optimizer, and autonomous loop fixed.
PREDICT:    The prior stop head will not transfer across an independently
            retrained GRU's arbitrary latent coordinates, even though its
            architecture is identical. A fresh readout should restore exact
            canonicality through the 16-step cap; q=17..20 should correctly
            continue and register as cap-induced non-stops rather than false
            canonicality predictions.
RESULT:     partly confirmed — a fresh frozen-GRU readout yields 100% autonomous
            remainder and exact stop steps q=0..13. q=14 is 96.88% from 3.12%
            early false-positive stops; q=15 and q=16 are 93.75%, including
            64 and 128 incorrect-but-canonical generated states respectively.
            q=17..20 are cap-limited (83.94–89.50% non-stops) and also contain
            small accumulated early-stop rates. Generated-state false-positive
            rate is 0.0826%; false-negative canonicality rate is 0%.

DATE:       2026-08-04
CARD:       taskb_serial_subtractor_stability_gated_halting
CHANGE:     With the q=1..10 serial subtractor and its rebound stop head frozen,
            replace learned-canonicality-only termination with learned-canonical
            AND exact equality between the current digit state and one further
            learned subtractor candidate; raise the diagnostic cap 16→24.
PREDICT:    If true remainders are learned fixed points while wrong canonical
            states remain dynamic, q=14..16 errors will be repaired and q=0..20
            will halt exactly. Refutation: true remainders change under one more
            learned step, or wrong canonical states are absorbing, so the gate
            cannot distinguish them without changing training.
RESULT:     refuted — F(r,N)!=r on all 2,048 unseen remainders in every q bin,
            so q=0..20 all non-stop at the 24-step cap. No wrong-canonical
            state is directly repaired or stable: each becomes another wrong
            state. Stability-gated halting therefore cannot repair this model.

DATE:       2026-08-04
CARD:       taskb_serial_subtractor_absorbing_recovery_support
CHANGE:     Keep q=1..10 transition support and the width-six serial GRU fixed;
            add canonical identity rows r→r plus deduplicated wrong-canonical
            states generated by the frozen q=1..10 checkpoint, both supervised
            to the true remainder r. Keep optimizer, batch size, and 4,000
            update horizon fixed.
PREDICT:    If missing absorbing/recovery support caused the stability failure,
            F(r,N)==r should become near-perfect on held-out N and generated
            wrong-canonical states should map toward r rather than drift. If
            fixed-depth q rollout falls despite these labels, data-family
            interference—not lack of the invariant—is the immediate issue.
RESULT:     refuted — unseen-N true remainders remain 0% absorbing (all 2,048
            change under F for every q), and stability-gated q=0..20 therefore
            has 100% non-stops. Fixed-depth rollout is now only 100% q=1..2,
            then 99.66% q=3, 94.43% q=5, 74.56% q=10, and 66.26% q=20: the
            identity/recovery data families interfere with the original
            q-transition dynamics rather than teaching an unseen-N invariant.

DATE:       2026-08-04
CARD:       taskb_serial_subtractor_piecewise_q0_q20
CHANGE:     Replace q=1..10-plus-recovery training data with an exactly balanced
            piecewise transition set: q=0 identity r→r and q=1..20 transitions
            qN+r→(q-1)N+r, one equal-sized remainder set per q; retain width,
            serial GRU, optimizer, batch size, and 4,000-update horizon.
PREDICT:    If the GRU can learn the intended piecewise transition, unseen-N
            q=0 fixed-point accuracy and q=1/5/10/20 one-step accuracy become
            near-perfect together, and q=0..100 prescribed-depth rollout is
            stable. Refutation: q=0 identity generalizes poorly or harms q>0
            transitions even without recovery data, implying a representation
            or function-class conflict rather than an optimization issue.
RESULT:     refuted — q=0 fixed-point transition exact is 0% on both 6,144 seen
            and 2,048 unseen remainders, while unseen q=1/5/10 transition exact
            is 100% and q=20 is 96.44%. Prescribed-depth rollout stays 100%
            through q=13, then 92.58% q=20 and 89.84% q=100. The identity mode
            was not fitted at this balanced fixed budget, so this is direct
            data-family optimization/interference, not evidence of an unseen-N
            representation failure; do not evaluate or retrain halting.

DATE:       2026-08-04
CARD:       taskb_serial_comparator_controlled_reducer
CHANGE:     Replace monolithic q=0/q>0 transition selection with a learned
            serial comparator whose probability gates learned subtractor logits
            versus an input identity residual. First train/evaluate the
            comparator alone; then initialize from it and the q=1..10 serial
            subtractor, jointly train the gated composition on q=0..20 traces.
PREDICT:    If the missing operation is comparison, the comparator has
            near-perfect unseen-N and N−1/N/N+1 accuracy, and the gated reducer
            gains q=0 fixed points without losing q>0 transitions; autonomous
            q=0..100 then halts at the true q. Refutation: comparator boundary
            generalization fails, or it succeeds but fixed points still fail,
            showing the missing capability lies beyond comparison.
RESULT:     confirmed for the branch condition — Stage 1 unseen-N comparison is
            99.93% and boundary accuracy is 100%. Stage 2 has 100% q=0 fixed
            points and q=0..28 autonomous remainder/exact-halt accuracy; first
            degradation is q=29 (94.04%), reaching 37.50% at q=100 through
            early stops after subtractor error. Comparison solves canonical
            identity but not the remaining high-q transition frontier.

DATE:       2026-08-04
CARD:       taskb_comparator_reducer_transition_vs_rollout
CHANGE:     Freeze the qualified comparator-controlled reducer and measure the
            same unseen-N inputs two ways: one correct qN+r state at a time,
            then autonomous self-fed execution. No weights, representation,
            loop, or architecture changes.
PREDICT:    Comparator accuracy remains near-perfect at every selected q. The
            teacher transition should first soften around q≈29 but remain above
            final rollout at q=50/100; a growing teacher→rollout gap would show
            recurrence accumulation on top of primitive extrapolation error.
RESULT:     confirmed — comparator accuracy is 100% at every tested q. The
            frozen learned transition first degrades at q=30 (93.75%), exactly
            matching rollout there; q=50 is 85.06% teacher-forced versus
            62.50% rollout, and q=100 is 86.04% versus 37.50%. Thus unsupported
            high-q states first defeat the primitive, with recurrence adding
            further loss only after that first transition error.

DATE:       2026-08-04
CARD:       taskb_comparator_reducer_q100_intermediate_curriculum
CHANGE:     Keep the comparator-controlled width-six serial reducer, digit
            scan, optimizer, batch size, and 4,000-update horizon fixed. Resume
            the qualified q=0..20 reducer and extend its balanced transition
            traces (including q=0 identity) to q=0..100; no architecture or
            inference-loop change.
PREDICT:    Because teacher-forced failure begins only once q=30 states leave
            support, direct q=0..100 support should restore strong in-range
            unseen-N teacher transitions and autonomous q=0..100 reduction.
            If q=30+ remains weak despite this support, the shared primitive's
            optimization/exposure capacity—not merely missing q range—is the
            limiting factor.
RESULT:     confirmed — all 206,848 held-out transition cases q=0..100 are
            exact, as are every learned continue/stop label, true q=0 fixed
            point, and autonomous final-remainder/exact-halt outcome. A separate
            frozen audit at q=1,5,10,20,30,50,100 confirms 100% comparator,
            raw subtractor, composed transition, and rollout exactness (2,048
            examples per q). Direct intermediate-state support removes the
            former q=30 primitive frontier in this controlled range.

DATE:       2026-08-04
CARD:       taskb_comparator_reducer_q100_horizon_probe
CHANGE:     Freeze the q=0..100 curriculum checkpoint and evaluate only unseen,
            representable q=101,110,120,130,140 inputs. No training, model,
            representation, or loop change.
PREDICT:    If q=0..100 trace support teaches a reusable primitive rather than
            merely its observed state range, both one-step transition and
            autonomous rollout remain near-perfect through q=140. A teacher
            collapse first would instead show a new support frontier; a
            teacher-to-rollout gap would show recurrence accumulation.
RESULT:     confirmed in the representable extrapolation band — q=101,110,120,
            130,140 are all 100% comparator, raw subtractor, composed
            transition, and autonomous rollout exact (2,048 examples/bucket).
            The initial q≥120 0% read was an audit-cap bug (110 steps), fixed
            before interpreting the experiment; the rerun used q_max+10 steps.

DATE:       2026-08-04
CARD:       taskb_serial_reducer_width14_control
CHANGE:     Change only decimal state width from six to fourteen digits. Re-run
            the qualified seed-0 serial pipeline: q=1..10 subtractor, boundary-
            balanced comparator, q=0..20 gated reducer, then q=0..100 support.
            Keep LSD-first order, GRU components, split, seed, optimizer,
            batch size, and each stage's update horizon unchanged.
PREDICT:    Leading-zero padding and learned place embeddings preserve unseen-N
            comparator boundaries, q=0 fixed points, and q=0..100 transitions.
            Refutation: width alone harms these controlled four-digit results,
            which would make serial positional scaling unsafe before public-
            scale N tests.
RESULT:     near-confirmed, not exact — W=14 preserves 100% held-out q=1
            subtraction, 100% boundary comparison, and 100% composed q=1..100
            teacher transitions, but q=0 fixed points and every autonomous
            bucket are 99.951% (one of 2,048 canonical states false-continues).
            Width padding is not catastrophic, yet this seed does not equal the
            exact W=6 control and is not a promoted submission component.

DATE:       2026-08-04
CARD:       taskb_serial_chunk_reducer_0248
CHANGE:     Replace the unit-transition target with a learned five-way chunk
            action k∈{0,1,2,4,8} plus learned next-state digits. Keep W=14,
            LSD-first shared GRU digit encoding, seed-0 modulus split, q=0
            identity rows, optimizer, batch size, and 4,000 updates. The
            forward predicts both action and digits; it contains no explicit
            comparison, subtraction, or multiplication.
PREDICT:    A learned chunk controller preserves q=0/q=1 and reaches q=100
            exactly in 13 or fewer updates, the minimum possible with max k=8.
            It should make q=1000 materially shorter than a unit loop, though
            raw q=1000 transition extrapolation may fail. Refutation: q=0/q=1
            transfer breaks, or q=100 cannot reach the true remainder within
            13 updates despite direct q≤100 transition support.
RESULT:     refuted — the five-way action head partially fits (84.91% q=0,
            93.31% q=1, 100% q≥20) but the simultaneously learned chunked
            digit transition is 0% q=0/q=1 and only 51.86% q=100. Autonomous
            remainder exact is 0.146% q=100 (43.45 average steps versus 13
            target) and 0.293% q=1000. This fresh multi-chunk formulation does
            not preserve the established unit subtraction primitive.

DATE:       2026-08-04
CARD:       taskb_chunk_decoder_from_unit_init
CHANGE:     Initialize only the chunk reducer's serial digit decoder from the
            qualified W=14 unit-reducer checkpoint; retain a random five-way
            k∈{0,1,2,4,8} action head. Keep chunk data, loss, optimizer, seed,
            W=14 encoder, batch size, and 4,000 updates identical to the fresh
            chunk card.
PREDICT:    Preserving the unit digit law restores q=0/q=1 exact and gives the
            chunk decoder enough local structure to reach q=100 in its 13-step
            k≤8 minimum. Refutation: initialized q=0/q=1 or q=100 stays near
            the fresh-card failure.
RESULT:     refuted — initialization raises q=0 action accuracy to 99.32% but
            chunk digit next-state exact remains 0% q=0/q=1 and 39.16% q=100;
            terminal q=100 exact is 0%. The unit decoder initialization alone
            does not transfer to direct x−kN digit generation.

DATE:       2026-08-04
CARD:       taskb_frozen_unit_chunk_controller
CHANGE:     Freeze the complete qualified W=14 comparator-controlled unit
            reducer and train only a random five-way k∈{0,1,2,4,8} controller
            on its frozen serial state. At evaluation the predicted action
            schedules that many repeated learned unit transitions; no decimal
            arithmetic is implemented in the forward.
PREDICT:    A controller can learn q=0 stop and chunk actions while the frozen
            unit primitive keeps q=0/q=1 arithmetic exact; q=100 should finish
            in 13 outer actions, albeit still 100 inner learned-unit updates.
            Refutation: controller action accuracy or q=0/q=1 terminal exact
            fails, showing chunk control itself cannot bind to the learned
            state.
RESULT:     refuted as configured — frozen learned arithmetic remains 99.95%
            macro-transition exact q=0/q=1, but controller action accuracy is
            0% q=0..5 and 100% q≥8. It never chooses action 0, giving 100%
            non-stops in all buckets. The q-balanced traces are action-imbalanced
            (93/101 q values label k=8), so this isolates controller-class
            imbalance rather than loss of the frozen unit primitive.

DATE:       2026-08-04
CARD:       taskb_frozen_unit_chunk_controller_balanced_actions
CHANGE:     Change only frozen-controller batch sampling from q-balanced to
            five-way action-class-balanced. Keep the frozen W=14 unit reducer,
            controller architecture, q=0..100 rows, targets, loss, optimizer,
            seed, batch size, and 4,000 updates fixed.
PREDICT:    Equal action exposure restores k=0 and small-k prediction while
            frozen unit execution retains q=0/q=1 macro exactness; q=100 then
            reaches the remainder in 13 outer actions but still 100 inner unit
            updates. Refutation: k=0 action remains poor or q=100 fails despite
            correct action exposure.
RESULT:     refuted — balanced sampling makes q=0 action/macro/terminal exact
            100% and q=1 terminal 93.75%, but multi-chunk action remains weak
            (0% q=2, 41.85% q=100) and q=100 terminal exact is 0%. Thus class
            imbalance explained the missing stop action, not the failure to
            infer a useful chunk from frozen serial state.

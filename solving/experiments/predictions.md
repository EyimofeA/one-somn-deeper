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
CARD:       taskb_input_conditioned_workspace
CHANGE:     Fixed learned K=8 workspace initialization becomes one learned
            ordered-input cross-attention read using the existing transition
            attention; shuffled-context uses the same read from a deterministic
            non-identity batch derangement.
PREDICT:    Correct context will improve held-out-u exact match, especially
            q>=10, while shuffled context remains near the fixed-workspace
            result; otherwise the fixed initializer is not the primary
            bottleneck.

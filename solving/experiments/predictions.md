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

DATE:       2026-08-04
CARD:       taskb_frozen_unit_threshold_bank
CHANGE:     Replace the refuted five-way frozen-state chunk classifier with four
            independently learned bits encoding the safe greedy chunk
            k=min(q,15). Preserve the frozen W=14 comparator-controlled unit
            reducer, q<=100 seen-modulus traces, and unseen-modulus test split.
            Training balances the sixteen chunk-code targets. Literal
            independent "can subtract 1/2/4/8" predicates would select 15 at
            q=8 and overshoot, so this is a learned safe binary code instead.
PREDICT:    If quotient magnitude is recoverable as simple thresholds, bit and
            selected-k accuracy will exceed the five-way controller, retain
            q=0/q=1, and produce nonzero q=100 terminal exact in about seven
            outer actions. If selected-k remains poor, frozen final-state
            compression—not multiclass coupling—is the controller limit.
RESULT:     refuted for promotion — q=100 terminal exact rises from 0% to
            30.18% (4.22 mean outer actions), but q=1 terminal exact falls to
            77.25% and exact q=100 chunk selection is only 73.29%. Threshold
            coding helps high-q action selection but does not preserve local
            behavior, so it is not a viable controller replacement.

DATE:       2026-08-04
CARD:       taskb_frozen_unit_threshold_bank_per_position
CHANGE:     replace only the threshold bank's frozen controller feature from
            the final serial-GRU state to the concatenation of all fourteen
            per-position GRU states. Preserve frozen W=14 arithmetic, binary
            safe-chunk code, q<=100 traces, sixteen-code balancing, optimizer,
            batch size, update count, seed, and unseen-N evaluation.
PREDICT:    If q information is discarded by final-state compression, position
            features restore q=1 selected-code/remainder exact and improve
            q=100 terminal exact beyond 30.18%. If they do not, action
            selection is not limited by that compression and this controller
            family should be closed without added mechanisms.
RESULT:     confirmed — per-position features restore q=1 remainder exact to
            99.95% and q=100 selected-code/macro exact to 100%, producing
            99.51% q=100 terminal exact. The final-state readout had discarded
            action-relevant quotient information; the remaining issue is that
            scheduling k repeated unit transitions still costs O(q) inner work.

DATE:       2026-08-07
CARD:       research_structured_position_final_label
CHANGE:     Replace the pooled/register evolving state with a persistent
            LSD-aligned per-position latent state and shared local updates;
            retain final-label-only supervision and the small-N VDF harness.
PREDICT:    Structured state should beat the matched global/register controls
            at held-out-N T=1 and retain a smaller advantage at T=2 because
            positional arithmetic information is not compressed. Refute if
            held-out-N T=1 is not materially above both controls.

DATE:       2026-08-07
CARD:       research_structured_position_trace_supervision
CHANGE:     Keep the structured per-position latent architecture fixed and
            add generated intermediate VDF-state supervision.
PREDICT:    Trace supervision should improve local transition exactness and
            T=2/4/8 rollout over the structured final-label cell. Refute if
            it fits traces without a held-out-N rollout gain.

DATE:       2026-08-07
CARD:       research_t1_state_topology_tournament
CHANGE:     Restrict the matched state-topology comparison to T=1 examples;
            compare register, global latent, and structured LSD-position tape
            under the same split, optimizer, and wall-clock budget.
PREDICT:    Removing depth credit assignment should raise all models' T=1
            exactness, while structured state should retain the best unseen-N
            result if quotient-relevant position information is causal. Refute
            the topology branch if it does not beat both controls at T=1.

DATE:       2026-08-07
CARD:       research_t1_discrete_iterative_refinement
CHANGE:     Replace one-shot output decoding with a shared discrete masked-token
            refinement cell; train only on T=1 targets and evaluate K=1,2,4,8
            refinement steps.
PREDICT:    Refinement should improve exact match from K=1 to K<=4 if residual
            digit errors can be corrected conditionally; kill if extra K only
            adds latency or if unseen-N exact remains at the marginal baseline.
RESULT:     refuted for promotion — K=1 unseen-N exact was 0.47% and K=4/8
            rose only to 8.18%/8.88%, while the deterministic structured tape
            reached 17.06% at the same T=1 budget; refinement added compute
            without entering a qualitatively stronger regime.

DATE:       2026-08-08
CARD:       research_t1_representation_decimal_binary_limbs
CHANGE:     Keep the structured local recurrent tape and final-label T=1
            objective fixed while replacing decimal tokens with either 7
            little-endian binary bits or two little-endian 4-bit limbs.
PREDICT:    If representation is the main ceiling, binary or fixed-width limbs
            should make a qualitative held-out-x/unseen-N jump over decimal,
            not merely 1–3 points. The limb width is fixed at 4 bits with two
            limbs before the run and will not be tuned post hoc. Refute the
            representation hypothesis if both alternatives remain in the same
            low-generalization regime.

DATE:       2026-08-04
CARD:       easy_serial_recurrent
CHANGE:     Replace the unavailable/stale Easy anchor with a smallest legal
            end-to-end prompt model: LSD-relative field/place features, a
            shared bidirectional-attention plus right-to-left GRU transition,
            and a T-conditioned straight-through recurrent token state. It
            receives only evaluator `(N,x,T)` tokens and final output labels.
PREDICT:    On e1, learned decimal alignment plus weight tying will beat the
            historical direct baseline's test exact accuracy, with strongest
            signal at T=1; refutation is failure to validate, failure to train
            in the 60-second local envelope, or no test/OOD gain over the
            4.7%/9.0% historical split values.
RESULT:     refuted — source validation and smoke pass, but the tier-faithful
            local e1 run reaches only 1.33% test, 0% OOD, and 0.67% mean exact
            in 500 steps/60 seconds, below the historical 4.7%/9.0% e1 split
            results. The mechanism is legal but not locally competitive.

DATE:       2026-08-04
CARD:       taskb_action_conditioned_macro_transition
CHANGE:     Replace the refuted direct chunk decoder with a serial digit decoder
            explicitly conditioned on the frozen per-position controller's
            four-bit learned chunk code. Freeze the qualified W=14 unit reducer
            and controller; train only the action-conditioned decoder on the
            same q<=100, seen-modulus macro targets and evaluate unseen N.
PREDICT:    If previous direct chunk failure arose because digits never received
            the selected action, q=0/q=1 macro exact remains near the frozen
            primitive and q=100 direct macro-transition exact becomes nonzero
            without applying fifteen unit steps. Refutation is q=0/q=1 damage
            or near-zero q=100 direct exact despite the correct learned action.
RESULT:     refuted — controller selected-k is 100% at q=100, but learned
            one-call macro-transition exact is 0% (q=1 only 47.07%). Giving
            the decoder the action fixes neither exact multi-unit digits nor
            autonomous rollout; action selection is not the remaining barrier.

DATE:       2026-08-04
CARD:       recurrent_vdf_square_reduce_smalln
CHANGE:     Build a clean staged, tied VDF transition F(s,N) on complete
            two-digit semiprime families: an LSD-first learned raw-square
            module followed by a learned serial comparator/subtractor reducer.
            Train each learned primitive with synthetic intermediate-state
            labels, freeze neither phase-specific weights into recurrence, and
            apply the same composed F exactly T times at evaluation.
PREDICT:    If validated serial reduction can compose with learned squaring,
            unseen-N one-step square-mod exact will be nonzero and multi-step
            accuracy will track one-step quality rather than collapsing at T=2.
            Refutation is near-zero held-out-N one-step exact or a major gap
            between one-step and self-fed rollout despite correct primitive loss.
RESULT:     refuted as a full unseen-N VDF step, with a localized reduction
            failure — held-out-N Squareθ/raw-square exact is 100%, comparator
            is 99.97%, but subtractor teacher exact is 80.36% and full
            reduction after a correct square is 46.96% (q>=10: 29.92%). T=8
            rollout remains 33.88%, so the primary problem is unseen-N serial
            reduction, not square generation or catastrophic recurrent drift.

DATE:       2026-08-05
CARD:       recurrent_vdf_reducer_square_trace_support
CHANGE:     Replace only reducer/comparator training rows from uniform algebraic
            qN+r states to every intermediate state in true seen-modulus
            VDF square traces s²→s² mod N. Reuse the established square
            checkpoint unchanged; retain the serial architecture, optimizer,
            update horizon, modulus split, and all-residue held-out evaluation.
PREDICT:    If raw state-distribution mismatch caused held-out reduction error,
            q≥10 reduction and held-out VDF T=1/T=8 exact will rise materially
            above 29.92%/46.96%/33.88%. If they remain near those values, the
            serial reducer's unseen-modulus representation is the bottleneck.
RESULT:     confirmed — holding Squareθ fixed, held-out q≥10 reduction rises
            29.92%→94.67%, T=1 46.96%→95.56%, and T=8 33.88%→89.02%. The
            primary prior failure was reducer state-distribution mismatch, not
            square/reduce composition or an intrinsically inadequate reducer.

DATE:       2026-08-05
CARD:       fable_tcap_adamw_e1_baseline_control
CHANGE:     Evaluate the existing legal Fable T-cap/AdamW submission under the
            current logged Easy e1 evaluator; no source, model, optimizer, or
            manifest setting is changed.
PREDICT:    Its train-time random depth exposure should generalize better than
            Fable v2's 0.00% mean, but it will still fail the submission gate
            unless held-out mean exact exceeds the 1.00% hosted Easy reference.
            Refutation: zero or near-zero mean exact despite nontrivial train fit.
RESULT:     confirmed — unchanged source reaches 0.67% test and 8.00% OOD
            exact (4.33% mean) in the current e1 harness, above the 1.00%
            hosted Easy reference. It does not certify T=1, so it is promoted
            only for an Easy mean-exact attempt, not Medium or Hard.

DATE:       2026-08-05
CARD:       recurrent_vdf_square_trace_support_long_horizon
CHANGE:     Increase only comparator/subtractor optimization from 3,000 to
            12,000 updates on the already validated VDF-square-trace support;
            keep the frozen learned Square checkpoint, serial architecture,
            optimizer, data, split, batch size, and evaluation fixed.
PREDICT:    If the residual 4.44% held-out T=1 error is optimization-limited,
            T=1 exact rises above 95.56% and T=8 rises above 89.02%; if both
            remain near the original values, the residual is a representation
            or unseen-modulus generalization limit rather than insufficient
            update count.
RESULT:     unclear — the remote run exited before training because the frozen
            Square checkpoint was absent from that mirror; it produced no
            transition or rollout metric and must be relaunched only after
            artifact synchronization, outside the deadline window.

DATE:       2026-08-05
CARD:       fable_muon_adamw_e1_optimizer_control
CHANGE:     Replace only the existing T-cap/AdamW submission's optimizer with
            the already-written Muon+AdamW hybrid source; architecture, prompt
            features, depth cap, manifest, and current e1 evaluator are held
            fixed.
PREDICT:    Muon may improve transformation-matrix conditioning enough to beat
            the T-cap/AdamW 8.50% hosted mean; refutation is local e1 mean at
            or below 4.33%, which gives no margin over its pre-hosted baseline
            and does not justify another Easy attempt.
RESULT:     refuted — Muon reaches 100% train exact but only 1.33% test / 6.00%
            OOD (3.67% mean), below the AdamW control's 4.33% local mean. The
            optimizer accelerates memorization, not generalization.

DATE:       2026-08-05
CARD:       fable_tcap_adamw_batch512_throughput_control
CHANGE:     Increase only the selected Fable T-cap/AdamW source's training
            batch size from 256 to 512; leave its evaluation batch 1,024,
            architecture, loss, optimizer, schedule, recurrence, and e1
            manifest unchanged.
PREDICT:    Since the 256-batch control used under 1 GiB and low GPU
            utilization, batch 512 increases examples per second without
            reducing local held-out mean below 4.33%. Refutation: throughput
            does not improve or e1 mean falls at/below the 4.33% control.

DATE:       2026-08-05
CARD:       vdf_square_reduce_final_label_e1
CHANGE:     Replace the Fable register with a final-label-only VDF cell: a
            tied learned LSD-first SquareCell followed by a distinct learned
            LSD-first ReduceCell, with a straight-through learned digit
            register between T applications. It receives only prompt tokens
            and evaluator final labels; no diagnostic trace labels or
            precomputed arithmetic enter the submission.
PREDICT:    The explicit two-stage inductive bias will produce nonzero e1
            held-out exactness, with T=1 stronger than T=2/3. Refutation is a
            failure to train in the 60-second envelope or no held-out signal
            above the random/direct baseline.
RESULT:     unclear — it reaches 2.67% test exact after 64 updates, but the
            first implementation unconditionally ran all 64 cells even where
            Easy provides T≤3. This is a compute-execution confound, not a
            test of the VDF mechanism under a useful training horizon.

DATE:       2026-08-05
CARD:       vdf_square_reduce_dynamic_depth_execution
CHANGE:     Change only the VDF candidate's outer loop from an unconditional
            64 iterations with masked updates to exactly `max(T)` tied learned
            Square→Reduce iterations for the current batch. All cells, state,
            loss, optimizer, batch, and data remain fixed.
PREDICT:    Easy's T≤3 batches will execute about twenty times less forward
            work, yielding far more than 64 optimizer updates in 60 seconds
            and a materially stronger held-out curve. Refutation: update count
            does not rise substantially or held-out exact remains near 1.33%.
RESULT:     mixed — the update count rises 64→434 and test exact rises
            2.67%→3.33%, but OOD remains 0% (1.67% mean). The speed fix is
            confirmed; the architecture is not yet a competitive submission.

DATE:       2026-08-05
CARD:       vdf_square_reduce_muon_optimizer
CHANGE:     Replace only AdamW in the dynamic-depth final-label VDF candidate
            with a standard Muon-for-matrices plus AdamW-for-vectors hybrid;
            retain cells, state, T execution, batch, loss, and data.
PREDICT:    Muon will improve early fitting of the two serial arithmetic cells
            without reducing the 434-update horizon; promotion requires OOD
            exact above 0% or mean exact materially above 1.67%. Refutation:
            faster train fit with no held-out gain, as in the Fable control.
RESULT:     refuted for the all-matrix split — it reaches 4.00% test but 0%
            OOD (2.00% mean) while training reaches 99.8%, confirming an
            overfitting optimizer assignment rather than a generalization gain.

DATE:       2026-08-05
CARD:       vdf_square_reduce_muon_transform_matrices_only
CHANGE:     Restrict the VDF Muon+AdamW hybrid's Muon group from every matrix
            to only SquareCell/ReduceCell attention and MLP transformation
            matrices; leave all embeddings, GRU weights, register projection,
            head, norm, model, data, and schedule unchanged.
PREDICT:    Keeping representational tables and recurrent state on AdamW will
            reduce the previous Muon overfit and restore nonzero OOD exact.
            Refutation: OOD remains 0% or mean does not exceed 2.00%.
RESULT:     refuted — targeted Muon reaches 99.8% train exact but only 2.67%
            test / 0% OOD (1.33% mean), worse than broad Muon and dynamic-depth
            AdamW. Muon is not a viable optimizer for this VDF candidate.

DATE:       2026-08-05
CARD:       vdf_square_reduce_fused_valid_gru
CHANGE:     Replace only each SerialCell's padded Python GRUCell reverse loop
            with a fused nn.GRU over the reversed valid prefix. Keep VDF cells,
            dynamic T execution, AdamW, data, batch size, and output unchanged.
PREDICT:    Removing padding-contaminated state and thousands of small scan
            launches increases updates beyond 434/60 seconds while preserving
            or improving the 3.33% test exact baseline. Refutation: no speed
            gain or lower held-out exactness.
RESULT:     mixed — fused valid-GRU increases updates 434→490 (+13%) and
            fixes padding exposure, but test/OOD become 2.00%/0% (1.00% mean),
            below dynamic-depth GRUCell AdamW. Keep it as a speed control, not
            a promoted accuracy source.

DATE:       2026-08-05
CARD:       vdf_square_reduce_active_row_compaction
CHANGE:     In the fused-valid-GRU VDF cell, execute each tied outer step only
            for rows whose prompt T exceeds that depth; scatter their learned
            register and hidden states back. Keep cells, AdamW, data, batch,
            and output math fixed.
PREDICT:    Mixed Easy T=1/2/3 batches avoid one-third of their outer cell work,
            raising updates beyond 490/60 seconds without lowering held-out
            accuracy. Refutation: indexing overhead erases speed or exactness
            drops further.
RESULT:     mixed — it gives 4.00% test / 0% OOD (2.00% mean) and 717 MiB,
            but completes 463 updates: better accuracy than fused GRU alone,
            yet no reliable total-throughput gain over 490 updates. Retain as
            an optional mixed-depth execution path, not a speed promotion.

DATE:       2026-08-05
CARD:       vdf_square_reduce_register_only_intermediate_logits
CHANGE:     In the fused-GRU active-row VDF cell, compute head/softmax/STE only
            for learned register positions during recurrence; keep final full
            logits, cells, state, AdamW, data, and batch fixed.
PREDICT:    Avoiding vocabulary heads on non-register prompt tokens raises
            updates beyond 463/60 seconds without changing register semantics
            or lowering the 2.00% mean exact control. Refutation: gather/scatter
            overhead outweighs head savings or exactness regresses.
RESULT:     confirmed narrowly — 494 updates (+1% over active compaction) with
            3.33% test / 0% OOD (1.67% mean), preserving the dynamic AdamW
            accuracy reference. The output head is not a dominant runtime cost.

DATE:       2026-08-05
CARD:       vdf_square_reduce_tensorized_t_parse
CHANGE:     Replace only the per-token Python decimal T parsing loop with a
            place-weighted tensor sum that saturates safely at MAX_STEPS=64;
            keep fused scan, active compaction, register-only logits, cells,
            AdamW, data, and batch fixed.
PREDICT:    Because layout runs once per forward, speed gain will be small but
            positive; parsed T and held-out accuracy should remain unchanged.
            Refutation: no throughput gain or a T-dependent correctness change.
RESULT:     refuted — the tensorized parser completes 453 updates in 60.1 s,
            below the 494-update register-only control, and reaches only 1.33%
            test / 0% OOD (0.67% mean). The prior sequential parser is restored.

DATE:       2026-08-05
CARD:       vdf_square_reduce_integrated_medium_m1
CHANGE:     Integrate the three individually measured execution changes—fused
            valid-prefix GRU, active-row compaction, and register-only interim
            logits—into the final-label VDF model; retain its sequential T
            parser, AdamW, width, cells, loss, and data. Run the real m1
            Medium manifest for its evaluator-owned 600-second budget.
PREDICT:    The bundle will preserve basic fitting while increasing the number
            of Medium updates relative to the original GRUCell implementation;
            it will still fail the OOD-N ladder because no implementation-only
            change addresses the unresolved learned VDF transition. Refutation:
            compile/runtime failure, substantially lower training throughput,
            or a materially different depth profile.
RESULT:     partial / intentionally stopped — m1 reached 3,400 updates in
            407.4 seconds (about 8.35 steps/s), with loss 28.351→2.249 but
            batch exact only 0–0.39%. It was stopped before evaluator-owned
            final/depth evaluation, so it supplies optimization evidence only,
            not a Medium score or a depth-profile result.

DATE:       2026-08-05
CARD:       vdf_final_label_t_curriculum_e1
CHANGE:     Keep the final-label VDF architecture and exact per-input T
            execution fixed, but stage the legal token loss over existing e1
            rows: T=1 for the first third of training, T≤2 for the second,
            and T≤3 for the final third. No intermediate state labels.
PREDICT:    If final-label credit assignment is the primary blocker, direct
            T=1 final supervision will identify a useful tied transition and
            lift the seen-N depth ladder at T=1 before T=2/4. Refutation: no
            better T=1 rung than the 3.33% dynamic-depth AdamW reference, or
            no coherent improvement with depth.
RESULT:     refuted for promotion — 461 updates give the same 3.33% test / 0%
            OOD endpoint as the non-curriculum reference. The seen-N T=1 rung
            is 5.2632% (2/38) but every rung T>=2 is 0%; one-step exposure does
            not yield a composable learned transition.

DATE:       2026-08-05
CARD:       vdf_trace_supervision_ablation_e1
CHANGE:     Keep the final-label VDF model and public e1 prompt rows fixed in a
            diagnostic-only loop, but add equal-weight cross-entropy on each
            model register readout against its generated true intermediate VDF
            state. This is not a submission or legal competition objective.
PREDICT:    If architecture capacity rather than model form is the blocker,
            explicit transition targets will sharply improve final exactness at
            T=1/2/3 relative to final-label-only curriculum. Refutation: trace
            loss fits without material final T accuracy, implicating the state
            interface or recurrent cell instead of final-label credit assignment.
RESULT:     refuted for this architecture/budget — trace supervision raises
            step-500 train final exact to 24.22% (versus 19.7% at the final
            curriculum step) but held-out final exact is 0% at each T=1/2/3
            and OOD T=6. The state interface/cell still does not generalize.

DATE:       2026-08-05
CARD:       vdf_architecture_audit_direct_transformer_e1
CHANGE:     Run the upstream direct final-output Transformer baseline unchanged
            on the same public e1 manifest and evaluator reporting used for the
            tied VDF and tied-VDF-curriculum controls.
PREDICT:    It will fit prompt/output statistics but show no certified depth or
            OOD-N mechanism. Refutation: it exceeds both tied VDF controls at
            T=1 and retains a materially stronger higher-T ladder, showing the
            recurrence/state interface is not the primary issue.
RESULT:     confirmed as a non-mechanistic control — 697 updates yield 2.00%
            test / 1.00% OOD, zero certified depth, and a noisy ladder. It does
            not establish a reusable transition, but its OOD endpoint exceeds
            both tied VDF controls in this one-seed, 60-second comparison.

DATE:       2026-08-05
CARD:       vdf_final_label_true_depth_curriculum
CHANGE:     In a research-only fixed-N=323 final-label dataset, expose only
            existing T=1 examples for one third of a 180-second run, then
            T<=2, then T<=4. The tied transition, exact per-row execution, and
            final-label loss remain unchanged; no intermediate labels.
PREDICT:    Genuine T=1→2→4 exposure will give a monotone held-out-x curve at
            least through T=4, unlike the weak e1 T=1 bump. Refutation: held-
            out T=1 remains near chance or T=2/4 collapse despite phase-wise
            final-label fitting, showing final-only curriculum is insufficient.
RESULT:     refuted — the 180-second run reaches 100% training exact in T=1,
            T<=2, and near 100% T<=4 phases, but held-out-x exact is 0% at
            T=1/2/4 (1.54% at T=8, zero otherwise) and unseen-N is <=0.66%.
            Final-label curriculum fits examples without learning F_theta.

DATE:       2026-08-05
CARD:       competition_fable_tcap_adamw_easy_e1_revalidation
CHANGE:     No model change; re-run the strongest packaged full-task Fable T-cap + AdamW candidate on the current evaluator checkout and active L40, using the e1 Easy manifest.
PREDICT:    The run will fit substantially faster than the hosted H100 baseline but remain below the historical 8.50% mean due to hardware/seed variance; promotion requires mean exact above 8.50% and a nonzero certified T=1 rung locally.

RESULT:     refuted — 482 updates in 60.0s reached 1.333% test / 5.000% OOD / 3.167% mean, with no certified T=1 rung; the result missed the 8.50% gate.

DATE:       2026-08-05
CARD:       competition_fable_tcap_adamw_medium_m1_revalidation
CHANGE:     No model change; run the same legal Fable T-cap + AdamW candidate on the public Medium m1 manifest after Easy revalidation.
PREDICT:    The longer budget will lower training loss and raise train exactness, but held-out exactness and T=1 certification will remain weak; promotion requires beating the best known hosted Easy reference (8.50% mean) and nonzero local T=1 certification until a Medium-specific hosted reference is available.
RESULT:     refuted — 8,363 updates in 600.1s reached 0.067% test / 0.000% OOD / 0.050% mean, with no certified T=1 rung; the longer budget did not improve held-out generalization.

DATE:       2026-08-05
CARD:       research_clean_latent_workspace_smalln_final_label
CHANGE:     Replace the answer-aligned token/register evolving state with one global learned latent h; encode (N,x) once, apply the same learned transition exactly T times, then decode digits. Compare against a parameter-matched per-position register control under final-label-only supervision.
PREDICT:    The global latent will improve held-out-N T=1 and preserve more of the T=1 signal through T=2/4/8 because the transition state is not tied to output positions. Success requires held-out-N T=1 >=10% and T=8 >=5%, with latent beating the register control by >=5 percentage points at T=1. Kill if latent does not beat control at T=1 or training exact remains <50% after 120 seconds/model. Budget: <=90 minutes coding and 240 GPU seconds.
RESULT:     confirmed narrowly — global latent reaches 17.29% unseen-N T=1 vs 9.35% register control (+7.94 points), with 11.68% vs 13.55% at T=4 and 14.49% vs 14.02% at T=8; both fit seen moduli, so the state-interface hypothesis gains a small-N signal but not a competition-ready solution.

DATE:       2026-08-06
CARD:       deadline_existing_easy_fable_tcap_adamw_e1
CHANGE:     Resubmit the exact previously hosted 8.50%-mean Fable T-cap AdamW
            source on Easy e1; no architecture or optimizer change.
PREDICT:    It remains the highest-EV Easy attempt because it is the only
            currently valid source with an 8.50% hosted result on e1.
            Refutation: a materially lower score would establish that the
            result was not robust across evaluator runs.
RESULT:     confirmed for selection, with ordinary run variance — hosted e1
            `1b06d008-89e0-4a49-90d4-f2589a969ed6` scored 8.00%, close to the
            prior 8.50% and far above the available legal recurrent controls.

DATE:       2026-08-06
CARD:       deadline_existing_medium_fable_tcap_adamw_m5
CHANGE:     Resubmit the exact batch-256 Fable T-cap AdamW source that
            previously reached 0.25% hosted mean on Medium m5; no model change.
PREDICT:    It should outperform the recent batch-512 Fable m1 result (0.03%)
            because it is the strongest directly observed Medium configuration.
            Refutation: <=0.03% would remove the only material Medium evidence.
RESULT:     confirmed for selection — hosted m5
            `60510147-ed5b-4944-97b5-ddce0340b883` scored 0.17% (0.20% test,
            0.10% OOD), below its 0.25% historical run but above the 0.03%
            recent m1 control.

DATE:       2026-08-06
CARD:       deadline_existing_hard_fable_v2_h1
CHANGE:     Submit the exact historically strongest legal Hard source, Fable
            v2, with no architecture, optimizer, or training change.
PREDICT:    It should exceed the 0.0367% final-label VDF Hard score because
            its prior hosted run reached 0.0467%. Refutation: <=0.0367% or a
            validation/runtime failure.

DATE:       2026-08-08
CARD:       hard_t1_weighted_exact_match
CHANGE:     In the 2026-08-07 GPT-5 Pro exact-match/SAM card, weight prompt
            rows with T=1 by 8x inside the existing sequence-aware loss and
            normalize row weights to mean one. Architecture, optimizer, SAM,
            batch reuse, recurrence, and all inference behavior are unchanged.
PREDICT:    Concentrating the fixed training budget on the first Hard rung
            will produce nonzero exact accuracy at seen-N or OOD-N T=1 even
            if aggregate Easy mean stays flat or falls. Refutation: both T=1
            profiles remain exactly zero after the bounded Easy screen.
RESULT:     confirmed narrowly on the screen — local e5 produced 5/512 seen-N
            T=1 and 0/512 OOD-N T=1; hosted e5 produced 3/512 and 0/512.
            Hosted aggregate mean fell to 0.375%, and total T=1 hits tied the
            parent card. Final Hard refutes transfer: 0/768 at T=1 on both
            seen-N and OOD-N profiles, no certified rung, and 0.02333% overall
            (3/9,999 test, 2/10,002 OOD-T, 2/10,002 OOD-N).

DATE:       2026-08-08
CARD:       t1_phase_square_reduce_information_flow
CHANGE:     In a matched final-label-only T=1 decimal model, hide N during a
            four-step square phase, then expose N during a four-step reduction
            phase. Compare with an identical-depth/parameter control that sees
            N in both phases. No arithmetic intermediates enter model or loss.
PREDICT:    Withholding N will impede per-modulus table memorization and force
            an N-independent x² representation: across seeds 0/1/2, factored
            median unseen-N exact will be >=35%, held-out-x >=25%, and unseen-N
            will beat entangled by >=10 points. Kill if factored unseen-N is
            <=20% or has no consistent advantage. Budget: 18 GPU minutes.
RESULT:     refuted — factored medians are 11.76% held-out-x and 17.29%
            unseen-N versus 13.03% and 17.29% for entangled; all six runs fit
            train 100%. Hiding N neither forces squaring nor improves transfer.

DATE:       2026-08-08
CARD:       t1_pairfold_square_reduce_final_label
CHANGE:     Replace the refuted generic x-only square phase with learned digit-
            pair categories, fixed pair-to-column routing, a shared pair fold,
            and an LSD-first learned carry scan. Keep final-label-only training
            and the four-step N-conditioned learned reduction phase.
PREDICT:    The pair-routed square tape will attack held-out-x composition:
            median held-out-x and unseen-N exact across seeds 0/1/2 will each
            be >=25%, and unseen-N will beat the generic factored median by
            >=10 points. Kill if median held-out-x is <=15% or unseen-N fails
            to beat 17.29%. Budget: 9 GPU minutes.
RESULT:     refuted — all seeds fit train 100%, but median held-out-x is
            10.50% and unseen-N 16.36%, both worse than the generic factored
            tape. Pair topology does not repair final-label credit assignment.

DATE:       2026-08-09
CARD:       t1_factored_e5_support
CHANGE:     Keep the refuted factored PhaseSquareReduce model, optimizer,
            four square/four reduction steps, T=1 final-label loss, seed, and
            180-second budget fixed. Replace only its 18 tiny two-digit
            training moduli with the public Easy e5 T=1 training rows, then
            evaluate the public seen-N and 12–13-bit OOD-N T=1 profiles.
PREDICT:    Broader modulus support will reduce per-N shortcutting enough to
            raise OOD-N T=1 exact from the prior 17.29% median to >=25%, while
            seen-N reaches >=50%. Refutation: OOD-N <20%, seen-N <35%, or the
            training rows do not reach 99% exact. Screen seed 0 first; run two
            more seeds only if the OOD-N promotion threshold clears.
RESULT:     refuted at seed 0 — the model fit all 1,600 public T=1 training
            rows exactly in 18,019 updates, but reached only 7/512 (1.37%)
            seen-N and 1/512 (0.20%) OOD-N exact. Both kill thresholds fired,
            so no additional seeds are authorized. Broader modulus exposure
            strengthened row memorization without identifying modular square.

DATE:       2026-08-09
CARD:       t1_factored_e5_interface_noise
CHANGE:     Relative to the public-support seed-0 anchor, add only training-time
            Gaussian noise with std 0.1 after the fourth square step and before
            reduction. Data, parameters, optimizer, seed, final-label loss,
            evaluation, and 180-second budget remain fixed.
PREDICT:    If OOD failure is caused by a brittle continuous hidden code, noise
            will retain >=95% train exact while lifting seen-N T=1 to >=5% and
            OOD-N T=1 to >=2%. Kill if train exact is <95% or OOD-N remains
            <=1%. Screen seed 0; run more seeds only if all promotion gates pass.
RESULT:     refuted at seed 0 — noise retained 1,599/1,600 (99.94%) train exact
            but reduced seen-N to 2/512 (0.39%) and OOD-N to 0/512. The OOD
            kill threshold fired and both profiles are worse than the no-noise
            anchor's 7/512 and 1/512, so no additional seeds are authorized.

DATE:       2026-08-09
CARD:       fable_tcap_completion_e2
CHANGE:     Hold exact source SHA-1 aa75819a878fab6c03c6a23d979f6234560f6e3d
            fixed and change only the hosted dataset from e1/e5 anchors to e2
            (fixed N=899, T=1/2/4).
PREDICT:    Fixed-N answer-space structure will preserve a nontrivial 3–9% mean,
            but no T rung will certify. Refuted if mean is <3% or any rung certifies.
RESULT:     refuted — e2 scored 1.21% mean (about 2.1% test / 0.3% OOD)
            after 634 updates, with no certified rung. The e1 fixed-N score
            does not transfer to a second modulus.

DATE:       2026-08-09
CARD:       fable_tcap_completion_e3
CHANGE:     Same exact source; evaluate e3 (10–11-bit varying N, fixed T=2).
PREDICT:    Removing mixed T will help modestly relative to e5: 1–3% mean exact.
            Refuted if mean is <1% or >3%; no rung prediction because e3 has no T=1.
RESULT:     refuted — e3 scored 0.50% mean with no certified rung. Fixing T=2
            does not remove the varying-modulus generalization failure.

DATE:       2026-08-09
CARD:       fable_tcap_completion_e4
CHANGE:     Same exact source; evaluate e4 (11–12-bit varying N, fixed T=2).
PREDICT:    Larger moduli will reduce exactness below e3, with 0.3–1.5% mean.
            Refuted outside that interval; no rung prediction because e4 has no T=1.
RESULT:     refuted narrowly — e4 scored 0.27% mean with no certified rung,
            below the predicted interval and below e3's 0.50%.

DATE:       2026-08-09
CARD:       fable_tcap_completion_m2
CHANGE:     Same exact source; evaluate m2 (fixed N=38,021, T=4/8/16).
PREDICT:    Like m1, long training will fit without a reusable transition;
            mean exact will remain 0–0.2% and no rung will certify.
RESULT:     confirmed — m2 scored 0.15% mean with no certified rung.

DATE:       2026-08-09
CARD:       fable_tcap_completion_m3
CHANGE:     Same exact source; evaluate m3 (11/13/15-bit N, fixed T=2).
PREDICT:    Variable-N exactness will be 0–0.3%, with larger-N buckets weakest.
            Refuted if mean exceeds 0.3%.
RESULT:     confirmed — m3 scored 0.27% mean with no certified rung.

DATE:       2026-08-09
CARD:       fable_tcap_completion_m4
CHANGE:     Same exact source; evaluate m4 (14/18/22-bit N, fixed T=8).
PREDICT:    This is the hardest public geometry for the card: mean exact will
            be <=0.1%, with no evidence of an eight-step reusable transition.
RESULT:     confirmed — m4 scored 0.0778% mean with no certified rung.

DATE:       2026-08-09
CARD:       t1_factored_e5_identity_reducer
CHANGE:     Relative to the deterministic public-support seed-0 anchor, replace
            each reduction assignment by one learned scalar residual gate
            initialized at 0.01. Everything else, including final-label-only
            supervision and the 180-second clock, remains fixed.
PREDICT:    If a random reducer blocks useful credit to the square phase, the
            gate will grow above 0.05 while retaining >=95% train exact and
            lifting seen-N T=1 to >=5% and OOD-N T=1 to >=2%. Kill if train is
            <95% or OOD-N remains <=1%; add seeds only if all gates pass.
RESULT:     refuted — the gate opened from 0.01 to 0.759 and train reached
            1,600/1,600, but transfer was exactly the deterministic anchor:
            7/512 seen-N and 1/512 OOD-N. The OOD kill threshold fired, so no
            additional seeds are authorized.

DATE:       2026-08-08
CARD:       hard_exact_source_e5_replication
CHANGE:     None. Re-run SHA-1 8c796bf39f3b0d2f90043b08430be26c23f0f180,
            the exact active Hard source, on public Easy e5 on the idle L40.
PREDICT:    Reproduce the prior local 0.4583% mean neighborhood and retain at
            least one seen-N T=1 exact hit. Refutation: zero seen-N T=1 hits or
            an absolute mean shift greater than 0.50 percentage points.
RESULT:     confirmed — exact SHA produced 0.6250% mean, 4/512 seen-N T=1,
            and 1/512 OOD-N T=1, within 0.1667 points of the prior local mean.
            No rung certified; this validates identity/runtime, not mechanism.

DATE:       2026-08-08
CARD:       t1_identifiability_ablation
CHANGE:     Replace the prompt-reinjecting recurrence with a canonical mutable
            LSD-first register initialized from x once, immutable N context,
            and T as loop count only. Tie answer logits to state logits. Compare
            plain CE/AdamW with a T=1-only first 50% curriculum and 4x late T=1
            weight. No intermediate label or arithmetic oracle is introduced.
PREDICT:    The canonical curriculum arm gives the strongest T=1 transfer.
            Promotion requires >=5% exact on both seen-N and OOD-N T=1 and >=3
            points over canonical plain. Kill if OOD-N T=1 stays below 1% or
            canonical plain matches it within 1 point. The older prompt-
            reinjecting curriculum is a loose architecture baseline, not a
            parameter-matched control. Budget: three 60-second e5 runs.
RESULT:     refuted for promotion, positive as a weak first-rung bet. Canonical
            curriculum reached 5/512 seen-N and 2/512 OOD-N T=1 with 0.6667%
            e5 mean, versus canonical plain at 1/512 and 1/512 with 0.5833%
            mean. Prompt reinjection reached 1/512 and 1/512 despite 0.7083%
            aggregate mean. The treatment misses the 5%/5% and 3-point gates.

DATE:       2026-08-08
CARD:       canonical_register_m1_downward_transfer
CHANGE:     Run the canonical mutable-state card for the full public Medium m1
            budget. Because m1 has only T=4/8/16 training labels, the T=1
            curriculum automatically falls back to ordinary final-label
            training at actual T. Architecture and optimizer are unchanged.
PREDICT:    If composed final labels identify the tied transition, the final
            checkpoint will produce >=1% seen-N T=1 and >=0.2% OOD-N T=1.
            Kill the Hard thesis if both T=1 profiles are zero. Budget: one
            600-second m1 run, seed 74.
RESULT:     refuted — after 9,815 updates / 600.02 seconds, final train CE was
            2.2930, test was 0/3,000, OOD was 2/3,000, and both seen-N and
            OOD-N T=1 profiles were exactly zero (0/192 and 0/512). Composed
            T=4/8/16 labels did not identify the canonical one-step cell.

DATE:       2026-08-08
CARD:       canonical_register_e5_seed_robustness
CHANGE:     Repeat the exact robust canonical-register curriculum source on
            public e5 with runtime seeds 75 and 76; seed 74 is already complete.
            Data, budget, optimizer, architecture, and curriculum are fixed.
PREDICT:    The three-seed median will retain >=3/512 seen-N T=1 and >=1/512
            OOD-N T=1. Refutation: either new seed is zero on both profiles or
            the median falls below either threshold. Budget: two 60-second runs.
RESULT:     confirmed — seeds 74/75/76 produced seen-N T=1 counts 5/4/3 and
            OOD-N counts 2/1/1, so medians were 4 and 1. E5 means were
            0.6667%/0.7500%/0.8333%. The signal is weak but not single-seed.

DATE:       2026-08-08
CARD:       canonical_dynamic_slots_hosted_repeat
CHANGE:     Resubmit exact SHA-1 5b622f06680600f4b346e34b635b839dde18471c
            to hosted Easy e5. No source, dataset, or tier change.
PREDICT:    The repeat retains nonzero seen-N and OOD-N T=1. Refutation: either
            first-rung profile is exactly zero. Budget: one hosted Easy e5 slot.
RESULT:     confirmed — the repeat improved to 5/512 seen-N and 4/512 OOD-N
            T=1 with 0.8333% mean and 1,189 updates. The first exact-source
            hosted run was 2/512 and 3/512 with 0.7083% mean and 1,036 updates.

DATE:       2026-08-08
CARD:       canonical_register_lr6e3
CHANGE:     Increase only AdamW peak learning rate from 3e-3 to 6e-3 in the
            compact width-256, batch-512 canonical card. Schedule, warmup,
            architecture, curriculum, and all other hyperparameters are fixed.
PREDICT:    Earlier learning will yield >=8/512 seen-N and >=2/512 OOD-N T=1
            with e5 mean >=0.60%. Promote only if both T=1 counts match or beat
            compact 3e-3 and total hits exceed 12. Budget: one 60-second e5 run.
RESULT:     refuted — 2,422 updates reached only 0/512 seen-N and 2/512 OOD-N
            T=1 with 0.5000% mean. Higher LR delayed useful fitting and shifted
            chance hits away from the primary seen-N profile.

DATE:       2026-08-08
CARD:       canonical_register_batch256
CHANGE:     Reduce only training batch size from 512 to 256 in the compact
            width-256 canonical card. Evaluation batch, model, optimizer,
            curriculum, state, data, and wall-clock budget remain fixed.
PREDICT:    E5 will complete >=3,500 L40S updates and retain >=8/512 seen-N
            plus >=1/512 OOD-N T=1. Promote only if both profile counts are
            nonzero and total T=1 hits exceed compact batch-512's 12.
            Budget: one 60-second e5 run, seed 74.
RESULT:     refuted — 3,233 updates, below the 3,500 prediction, produced only
            2/512 seen-N and 0/512 OOD-N T=1 with 0.6667% mean. Smaller batches
            delayed the learning transition and lost OOD-N first-rung signal.

DATE:       2026-08-08
CARD:       canonical_register_width128
CHANGE:     Reduce only D_MODEL from 256 to 128 in the dynamic-slot canonical
            card. Heads, layers, state interface, optimizer, curriculum, batch,
            and data remain fixed.
PREDICT:    The smaller card will complete >=4,500 L40S e5 steps while retaining
            at least one exact seen-N and one exact OOD-N T=1 example. Promote
            over width 256 only if total T=1 hits are >=5 and neither profile
            is zero. Budget: one 60-second e5 run, seed 74.
RESULT:     refuted as a speed intervention. L40S completed only 2,390 updates,
            but optimization per update improved; local profiles were 6/512
            seen and 1/512 OOD-N with 0.6250% mean. Hosted execution completed
            568 updates and fell to 2/512 seen, 0/512 OOD-N, so it is rejected.

DATE:       2026-08-08
CARD:       canonical_register_dynamic_slots
CHANGE:     Derive active recurrent register slots from ModelSpec.max_seq_len
            instead of always attending over 16 mutable plus 16 N slots. E5
            uses five active slots per side; larger suites expand automatically.
            Parameters, optimizer, loss, curriculum, routing, and loop semantics
            are unchanged.
PREDICT:    Hosted-style throughput should improve by >=2x without losing the
            public first-rung signal: local e5 must retain >=3/512 seen-N and
            >=1/512 OOD-N T=1. Kill if both profile counts fall or mean exact
            drops below 0.60%. Budget: one 60-second e5 run, seed 74.
RESULT:     mixed prediction, selected for Hard profile evidence. L40S
            throughput was only 2,386 versus 2,306 updates, but local T=1 was
            11/512 seen and 1/512 OOD-N. Hosted exact-source runs completed
            1,036 and 1,189 updates and produced 2/512 + 3/512, then 5/512 +
            4/512 (seen + OOD-N). Hosted means were 0.7083% and 0.8333%.

DATE:       2026-08-08
CARD:       canonical_dynamic_slots_hard_h1
CHANGE:     Submit exact SHA-1 5b622f06680600f4b346e34b635b839dde18471c
            to Hard h1 after two exact hosted e5 runs were nonzero on both T=1
            profiles. No post-screen source change.
PREDICT:    No rung will certify. The first-rung lottery target stated before
            upload is nonzero on both private profiles, plausibly 2–8/768
            seen-N and 1–6/768 OOD-N. Refutation: either T=1 profile is zero.
RESULT:     refuted — job `7714d650-78a4-4d4a-8fc1-a384914d7658` scored
            0.0500% mean exact and certified no rung. Both private T=1
            profiles were exactly zero (0/768 seen-N and 0/768 OOD-N), outside
            the predicted nonzero ranges. The 163,274-update run ended at
            train loss 2.17846; public Easy hits did not transfer.

DATE:       2026-08-09
CARD:       shifted_long_division_reducer
CHANGE:     Starting from the qualified width-14 learned unit reducer, expose
            the same comparator/subtractor to decimal-shifted divisors
            N*10^p for p=0..8 and single quotient digits k=0..9. The fixed
            autonomous evaluator sweeps shifts high to low with nine learned
            compare/subtract opportunities per shift.
PREDICT:    If the learned digit-serial primitives are genuinely aligned and
            translation-reusable, unseen-N comparator and subtraction accuracy
            will be >=99.9% at every shift and autonomous remainder exactness
            will be >=99% through q=999,999. Kill if any shift is below 99% or
            q=999,999 is below 95%. Research-only pending Rule-7 audit.
RESULT:     refuted at long range but solved the learned subtraction primitive.
            Every shift reached 100% unseen-N subtraction exactness and at
            least 99.9609% comparator accuracy. Autonomous exactness was 100%
            at q=1/10/100/1000 and 96.19% at q=999, but only 19.34% at q=9,999
            and 37.40% at q=999,999 because rare false leading-place fires
            compound. The registered q=999,999 kill threshold fired.

DATE:       2026-08-10
CARD:       shifted_long_division_boundary_repair
CHANGE:     Freeze the 100%-exact shifted subtractor and fine-tune only the
            comparator on long-division boundary states D-N+r (negative) and
            D+r (positive). Uniform negative sampling almost never presents
            D-N+r when D=N*10^p at large p.
PREDICT:    If false leading-place subtractions are the rollout failure, q=9,
            99, 999, 9,999, and 999,999 will all reach >=99% exact while every
            shift retains >=99.9% one-step subtraction/comparator accuracy.
            Refute if q=9,999 remains below 95%. Subtractor is frozen.
RESULT:     mechanism confirmed, full gate mixed. All 11 autonomous quotient
            scales from 0 through 99,999,999 were 100% exact on 1,024 unseen-N
            cases each, with selected subtraction counts exactly matching the
            quotient digit sums. Boundary-only fine-tuning degraded uniform
            one-step comparator accuracy to 99.4922% at the weakest shift,
            below the registered 99.9% requirement; subtraction stayed 100%.

DATE:       2026-08-10
CARD:       shifted_long_division_mixed_comparator
CHANGE:     Starting again from the first shifted reducer, freeze the solved
            subtractor and fine-tune the comparator on a shuffled mix of the
            boundary cases and its original uniform k=0..9 support.
PREDICT:    Consolidation succeeds if all autonomous quotient scales through
            99,999,999 remain >=99.9% exact and every shift's uniform one-step
            comparator accuracy is >=99.9%. Refute if either minimum fails.
RESULT:     confirmed. Every shift retained 100% subtraction exactness and
            99.9609%--100% comparator accuracy on unseen N. Autonomous
            reduction was 100% exact at 10/11 quotient scales and 1023/1024
            (99.9023%) at q=100; all scales through 99,999,999 cleared 99.9%.
            The selected subtraction count exactly matched the quotient digit
            sum at every fully exact scale.

DATE:       2026-08-10
CARD:       canonical_local_conv
CHANGE:     Add one translation-shared depthwise-Conv1d/GLU residual over
            adjacent LSD-first mutable digits after the canonical register's
            unchanged global attention step. All data, state routing,
            curriculum, optimizer, recurrence, and final-label loss are fixed.
PREDICT:    Local arithmetic propagation improves the public e5 T=1 profiles
            beyond the canonical chance-scale anchor. Hard promotion requires
            hosted mean >1% and >=10/512 exact on both seen-N and OOD-N T=1.
            Kill on either zero profile or any legality/validation failure.
RESULT:     refuted. Local e5 was 0.6667% with T=1 5/512 seen and
            1/512 OOD-N; hosted job d53c55a8 was 0.2917% with 3/512 and
            1/512. Source audit found sigmoid(0.1)=0.525, so this is the
            strong-residual card rather than the intended 0.1 multiplier.

DATE:       2026-08-10
CARD:       canonical_local_conv_scale01
CHANGE:     Relative to the failed strong-residual card, change only the local
            ConvGLU residual multiplier from sigmoid(0.1)=0.525 to a directly
            learned scalar initialized at exactly 0.1.
PREDICT:    Preserving 90% of the canonical path should recover its optimization
            while retaining useful local propagation. Promote to the forced
            Hard bet only if hosted e5 is nonzero on both T=1 profiles and
            improves either total T=1 hits or mean exact over strong residual.
            Kill on either zero profile.
RESULT:     refuted. Local mean improved to 0.7500% with T=1 3/512 seen and
            1/512 OOD-N, but hosted job 00fdf63b scored 0.4167% and 3/512
            seen plus 0/512 OOD-N. The zero-profile kill fired.

DATE:       2026-08-10
CARD:       canonical_local_conv_forced_hard
CHANGE:     Upload the exact strong-residual Easy-tested source SHA-1
            64639a3c3c51aa0ee6ab23f5cc286e2dc0c1a05a to Hard h1 with no
            post-screen source change, at the owner's explicit request.
PREDICT:    No rung certifies. The chance-scale target is 0--4/768 on each
            private T=1 profile; either profile may be zero. A certified T=1
            rung would falsify the expectation and promote local mixing.
RESULT:     refuted. Hosted job f79ebe42-b146-4cce-92e5-1e980c27d55e scored
            0.02333% mean exact (3/9,999 test, 1/10,002 OOD-T, 3/10,002
            OOD-N), certified no rung, and produced 0/768 on both seen-N and
            OOD-N T=1. It completed 148,084 updates with final train loss
            2.2095. Generic local mixing did not transfer to hidden T=1.

DATE:       2026-08-10
CARD:       canonical_local_conv_scale_trajectory
CHANGE:     Add read-only logging every 100 optimizer steps to the exact local
            true-0.1 control: residual scalar value/gradient and local Conv/GLU
            weight norms. Training computation and seed are unchanged.
PREDICT:    If locality is merely underused, the scalar stays in [0.05,0.20]
            with vanishing gradients. If the branch actively conflicts with
            canonical learning, scale or local norms grow with nontrivial
            gradients while held-out T=1 remains chance-scale. Stop after one
            public-e5 60-second local mirror; no hosted submission.
RESULT:     locality-underuse refuted. The scalar grew from 0.10297 after the
            first step to 1.77103 at step 1600, and the GLU norm grew from
            9.119 to 44.360 with nonzero gradients. Yet local public e5 T=1
            remained chance-scale at 5/512 seen-N and 2/512 OOD-N (0.8333%
            mean exact). Final-label training actively recruits the branch,
            but does not turn it into a transferable local arithmetic rule.

DATE:       2026-08-10
CARD:       canonical_worst_digit_loss
CHANGE:     On the exact unmodified canonical architecture, replace only the
            per-example mean token CE with a tau=0.5 smooth maximum over valid
            output-digit CE. Keep data, curriculum, T=1 weighting, optimizer,
            recurrence, architecture, and seed unchanged.
PREDICT:    If mean CE hides a small number of decisive carry/reduction digit
            errors, local public e5 reaches >=8/512 seen-N and >=5/512 OOD-N
            T=1. Refute and stop after one local run if either gate fails. No
            hosted or Hard submission is permitted from this card.
RESULT:     refuted by OOD-N. Local e5 reached the seen-N boundary exactly at
            8/512 T=1 but only 1/512 OOD-N; mean exact was 0.5833%. Emphasizing
            the hardest output digit sharpened seen-modulus fit without
            producing a modulus-general reduction rule.

DATE:       2026-08-10
CARD:       canonical_full_t1_curriculum
CHANGE:     On the exact canonical architecture and ordinary mean CE, change
            only T1_ONLY_FRACTION from 0.50 to 1.00. Every optimizer update is
            then selected from T=1 rows and executes exactly one transition.
PREDICT:    If the late mixed-depth phase overwrites the one-step rule, local
            public e5 improves over the canonical 5/512 seen-N and 2/512 OOD-N
            anchor, clearing >=8/512 and >=5/512. Refute if either gate fails;
            one local run only, with no hosted submission.
RESULT:     refuted. Training loss reached 0.0516, while held-out mean exact
            collapsed to 0.1667% and evaluation loss exceeded 8.5. T=1 was
            4/512 seen-N and 0/512 OOD-N. More one-step gradient intensifies
            memorization rather than identifying the reusable transition.

DATE:       2026-08-10
CARD:       canonical_modulus_length_group_dro
CHANGE:     On the exact canonical architecture and 50% T=1 curriculum, group
            ordinary final-label row CE by the observed decimal length of N
            and use a tau=0.25 smooth maximum over present groups. Keep all
            forward computation, labels, optimizer, seed, and recurrence fixed.
PREDICT:    If common/easy modulus scales dominate gradient, unseen-N T=1
            rises to >=8/512 while seen-N also remains >=8/512. Kill immediately
            if OOD-N <=1/512 or either profile is worse than the full-T1 control.
            One local run only; no hosted submission.
RESULT:     refuted at the immediate kill: local e5 T=1 was 2/512 seen-N and
            1/512 OOD-N (0.7500% mean exact). Equalizing final-label pressure
            over observed modulus lengths did not induce a general rule.

DATE:       2026-08-10
CARD:       canonical_local_conv_k4
CHANGE:     Relative to the exact strong local-Conv card, reuse its same
            translation-shared ConvGLU residual four times instead of once
            inside each modular transition. Parameters, data, loss, schedule,
            state routing, and recurrence are fixed.
PREDICT:    If the k=1 local block fails because carry/reduction information can
            move only one adjacent slot, four tied microsteps clear >=8/512
            seen-N and >=5/512 OOD-N T=1. Refute if either gate fails or the
            lower updates erase any local-profile gain. One local run only.
RESULT:     refuted. Four tied microsteps completed 1,491 updates and produced
            only 3/512 seen-N plus 2/512 OOD-N T=1 (0.3750% mean exact).
            Additional local propagation distance did not recover a reusable
            rule and reduced throughput versus the k=1 card.

DATE:       2026-08-10
CARD:       canonical_structured_tape_residual
CHANGE:     Add one generic LSD-aligned learned tape residual before the exact
            canonical global step: project mutable/N slots to width 128, mix
            left/self/right state, update each slot with one shared GRU paired
            to same-position N, then project back. Keep the canonical discrete
            recurrent state, output head, loss, curriculum, optimizer, and data.
PREDICT:    If explicit state/N alignment plus a gated serial workspace is the
            missing inductive bias, local public e5 reaches >=10/512 on both
            T=1 profiles and >1% mean exact. Kill if either profile is zero or
            both remain below 8/512. One local run; hosted promotion requires
            a fresh decision.
RESULT:     gate refuted with an optimization confound. The card completed
            1,362 updates, ended at train loss 1.8876, and produced 3/512
            seen-N plus 2/512 OOD-N T=1 (0.7917% mean exact). The retained
            canonical attention plus tape slowed and underfit; this does not
            cleanly falsify the direct structured topology.

DATE:       2026-08-10
CARD:       direct_structured_tape
CHANGE:     Replace the canonical attention+STE transition with the exact
            research topology: continuous LSD-aligned slot state initialized
            from paired x/N digits, one shared left/self/right mixer, one shared
            per-slot GRU conditioned on same-position N, and a shared decoder.
            Keep prompt parser, final-label CE, T curriculum, optimizer, and data.
PREDICT:    If the positive small-N result transfers when its topology is not
            bottlenecked by canonical attention, local public e5 reaches >1%
            mean and >=10/512 on both T=1 profiles. Kill if OOD-N <=2/512 or
            train loss remains above 1.0. One local run, no hosted submission.
RESULT:     60-second gate refuted. The 263,473-parameter tape completed 1,953
            updates, ended at train loss 1.0198, and produced only 2/512 seen-N
            plus 1/512 OOD-N T=1 (0.4167% mean exact). Both OOD and underfit
            kills fired, leaving optimization versus topology unresolved.

DATE:       2026-08-10
CARD:       direct_structured_tape_300s
CHANGE:     Run the exact failed direct-tape source and public e5 data for 300
            local seconds instead of 60; evaluator, seed, optimizer, schedule,
            architecture, and all other manifest fields are fixed.
PREDICT:    If 60 seconds was the limiting factor, final train loss falls below
            0.2 and both T=1 profiles reach >=10/512. If train loss falls below
            0.2 while OOD-N remains <=2/512, classify the topology as another
            memorizer. Stop after this one local diagnostic.
RESULT:     refuted without reaching either predicted branch. After 9,640
            updates, final train loss plateaued at 0.8536 rather than <0.2;
            held-out losses exploded to 7.394/9.253 and T=1 was 2/512 seen-N
            plus 0/512 OOD-N. More optimization increases specialization but
            does not uncover the reusable rule.

DATE:       2026-08-10
CARD:       direct_structured_tape_no_position
CHANGE:     Relative to the exact 60-second direct tape, remove only learned
            absolute place embeddings from input and decoder. LSD ordering,
            zero boundaries, shared neighbor mixer/GRU, width, loss, optimizer,
            curriculum, data, and 60-second budget remain fixed.
PREDICT:    If absolute position lets the tape hash the 27 training moduli,
            translation equivariance improves T=1 to >=5/512 on both profiles
            and >0.75% mean exact despite weaker train fit. Kill if OOD-N stays
            <=2/512 or mean exact does not improve over 0.4167%.
RESULT:     refuted. The card completed 1,864 updates and scored 0.5000% mean
            exact, with 3/512 seen-N and 1/512 OOD-N T=1. Removing absolute
            position slightly improved aggregate exactness over the parent but
            left unseen-N T=1 at chance; the OOD kill fired.

DATE:       2026-08-10
CARD:       x2modn_sanity_seed0
CHANGE:     Reproduce the existing two-digit square-to-reduce diagnostic on the
            newly provisioned L40 with VDF-square reducer trace support.
PREDICT:    Protocol violation: this prediction was recorded after launch. The
            intended sanity expectation was approximately 100% unseen-N square
            exact and greater than 90% full T=1 exact; failure would invalidate
            the environment or reproduction before new research.
RESULT:     reproduction passed, but is ineligible as a preregistered decision
            card. Unseen-N square was 100.00%, full T=1 was 95.79%, and T=8 was
            92.52%. The known O(q) scaling limitation remains.

DATE:       2026-08-10
CARD:       x2modn_direct_mlp
CHANGE:     Replace the trace-supervised digit-serial arithmetic system with a
            roughly parameter-matched plain GELU MLP trained only on final
            width-three x2 mod N labels. Use 185 train semiprime N, 39
            validation N, 41 exhaustive test N, and three seeds.
PREDICT:    The MLP reaches >=99% train exact but remains below 10% on unseen x
            for seen N and below 2% exact on unseen N in every seed. A reusable
            function requires >=90% unseen-N exact in all three seeds; anything
            lower is not realistically reliable program learning.
RESULT:     mechanism confirmed, exact numerical floor refuted. All seeds fit
            train 100%; seen-N/unseen-x was 4.45%--4.53% and unseen-N was
            3.79%--3.92%, not <2%. A post-run audit puts identity at 0.77%,
            constant zero at 0.19%, and a test-label digit-mode vector at 0.23%,
            so the MLP learns partial correlations beyond trivial frequency.
            With only 19.55%--19.82% digit accuracy and loss above 11.6, it is
            still memorization rather than modular-squaring discovery.

DATE:       2026-08-10
CARD:       x2modn_direct_transformer
CHANGE:     Replace only the direct 2.19M-parameter MLP with a roughly matched
            standard four-layer, eight-head, pre-norm Transformer encoder with
            learned output-query tokens. Freeze data, splits, final-label loss,
            AdamW, batch size, update count, dropout, and three seeds.
PREDICT:    Attention slightly improves shared digit interactions but all seeds
            remain below 10% unseen-N exact. A reusable function requires >=90%
            unseen-N exact in all seeds; >=50% in any seed would instead be
            evidence of partial algorithm learning worth a scaling follow-up.
RESULT:     confirmed. All seeds fit train 100%, while unseen-N exact was only
            4.06%--4.26% and unseen-N digit accuracy 19.62%--19.66%. This is a
            repeatable 0.17--0.47 point improvement over the MLP, but about 96%
            of unseen outputs remain wrong and held-out cross-entropy exceeds
            11.0. No seed shows partial algorithm learning.

DATE:       2026-08-10
CARD:       multilane_neural_gpu_square
CHANGE:     Replace the explicit digit-serial Square cell with the generic
            six-lane local grid forward: eight LSD-first positions, width 64,
            learned left/self/right and lane mixing, and one tied GRU update for
            16 microsteps. Train only on final raw-square labels for 8,000 x and
            evaluate 2,000 disjoint x; no N/reduction signal enters this card.
PREDICT:    If the generic grid can represent and discover decimal squaring,
            seed 0 reaches >=90% unseen-x exact (strong >=99%) and >=90% train
            exact after 12,000 updates. Kill below 50% unseen-x or 90% train;
            do not add reduction, more depth, or optimizer tuning after a kill.
RESULT:     refuted; both kill conditions fired. Final train exact was 12.9125%
            and unseen-x exact was 4.0000%, with 81.9000%/75.6125% per-digit
            accuracy. The grid learned transferable digit correlations but no
            exact squaring procedure, and no grokking transition occurred.

DATE:       2026-08-10
CARD:       multilane_neural_gpu_square_carry_aux
CHANGE:     Relative to the failed answer-only multi-lane Neural GPU, add only
            a 130-parameter head and weight-1 MSE for normalized carry-in and
            carry-out at each output column. Freeze the split, backbone, main
            loss, optimizer, batch, 12,000 updates, and seed.
PREDICT:    If carry credit assignment is the main blocker, unseen-x exact is
            >=50% and both central digit positions are >=70%; >=70% exact is
            strong confirmation. Kill below 20% exact or below 50% at either
            central position. A pass requires a shuffled-carry control before
            attributing the gain specifically to carry semantics.
RESULT:     refuted at 12,000 updates. Unseen exact was 4.5500% versus the
            answer-only parent's 4.0000%, and the two central positions were
            28.00%/16.45% versus 29.05%/15.20%. Normalized unseen carry MSE was
            0.02708, so terminal carry was decodable but not usefully consumed.

DATE:       2026-08-10
CARD:       multilane_neural_gpu_square_carry_50k
CHANGE:     Extend both the answer-only and carry-supervised Neural GPU from
            12,000 to 50,000 updates, preserving each script's seed, data,
            architecture, loss, constant AdamW settings, batch, and 500-step
            evaluation cadence. The within-budget comparison changes only
            correct carry supervision and its 130-parameter head.
PREDICT:    If the 12k null is the delayed-learning pattern seen in the old
            Transformer, carry-supervised unseen exact reaches >=50% and beats
            the matched answer-only run by >=20 points. Kill the carry route if
            it remains <20% or leads by <10 points. If both rise similarly,
            duration rather than carry semantics is the mechanism.
RESULT:     refuted; both kill conditions fired. At 50,000 updates answer-only
            reached 3.8500% unseen exact and carry-supervised reached 6.2500%,
            a 2.40-point advantage. Central digits changed only from
            29.10%/15.60% to 29.95%/21.80%. Carry is a modest regularizer here,
            not the delayed algorithmic transition seen in the old Transformer.

DATE:       2026-08-10
CARD:       easy_multilane_neural_gpu
CHANGE:     Replace the legal translation-equivariant direct tape's single
            state lane/update with six generic 64-wide scratch lanes and four
            tied local GRU microsteps per requested recurrence step. Freeze its
            parser, curriculum, final-label loss, optimizer, schedule, and batch.
PREDICT:    On local Easy e5, improve over 0.5000% mean exact and reach >=5/512
            on both T=1 profiles plus >0.75% mean for promotion. Kill at OOD-N
            T=1 <=2/512 or mean <=0.5000%. A user-requested hosted e5 after a
            local kill is forced evidence, not promotion.
RESULT:     local promotion refuted. Mean exact improved to 0.6667%, but T=1
            was only 1/512 seen-N and 2/512 OOD-N; the registered OOD kill
            fired. For the owner-requested forced hosted e5, predict 0.3%--1.0%
            mean, no certified rung, and <=4/512 on each T=1 profile.
RESULT:     hosted prediction confirmed. Exact SHA-1 c436691686c76e406445484b64849ac06eac5cac
            scored 0.3333% mean (0.5000% test, 0.2000% OOD), certified no rung,
            and completed only 532 updates. Bounded hosted metrics omit exact
            T=1 counts. Stop this source before Medium or Hard.

DATE:       2026-08-11
CARD:       canonical_dynamic_slots_hosted_third
CHANGE:     Resubmit exact SHA-1 5b622f06680600f4b346e34b635b839dde18471c
            to hosted Easy e5 for a third exact-source noise observation. No
            source, dataset, tier, optimizer, or architecture changes.
PREDICT:    The third run retains nonzero seen-N and OOD-N T=1, while remaining
            below 2% mean and certifying no rung. Refutation: either first-rung
            profile is exactly zero. This measures chance-scale stability and
            cannot promote the mechanism after its prior 0/768 + 0/768 Hard T=1.
RESULT:     refuted on OOD-N stability. Hosted job
            cfb0fc73-2fe7-4fda-9280-52e8c78ca126 scored 0.5000% mean after
            1,569 updates, with 2/512 seen-N T=1 but 0/512 OOD-N T=1 and no
            certified rung. The exact-source public dual-profile signal is not
            reproducible enough to justify repeating its failed Hard source.

DATE:       2026-08-11
CARD:       fable_tcap_adamw_m5_hosted_repeat
CHANGE:     Resubmit exact SHA-1 aa75819a878fab6c03c6a23d979f6234560f6e3d
            to hosted Medium m5. No source, model, optimizer, batch, dataset, or
            tier changes relative to its recorded exact-source m5 evaluation.
PREDICT:    Mean exact remains in the chance-scale 0.05%--0.30% interval, no
            rung certifies, and both T=1 profiles remain below 1%. Refutation is
            mean >0.30% or either certified T=1 rung. This is a stability screen
            for the strongest historically observed Hard family, not promotion.
RESULT:     confirmed. Hosted job 233cbce0-ab80-4ef7-9769-724da9349e52
            scored 0.1556% mean (13/9,000 test, 5/3,000 OOD), certified no rung,
            and had 0/768 on both T=1 profiles after 13,672 updates. The family
            is a stable chance-scale baseline, not a learned one-step operator.

DATE:       2026-08-11
CARD:       fable_tcap_adamw_forced_hard_refresh
CHANGE:     Upload exact validated SHA-1 aa75819a878fab6c03c6a23d979f6234560f6e3d
            to Hard h1 with no source change after its exact-source Easy e1 and
            Medium m5 evidence. This is the owner's scheduled forced Hard pivot.
PREDICT:    No rung certifies and both T=1 profiles are 0/768; mean exact falls
            in 0.02%--0.08%. A nonzero first-rung profile would be interesting
            but not mechanism validation; a certified T=1 rung would refute the
            chance-baseline model and trigger full source/mechanism audit.
RESULT:     confirmed. Hard job 05f53719-7717-4923-88d5-a3cafe373167 scored
            0.0300%, certified no rung, and produced 0/768 on both T=1 profiles.
            It completed 45,376 updates with final train loss 2.16981. The exact
            source remains a chance-scale statistical learner, not an operator.
CARD:       2026-08-12_causal_message_hard
CHANGE:     Replace the gated tape's generic local convolution residual with explicit learned left/right messages consumed by the next recurrent update.
PREDICT:    No certified Hard rung and 0--1/768 exact T=1 hits per profile; nonzero would motivate controlled reproduction, while 0/768 on both refutes direct transfer.

DATE:       2026-08-14
CARD:       reduction_only_depth_sweep_seed0
CHANGE:     Remove learned squaring by supplying the exact x-squared digits, train one final-label-only local recurrent reducer for eight tied updates, then vary only its evaluation-time update count.
PREDICT:    Within a five-minute cap, the eight-step checkpoint fits its training rows but remains below 25% unseen-N exact. Stop early after three consecutive full-train checks at >=99.9%. If insufficient computation time is the main cause, unseen-N exact rises by at least 10 points at 16 or 32 updates; a flat or collapsing curve refutes duration and implicates the learned reduction rule/controller. Broken multiplication is excluded because the correct product is the input.
RESULT:     confirmed. Early stopping fired after 800 updates and 8.75 seconds with 100% train exact. At the trained depth, held-out-x was 11.34% and unseen-N was 18.69%; increasing evaluation depth to 16/32 reduced unseen-N to 17.99%/16.36%. Extra computation time is refuted for this checkpoint; final-label reduction-rule generalization remains the failure.

DATE:       2026-08-14
CARD:       reduction_only_grokking_60s_seed0
CHANGE:     Extend only the fresh-seed training horizon from the converged 8.75-second stopping point to 60 seconds, preserving data, model, eight recurrent updates, optimizer, and final-label objective; record held-out-x and unseen-N every 400 updates without using them for stopping.
PREDICT:    Training exact reaches 100% early. A grokking signal requires unseen-N exact to rise after interpolation and finish above 30%; remaining below 25% confirms continued memorization over this one-minute horizon.
RESULT:     confirmed continued memorization. Train exact reached 100% by 5.11 seconds and stayed there. Unseen-N fluctuated between 16.82% and 19.16% after interpolation, peaked at 28.35 seconds, and finished at 17.76%; held-out-x finished at 10.92%. No delayed generalization transition occurred within 60 seconds.

DATE:       2026-08-14
CARD:       transformer_2x2_multiplication_representation_factorial
CHANGE:     On one fixed commutativity-safe 80/20 split of every ordered pair in 0..99 squared, compare four otherwise identical encoder-decoder Transformers: MSD-first versus LSD-first digits crossed with natural-width versus explicit 2/4-digit leading-zero padding.
PREDICT:    All four arms fit train above 99%. LSD-first natural-width produces the best held-out exact product accuracy because output order follows carry direction without making leading zeros dominate the target. Padding improves output-length accuracy but not numerical exactness. A test result below 90% for every arm refutes this small Transformer as an adequate fixed-width multiplication baseline under 4,000 updates.
RESULT:     refuted as a multiplication baseline. No arm fit above 95% train or exceeded 31.66% test. MSD-natural unexpectedly led test exact at 31.66% versus LSD-natural 26.92%; padding reduced numerical exactness to 23.13%/19.94% while increasing length accuracy to 99.85%/100%. All arms stayed below 90%, and accuracy collapsed monotonically with carry count and product length.

DATE:       2026-08-14
CARD:       serial_vdf_style_2x2_multiplication
CHANGE:     Replace the best fixed-width Transformer with the project's 64-wide digit-serial GRU: scan paired LSD-first operand digits, then two zero flush positions, predicting one product digit per step. Preserve the exact unordered-pair split, optimizer family, batch size, update count, and final-digit objective.
PREDICT:    The serial recurrence improves three-carry and four-digit product accuracy over MSD-Transformer but remains below 90% total held-out exact because aligned paired-digit scanning does not expose cross-position products. Below 31.66% test exact refutes even a benefit over the generic Transformer.
RESULT:     refuted. The 34,762-parameter serial GRU reached only 28.12% train and 7.18% held-out exact. Its LSD was 100%, but the tens/hundreds digits were 17.75%/37.99%, and three-carry examples reached only 5.24%. Aligned pair scanning lacks accessible cross-position product interactions.

DATE:       2026-08-14
CARD:       simple_neural_gpu_2x2_multiplication
CHANGE:     Replace the serial scan with one 64-wide four-position tape and one tied local ConvGRU cell repeated eight times. Inject both LSD-first operands at every position, preserve the exact split/objective/optimizer/budget, and decode four final product digits. No pair-product routing, carry labels, or intermediate supervision is supplied.
PREDICT:    Local recurrent exchange beats the serial GRU's 7.18% because cross-position digits can meet, but remains below the Transformer 31.66% and below 90% train because a single generic lane has no explicit workspace for multiple partial products and carries. Above 31.66% test would promote the Neural GPU topology for a depth/width follow-up.
RESULT:     confirmed. The 47,562-parameter ConvGRU reached 37.92% train and 17.55% test exact: well above the serial GRU's 7.18% but below the Transformer 31.66%. Test two-digit by two-digit was 12.67%, the tens digit 23.23%, and three-carry cases 8.95%; one generic lane does not organize partial-product/carry state.

DATE:       2026-08-14
CARD:       paper_neural_gpu_decimal_2x2_multiplication
CHANGE:     Replace the simplified tape with arXiv:1511.08228's architecture: a width-4 two-dimensional mental image with 24 maps, two 3x3 CGRU layers repeated sequence-length times, first-column input/output, hard-cutoff gates, 9% recurrent dropout, Adam eps 1e-4, gradient clipping, and six relaxed parameter copies with an increasing pull. Preserve the exact decimal split and final-product objective.
PREDICT:    It beats the simplified Neural GPU's 17.55% held-out exact by providing a wider factored workspace, but remains below 90% because the paper itself reports failure on long decimal multiplication and its strongest result uses binary curriculum plus many seeds. Below 31.66% leaves the Transformer as the fixed-width leader.
RESULT:     refuted. The paper architecture on decimal reached only 13.25% train / 8.23% test exact; two-digit by two-digit was 4.76% and three-carry cases 0.90%. The architecture alone does not overcome the paper's documented decimal failure under this one-seed fixed-width regime.

DATE:       2026-08-14
CARD:       paper_neural_gpu_binary_7bit_multiplication
CHANGE:     Keep the paper architecture and exact numeric split, but change only representation to the paper's successful binary format: two padded 7-bit LSD-first operands separated by MUL, a four-symbol vocabulary, 14 output bits plus PAD, and 15 recurrent steps.
PREDICT:    Binary exceeds the decimal arm's 8.23% held-out exact and fits train materially better. Without the paper's length curriculum, gradient noise, and 729-seed search, it may remain below 90%; exceeding the Transformer 31.66% would establish representation as the main correction.
RESULT:     refuted. Binary lowered token loss but reached only 7.99% train / 7.68% test exact, versus decimal 13.25% / 8.23%. Per-bit accuracy was often high, especially in sparse upper bits, but the complete 14-bit product rarely matched. Architecture plus representation without the paper's curriculum, gradient noise, extensive seed search, and longer optimization does not reproduce the published multiplication result.

DATE:       2026-08-14
CARD:       attached_neural_gpu_adapted_multiplication
CHANGE:     Adapt the supplied binary squarer to multiplication: preserve its single shared 128-channel 3x3 ConvGRU, width-4 by 14-bit workspace, 14 recurrent updates, BCE, AdamW, and 20k budget; place the two 7-bit LSD operands in separate workspace rows and decode 14 product bits from row zero. Preserve the numeric unordered-pair split.
PREDICT:    The simplified shared cell and larger channel budget fit train above 90% and beat the paper-reproduction binary arm's 7.68% held-out exact. Promotion requires exceeding the Transformer at 31.66%; remaining below 20% after 20k refutes this workspace adaptation.
RESULT:     promoted. The 443,393-parameter adaptation reached 100% train and 64.86% final held-out exact, peaking at 65.00% at 19k. It scored 95.67% per-bit, 50.61% on four-digit products, and 50.90% on three-carry cases. The larger shared binary workspace more than doubles the Transformer and is the first generic architecture in this sequence with strong reusable multiplication signal.

DATE:       2026-08-14
CARD:       attached_neural_gpu_decimal_multiplication
CHANGE:     Change only the promoted attached Neural GPU's representation from seven binary input bits/fourteen binary outputs to two LSD-first decimal input digits/four LSD-first decimal outputs. Preserve its 128 channels, four workspace rows, single shared 3x3 ConvGRU, fourteen recurrent updates, split, optimizer, seed, and 20k budget.
PREDICT:    Decimal learns the fixed 0..99 training pairs but generalizes worse than binary's 64.86% because each output position is a ten-way class and a decimal multiplication column must represent larger partial products and carries. Above 64.86% would refute the representation advantage; below the prior decimal Transformer at 31.66% would show the binary encoding is essential here.
RESULT:     confirmed. Decimal reached 99.82% train / 26.92% peak test at 19k (98.15% / 25.27% final), below both the binary Neural GPU's 64.86% and decimal Transformer's 31.66%. Held-out LSD accuracy was 98.80%, but the tens column was only 33.80%, localizing failure to cross-products and carry composition rather than symbol decoding.

DATE:       2026-08-15
CARD:       neural_gpu_multiplication_baseline_recalibration
CHANGE:     Re-run the promoted 128-channel binary shared-cell model unchanged while deterministically splitting its former held-out set into 1,003-row validation and 1,003-row audit halves for honest ablation selection.
PREDICT:    Validation-selected audit exact remains near the prior 65% full-test peak; a large deviation would show seed/run instability and weaken every small ablation conclusion.
RESULT:     confirmed stability. The validation-selected step-10k checkpoint reached 64.81% validation and 63.31% untouched-audit exact, with 50.77% three-carry and 49.92% four-digit exact. This is close enough to the prior 64.86% result to serve as the tournament control.

DATE:       2026-08-15
CARD:       neural_gpu_diagonal_transport
CHANGE:     Partition recurrent channels into learned left-moving, stationary, and right-moving groups before the otherwise unchanged shared ConvGRU update.
PREDICT:    Directional transport improves validation and audit exact by at least three points and specifically improves bits 5--7 and three-carry examples; failure on those buckets refutes transport as the primary residual bottleneck.
RESULT:     refuted. Audit exact fell to 58.72% and three-carry exact to 43.88%. Learned channel transport alone does not resolve the remaining partial-product/carry organization problem. Revert.

DATE:       2026-08-15
CARD:       neural_gpu_recurrent_dropout
CHANGE:     Apply a 9% channel-wise variational mask to candidate state, fixed across all fourteen recurrent updates; preserve everything else.
PREDICT:    Dropout reduces train fit slightly but improves held-out exact by at least two points if the baseline relies on brittle channel conventions; lower audit exact refutes this mask placement.
RESULT:     refuted. Audit exact fell to 59.72% and three-carry exact to 44.39%. This recurrent mask placement removes useful computational state more than it regularizes brittle conventions. Revert.

DATE:       2026-08-15
CARD:       neural_gpu_hard_nonlinearities
CHANGE:     Replace smooth sigmoid/tanh inside the shared cell with hard sigmoid/hard tanh and add a 1e-3 saturation penalty; preserve all other settings.
PREDICT:    Quasi-discrete state improves exactness and extra-step stability, but success requires training interpolation; below 90% train exact refutes this coefficient/activation pair.
RESULT:     refuted. The selected checkpoint reached only 52.54% audit exact and training later destabilized. Hard activations plus this saturation coefficient do not create a more reliable discrete algorithm. Revert.

DATE:       2026-08-15
CARD:       neural_gpu_gradient_noise
CHANGE:     Add zero-mean Gaussian gradient noise with standard deviation 0.03/(1+step)^0.55 before clipping; preserve the baseline architecture and AdamW.
PREDICT:    Noise may escape the 65% shortcut basin and improve audit exact by at least two points; no gain on this seed leaves the technique unclear rather than globally refuted because its literature claim is success probability across seeds.
RESULT:     no gain on this seed. Audit exact fell to 52.14% and three-carry exact to 39.29%. Revert this schedule, while leaving multi-seed gradient-noise claims unresolved.

DATE:       2026-08-15
CARD:       neural_gpu_wide_192
CHANGE:     Increase only recurrent/embedding channels from 128 to 192.
PREDICT:    Added workspace capacity improves central-bit and carry accuracy above baseline, but below a two-point audit gain does not justify its roughly quadratic convolution cost.
RESULT:     promotion gate refuted. Validation rose to 67.00%, but untouched audit reached only 63.91%, a 0.60-point gain, while wall time rose from 184.69s to 348.53s. Revert for the current task.

DATE:       2026-08-15
CARD:       neural_gpu_sharing_relaxation
CHANGE:     Replace the one cell with six cyclic cell copies and linearly increase a parameter-agreement penalty through step 15k.
PREDICT:    Relaxation accelerates fit but may not generalize on fixed width; keep only if audit exact exceeds baseline while validation and audit agree within five points.
RESULT:     refuted. The six-copy model reached 46.86% audit exact despite 2.66M parameters, substantially below the shared-cell baseline. Revert.

DATE:       2026-08-15
CARD:       neural_gpu_muon
CHANGE:     Replace AdamW on matrix/convolution weights with PyTorch Muon while retaining AdamW for vector parameters.
PREDICT:    Muon accelerates optimization but has uncertain final generalization; keep only for at least a two-point audit improvement without destabilizing train exact.
RESULT:     confirmed with a schedule caveat. The selected step-3k checkpoint reached 82.05% validation and 80.96% audit exact, including 77.30% three-carry exact, but constant LR later collapsed. Keep Muon and repair its schedule.

DATE:       2026-08-15
CARD:       neural_gpu_sparse_past_memory
CHANGE:     Every recurrent step after step four receives a learned gated residual from the state four updates earlier.
PREDICT:    Sparse access preserves early operand/partial-product state and improves middle bits; regression indicates stale-state interference rather than missing memory.
RESULT:     refuted. Audit exact fell to 42.17% and three-carry exact to 26.02%, consistent with stale-state interference. Revert.

DATE:       2026-08-15
CARD:       neural_gpu_learned_microprogram
CHANGE:     Replace one tied transition with four cyclic learned transition cells, with no agreement penalty.
PREDICT:    Phase specialization fits train fastest and may improve fixed-width exactness, but likely sacrifices reusable algorithm bias; keep for this diagnostic only if audit exact clearly exceeds baseline.
RESULT:     refuted. Four cyclic cells reached only 49.75% audit exact despite 1.77M parameters. Phase specialization did not beat the tied transition. Revert.

DATE:       2026-08-15
CARD:       neural_gpu_muon_warmdown
CHANGE:     Preserve the winning Muon/AdamW split but decay only Muon's LR from 0.02 after step 1k to 0.002 by step 5k, then hold it; train for the full 10k budget.
PREDICT:    Warmdown preserves Muon's above-80% early solution and avoids the constant-LR collapse, finishing above 80% validation and audit; below the constant-Muon selected checkpoint leaves checkpoint selection preferable.
RESULT:     confirmed. The selected step-4k checkpoint reached 83.85% validation and 83.65% untouched-audit exact, including 78.83% three-carry and 77.76% four-digit exact. The full-budget endpoint remained near 83.55%, eliminating constant-Muon collapse. Keep as tournament winner.

DATE:       2026-08-15
CARD:       neural_gpu_muon_warmdown_post_interpolation
CHANGE:     Continue the isolated Muon-warmdown winner for an additional 5.12M sampled examples after its fixed-compute 5.12M-example endpoint, preserving architecture, data order, optimizer split, and the held 0.002 Muon learning rate.
PREDICT:    Train exact remains at least 98% throughout. Validation remains on its approximately 84% plateau rather than showing a delayed rise of at least five points; this would refute rapid grokking over the preregistered post-interpolation window without ruling out much longer horizons. Any delayed validation rise counts only after train remains near 100%, and audit is opened once after validation-based selection.
RESULT:     refuted rapid grokking. Train stayed 99.9--100% after 3.1M examples, while validation fluctuated around 82--84% and rose only from 83.55% at 5.12M to a validation-selected 84.15% at 9.728M. The once-opened audit was 83.75%. This is a stable interpolation plateau, not a delayed generalization transition.

DATE:       2026-08-15
CARD:       neural_gpu_dropout_fit_matched
CHANGE:     Extend the isolated 9% recurrent-dropout arm from 5.12M to at most 15.36M sampled examples, preserving its fixed variational channel mask and every other setting; compare at the first observed checkpoint with at least 98% train exact.
PREDICT:    Extra compute closes much of the fixed-budget underfit, but validation remains below the unchanged baseline's 64.81%; reaching at least 98% train with validation below baseline refutes this dropout placement as inductive bias rather than merely learning efficiency.
RESULT:     refuted the prediction and retained as a compute-expensive regularizer. At the first observed fit-matched checkpoint (8.192M examples), train was 99.84% and validation 72.08%; validation later peaked at 76.37% at 14.848M, with once-opened audit 74.18%, three-carry 66.33%, and four-digit 62.78%. It improves generalization over baseline when fit matched, but remains below Muon warmdown and shows no preregistered five-point delayed jump after interpolation.

DATE:       2026-08-15
CARD:       neural_gpu_gradient_noise_fit_matched
CHANGE:     Extend the isolated 0.03/(1+step)^0.55 gradient-noise arm from 5.12M to at most 15.36M sampled examples, preserving its seed and every other setting; compare at the first observed checkpoint with at least 98% train exact.
PREDICT:    Decayed noise eventually permits interpolation but does not exceed baseline validation; if it remains underfit at 15.36M, the schedule is an efficiency failure on this seed while the across-seed literature hypothesis remains unresolved.
RESULT:     confirmed for this seed. The first 100% train checkpoint at 8.192M examples had 56.83% validation; validation peaked at 58.42% at 13.312M with audit 55.63%, three-carry 40.05%, and four-digit 39.49%. No delayed rise occurred. Revert this schedule; do not generalize the single-seed result to all gradient-noise schedules.

DATE:       2026-08-15
CARD:       neural_gpu_constant_muon_fit_matched
CHANGE:     Extend constant-LR Muon alone to the 15.36M-example cap under the matched compiled harness, preserving its 0.02 learning rate and all other settings.
PREDICT:    Constant Muon repeatedly approaches interpolation early but cannot remain at 98% train and its validation collapses after the early peak; failure to produce any stable fit-matched checkpoint confirms learning-rate warmdown, rather than checkpoint luck, is necessary.
RESULT:     confirmed. Constant Muon peaked at 95.78% train / 77.47% validation at 1.536M examples, never reached the 98% fit gate, and collapsed to 0.73% train / 1.00% validation by the 15.36M cap. Its validation-selected checkpoint audited at 77.97%, but it is not a stable interpolating solution. Warmdown is necessary.

DATE:       2026-08-15
CARD:       neural_gpu_muon_warmdown_plus_dropout
CHANGE:     Add only the isolated 9% recurrent candidate-state dropout mask to the Muon-warmdown winner after all isolated cards completed; preserve data, architecture, seed, schedule, and evaluation protocol.
PREDICT:    Muon removes dropout's fixed-budget underfit while dropout raises validation beyond the 84.15% Muon-only plateau. Retain only if validation and once-opened audit both exceed Muon-only and carry-heavy metrics do not regress; otherwise the isolated benefits do not compose. A delayed rise counts as grokking only after train remains at least 98%.
RESULT:     strongly confirmed composition, but not grokking. At the 5.12M fixed boundary it reached 100% train / 96.81% validation. Validation-selected step 19k reached 100% train / 97.01% validation / 98.21% audit, with 97.45% three-carry and 97.43% four-digit exact. After interpolation at 2.048M, validation rose gradually by only 1.70 points rather than showing a delayed jump. Retain as final winner.

DATE:       2026-08-15
CARD:       neural_gpu_muon_dropout_diagonal
CHANGE:     Add only fixed left/stay/right channel transport to the Muon-warmdown-plus-dropout winner, preserving the frozen two-digit data, 128 channels, fourteen updates, seed, and optimizer schedule; allow up to 15.36M examples so slower fitting cannot reject it.
PREDICT:    Muon and dropout may make directional transport trainable, but the fixed shift constrains information mixing and will remain below the 97.01% validation winner. Retain for 11-bit scaling only if validation exceeds 97.01%, the once-opened audit exceeds 98.21%, and central/carry-heavy metrics do not regress.
RESULT:     confirmed regression despite unlimited fit. The diagonal arm reached 100% train by 2.048M examples but peaked at only 94.12% validation at 11.776M and 93.92% once-opened audit, with 91.58% three-carry and 90.92% four-digit exact. Exclude diagonal transport from 11-bit scaling.

DATE:       2026-08-15
CARD:       neural_gpu_muon_dropout_11bit_multiplication
CHANGE:     Scale only operand width from seven to eleven binary bits: use 22 output positions and 22 tied recurrent updates with the retained 128-channel Muon-warmdown-plus-dropout architecture. Train on 200,000 deterministic unique unordered pairs; select on 10,000 disjoint pairs and open a separate 10,000-pair audit once. No competition data is generated.
PREDICT:    The winner learns substantial 11-bit multiplication but does not reproduce 98.21% because longer partial-product/carry paths and a 25x larger numeric domain raise algorithmic pressure. Success requires at least 95% monitored-train and 75% validation exact by 25.6M examples; failure to reach train fit is capacity/optimization-inconclusive, while train fit with low validation refutes fixed-width algorithmic scaling.
RESULT:     unclear but strongly transferable. At 25.6M examples the model reached 77.08% full-train / 75.53% validation / 75.46% once-opened audit exact and 97.99% audit bit accuracy. It cleared the validation threshold but not the 95% train-fit requirement, so it demonstrates unseen-pair 11-bit signal without resolving capacity versus optimization or post-fit algorithmic generalization. Audit exact declines from 91.81% for 18-bit products to 53.84% for 22-bit products and from 90.51% at ten active-carry columns to 31.03% at twenty.
### 2026-08-15 — Direct 11-bit squaring on the retained Neural GPU

- **Change:** Replace arbitrary `a,b` multiplication pairs with disjoint-value
  `x,x -> x^2` examples. Keep the 128-channel Muon-warmdown + 9% recurrent-
  dropout model, 22-bit tape, 22 tied updates, optimizer, and seed unchanged.
- **Prediction:** Squaring should substantially exceed the 75.46% multiplication
  audit because commutativity removes operand-order variation and the model only
  needs diagonal partial products. Success is >=95% untouched-x exact; <90% with
  training near 100% indicates poor rule generalization, while low train and
  audit together indicates remaining optimization/capacity limits.
- **Selection rule:** Select checkpoints on the 224-value validation split only;
  open the disjoint 224-value audit once after selection.
- **Result:** The unchanged 128-channel model reached 100% train exact by
  2,000 steps and peaked at **99.11% validation exact** at step 14,000
  (7.168M examples). The owner stopped the run at step 20,000 because the
  learnability question was already answered. The original harness saved only
  on normal completion, so no checkpoint or audit metric exists; do not report
  99.11% as untouched audit accuracy. The result clears the validation gate and
  refutes an immediate width requirement for fixed-width 11-bit squaring, but
  it does not establish length generalization or a stable recurrent algorithm.
### 2026-08-15 — Legal Neural GPU square/reduce composition on Easy E6

```
CARD:       neural_gpu_square_reduce_e6
CHANGE:     replace the failed six-lane four-microstep local grid with two
            explicit learned phases: the successful 128-channel tied ConvGRU
            topology runs 2W updates for squaring, N is re-injected, and a
            separate identically shaped learned cell runs 2W reduction updates;
            retain Muon warmdown and 9% recurrent dropout
PREDICT:    fixed-N E6 may identify a useful composite transition and exceed
            the old chance-scale legal grid, but direct raw-square success will
            not transfer automatically from final labels. Promote to more Easy
            datasets only if mean exact exceeds 5% and T=1 is nonzero on both
            seen-N and OOD profiles; a chance-scale result falsifies this legal
            composition recipe.
```

RESULT:     confirmed — repaired E6 reached 33.09% mean exact (37.8% test,
            28.3% OOD) after 690 updates; the hosted Easy metrics did not
            expose exact T=1 profile counts.

### 2026-08-15 — Frozen square/reduce candidate across E7--E10

```

RESULT:     interrupted by owner after E7 to escalate; unchanged E7 scored
            39.15%, confirming the E6 signal on a second fixed modulus.

### 2026-08-15 — Frozen square/reduce escalation to Medium M6 and Hard H1

```
CARD:       neural_gpu_square_reduce_m6_then_h1
CHANGE:     compute tier/dataset only; use the exact frozen E6/E7 source on M6,
            then one owner-authorized Hard H1 regardless of Medium score
PREDICT:    M6 should exceed historical chance scale but fall below Easy
            because N grows from 8--9 bits to 11 bits while the phase budget is
            unchanged. Hard likely certifies no rung because final-label phase
            identifiability and hidden scale remain unsolved; any nonzero T=1
            profile would nevertheless be meaningful transfer evidence.
```

RESULT:     mixed — M6 scored 0.33%, but the run is optimizer-confounded:
            train exact peaked at 28.1% near step 2,100 before decaying to 0%
            and loss exploding above 20. Owner-authorized exact-source Hard H1
            job 37cedcd2-a172-4fb3-b289-24260777c83b is queued.
CARD:       neural_gpu_square_reduce_easy_fixed_n_sweep
CHANGE:     dataset only; freeze the exact E6 source and evaluate it unchanged
            on fixed-N E7, E8, E9, and E10
PREDICT:    mean exact should remain materially above 5% on every fixed-N rung,
            with variation caused by modulus-specific cycle/support geometry;
            collapse on E8 despite sharing N=287 with E7 would implicate split
            or curriculum sensitivity rather than arithmetic capacity.
```
### 2026-08-15 — Frozen full Easy E1--E10 sweep

```
CARD:       neural_gpu_square_reduce_full_easy_sweep
CHANGE:     dataset only; exact frozen SHA-1
            ff3381c9be98884f0409a3a63fa467cf6be47ab9 on E1--E10
PREDICT:    fixed-small-N E1/E2/E6--E10 should show the strongest signal;
            varying-N E3--E5 should fall sharply because the reducer cannot
            memorize one modulus. E8 should differ from E7 despite shared
            N=287 only if curriculum/split support materially changes learning.
            No source or optimizer changes are allowed during the sweep.
```

RESULT:     confirmed — fixed-N E1/E2/E6--E10 scored 16.25%--41.13%, while
            varying-N E3/E4/E5 scored only 0.54%--0.62%. E7 versus same-N E8
            differed materially (39.15% versus 26.12%), proving curriculum or
            split sensitivity even when modulus is held constant.
### 2026-08-15 — Learned decimal-to-binary interface

```
CARD:       learned_decimal_binary_encoder_11bit
CHANGE:     train a learned four-decimal-token Transformer encoder to predict
            the audited squarer's eleven little-endian input bits on the same
            deterministic 1600/224/224 disjoint-value split
PREDICT:    direct bit supervision should reach at least 99% validation exact
            within 5,000 updates; below 95% with near-perfect train exact means
            this generic fixed-width interface memorizes values and should not
            be composed with the frozen squarer, while 100% validation permits
            opening the audit once and proceeding to soft-versus-hard transfer
```
RESULT:     refuted — 100% train exact but only 16.96% validation and 17.86%
            audit exact; the generic encoder memorized decimal values.
### 2026-08-15 — Exact-square binary Neural GPU reducer

```
CARD:       exact_square_binary_neural_gpu_reducer
CHANGE:     give a 44-update 128-channel tied ConvGRU the exact 22-bit x squared
            plus an immutable padded 11-bit N context at every update, with only
            the final 11-bit remainder supervised
PREDICT:    binary representation and protected N should improve over the prior
            generic binary T=1 tape, but final-label reduction will remain the
            bottleneck; promote to pretrained-square composition only if seen-N
            validation exceeds 95% and unseen-N exceeds 80%, while train fit
            with unseen-N below 20% is a memorization refutation
```
RESULT:     refuted at the registered promotion gate — at 3,000 updates it
            reached only 5.38% train-probe, 4.52% seen-N validation, and 4.92%
            unseen-N exact; generic whole-square recurrence did not organize
            reduction.
### 2026-08-15 — Trace-supervised binary prefix reducer

```
CARD:       trace_supervised_binary_prefix_reducer
CHANGE:     replace whole-square generic reduction with one tied learned
            bit-serial transition over current remainder, immutable N, and the
            next MSB-first square bit; directly supervise transition targets
PREDICT:    the smaller conditional transition should exceed 99% one-step exact
            on seen and unseen N, but 22-step rollout will expose compounding
            errors; promote to final-label training only if both rollout
            profiles exceed 95%, and reject the transition architecture if
            unseen-N one-step exact remains below 90%
```
RESULT:     confirmed — validation-selected one-step exact was 99.60% seen and
            99.385% unseen N; 22-step rollout reached 96.46% seen and 95.49%
            unseen N on 10,000 fresh examples each.
### 2026-08-15 — Final-label fine-tuning of binary prefix reducer

```
CARD:       final_label_binary_prefix_reducer_trace_init
CHANGE:     initialize the successful tied prefix transition from its
            trace-supervised checkpoint, roll it through all 22 square bits
            with straight-through binary states, and train using only final
            remainder bits
PREDICT:    low-rate final-label fine-tuning should retain above 90% seen and
            unseen-N rollout exact if the trace-learned transition is a stable
            causal solution; collapse below 50% means final-label gradients
            destroy the local algorithm and pretraining alone is insufficient
```
RESULT:     refuted — one final-label update cut rollout to 71.2%/66.35%, then
            unrestricted fine-tuning collapsed to chance by step 250.
### 2026-08-15 — Frozen binary squarer plus frozen prefix reducer

```
CARD:       frozen_binary_square_prefix_reduce_composition
CHANGE:     compose the validation-selected 99.55% 11-bit squarer with the
            validation-selected trace-supervised prefix reducer, discretizing
            all 22 square bits before reduction and training no parameters
PREDICT:    composed exact should approximate the product of component success
            rates, around 94% on both seen and unseen N; a gap greater than five
            points below the reducer-with-true-square control indicates that
            the squarer's rare bit errors are adversarial for reduction
```
RESULT:     confirmed — squarer exact was 99.97% on both profiles; true-square
            reducer exact was 96.78% seen and 94.97% unseen N; composed exact
            was 96.75% and 94.94%, showing negligible interface loss.
### 2026-08-15 — Random final-label binary prefix reducer control

```
CARD:       final_label_binary_prefix_reducer_random_init
CHANGE:     use the identical 22-step straight-through prefix architecture and
            final-remainder-only loss, but initialize the transition randomly
            instead of from trace-supervised weights
PREDICT:    it will remain below 10% seen and unseen-N exact after 1,500 updates
            because the terminal loss does not identify locally correct prefix
            states; exceeding 50% would show the architecture alone supplies
            enough bias for legal final-label discovery
```
RESULT:     confirmed failure — random final-label training ended at 0.20%
            seen and 0.20% unseen-N exact after 1,500 updates.
### 2026-08-15 — Soft-state final-label prefix reducer

```
CARD:       final_label_binary_prefix_reducer_soft_state
CHANGE:     from random initialization, keep recurrent remainder bits as sigmoid
            probabilities during training instead of straight-through hard bits;
            retain identical architecture, data, optimizer, final-only loss,
            and hard-state evaluation
PREDICT:    smoother gradients may lower terminal loss faster than the 0.658
            STE control, but train/eval state mismatch will likely keep hard
            rollout below 10%; above 50% on both profiles would justify adding
            progressive sharpening rather than abandoning final-label discovery
```
RESULT:     confirmed failure — soft-state training ended at 0.05% seen and
            0.12% unseen-N exact; smoother recurrence did not repair credit
            assignment.

### 2026-08-15 — Legal binary square-prefix-reduce E5 submission

```
CARD:       legal_binary_square_prefix_reduce_e5
CHANGE:     exact decimal-to-binary representation preprocessing feeding a
            random learned Neural GPU squarer, random tied MSB-first prefix
            reducer, and learned decimal decoder; train only on final labels
PREDICT:    E5 T=1 will remain below 10% because our controlled random-init
            prefix experiments showed terminal supervision does not discover
            the reducer transition, but a valid hosted run distinguishes an
            implementation failure from the known credit-assignment failure
```

### 2026-08-15 — Exact final-binary supervision on competition E5

```
CARD:       e5_exact_final_binary_loss
CHANGE:     replace the learned decimal decoder and decimal token CE with exact
            binary output conversion plus BCE on the 11 final remainder bits
PREDICT:    train exact and T=1 evaluation will exceed the 3.1%/sub-1% hosted
            control if decoder entanglement mattered, but remain below 20% if
            intermediate reducer credit assignment is still dominant
```
RESULT:     refuted — 357 updates ended at 0% seen, OOD, and every T-profile;
            final binary BCE remained 0.676, so removing decoder entanglement
            did not repair terminal credit assignment in the Easy budget.

### 2026-08-15 — Direct square supervision on competition E5

```
CARD:       e5_exact_final_binary_plus_square_aux
CHANGE:     add BCE supervision on the first macrostep's 22 square bits to the
            otherwise identical exact-final-binary competition-shaped model
PREDICT:    square-bit exact will rise sharply; if modular exact remains below
            20%, the reducer rather than squarer is the decisive bottleneck
```
RESULT:     refuted at the 60-second gate — direct square supervision reached
            only about 0% whole-square exact and 65–68% square-bit accuracy by
            316 updates, while modular exact remained 0%; the combined model
            is too slow to learn even its squarer within the Easy budget.

### 2026-08-15 — Fit-matched square-supervised combined model

```
CARD:       e5_square_aux_15min_fit_diagnostic
CHANGE:     extend only the training budget from 60 to 900 seconds for the
            square-supervised competition-shaped model
PREDICT:    square-bit and whole-square accuracy will improve materially before
            final modular exact; square exact above 90% with modular exact below
            10% selects reduction, while square exact below 50% selects the
            combined model's squaring/throughput path first
```
RESULT:     partially confirmed — after 5,477 updates the directly supervised
            squarer reached roughly 93–96% bit accuracy and intermittently 50%
            whole-square exact on training minibatches, while final remainder
            bits stayed near chance and every seen/OOD T rung remained 0%; fix
            reduction first, but restore the proven Muon squarer recipe rather
            than treating this AdamW joint-training result as squaring solved.

### 2026-08-15 — Muon square-supervised E5 fit run

```
CARD:       e5_square_aux_muon_30min
CHANGE:     replace AdamW on matrix parameters with the successful flattened
            Muon recipe; retain identical E5 data, model, losses, and seed
PREDICT:    directly supervised whole-square exact will exceed the AdamW run's
            50% minibatch peak and approach 100%; if final modular exact remains
            near zero after square exact exceeds 95%, reduction is isolated
```
RESULT:     invalidated at 2,000 updates — Muon was mistakenly applied to the
            Transformer reducer as well as the Neural GPU squarer; stopped
            before using the result for a component decision.

### 2026-08-15 — E5 square-only Muon saturation

```
CARD:       e5_square_only_muon_saturation
CHANGE:     train only the Neural GPU squarer with direct square-bit loss and
            flattened Muon, using x values drawn from the exact E5 train split
PREDICT:    whole-square minibatch exact will reach at least 99% by 10,000
            updates, reproducing the dedicated held-out-x squarer result on the
            competition distribution; failure below 90% refutes transfer
```
RESULT:     invalidated at 4,500 updates — the wall-clock schedule held Muon at
            0.02 and the early square solution collapsed; replaced with the
            audited 1,000-to-5,000 step warmdown before judging transfer.

### 2026-08-15 — E5 square-only audited Muon warmdown

```
CARD:       e5_square_only_muon_step_warmdown
CHANGE:     replace the invalid wall-clock decay with the audited schedule:
            0.02 through step 1,000, cosine to 0.002 at 5,000, then clamp
PREDICT:    square exact will recover during warmdown and exceed 99% by 10,000
            updates on E5 x values; sustained failure below 90% would show the
            E5 x distribution differs materially from the prior square split
```
RESULT:     refuted under unmatched sample count — 10,000 x 32 examples reached
            68.05% train, 72.41% E5 test, and 76.41% OOD exact dropout-free;
            this was 16x fewer samples than the audited batch-512 squarer.

### 2026-08-15 — E5 compute-matched square-only Muon

```
CARD:       e5_square_only_muon_batch512_10k
CHANGE:     increase only square-only batch size from 32 to the audited 512,
            retaining 10,000 steps, E5 data, topology, dropout, and warmdown
PREDICT:    dropout-free exact will exceed 99% on E5 train and test x values;
            failure below 95% would refute transfer of the prior squarer result
            to the exact E5 x distribution
```
RESULT:     refuted because topology was not actually matched — dropout-free
            exact reached 82.78% train, 83.16% E5 test, and 87.66% OOD; audit
            found role vectors contaminating padded workspace cells unlike the
            proven squarer.

### 2026-08-15 — Exact proven squarer topology on E5

```
CARD:       e5_exact_proven_squarer_batch512_10k
CHANGE:     replace global row-role contamination with the proven topology's
            two operand-row markers and exactly zero unused workspace cells
PREDICT:    with identical batch-512 Muon warmdown, dropout-free E5 test exact
            will exceed 99%; failure below 95% means another reproduction
            mismatch remains
```
RESULT:     improved but below gate — dropout-free exact reached 90.63% train,
            90.67% E5 test, and 92.64% OOD; continue the validation-selected
            checkpoint at the post-warmdown Muon rate.

### 2026-08-15 — E5 supervised squarer low-rate continuation

```
CARD:       e5_squarer_muon_low_lr_continuation
CHANGE:     continue the 10,000-step checkpoint with Muon fixed at its clamped
            0.002 rate; retain exact E5 sampling, batch 512, dropout, and loss
PREDICT:    dropout-free train/test exact will exceed 99% after 10,000 more
            updates; a plateau below 95% indicates sampling/architecture rather
            than insufficient optimization time
```
RESULT:     refuted — after 20,000 total updates exact plateaued at 91.01%
            train, 91.19% E5 test, and 92.64% OOD; added time at low Muon rate
            did not close the gap.

### 2026-08-15 — E5 squarer no-dropout exactness continuation

```
CARD:       e5_squarer_disable_dropout_continuation
CHANGE:     disable 9% recurrent dropout only during continuation from the
            20,000-step checkpoint; retain E5 samples and low Muon rate
PREDICT:    training exact will exceed 99% and test exact improve above 95% if
            dropout is the ceiling; no movement means the remaining mismatch
            is data coverage or seed-specific optimization
```
### 2026-08-15 — Long E5 final-only emergence run

```
CARD:       e5_final_only_hidden_square_emergence_30min
CHANGE:     remove all square supervision from a random exact-topology combined
            model; retain only final 11-bit remainder BCE and report square
            metrics without including them in the loss
PREDICT:    hidden square exact and modular exact will remain below 10% after
            30 minutes because the final modular target does not identify the
            unreduced square; exceeding 50% square exact would show emergent
            factorization despite reducer credit assignment
```
### 2026-08-15 — Supervised versus final-only representation audit

```
CARD:       e5_square_representation_checkpoint_audit
CHANGE:     capture full-model checkpoints at matched updates and compare
            supervised-square versus final-only hidden states without changing
            either training objective
PREDICT:    supervised checkpoints will progressively linearly decode square
            bits and multiplication intermediates; final-only checkpoints will
            retain x but show no causal or linearly decodable literal square,
            localizing failure before the explicit square/reducer interface
```
RESULT:     confirmed with a stronger two-sided failure — at step 10,000 the
            supervised hidden probe decoded square at 89.51% exact while the
            final-only probe reached only 13.60%, despite 99.87% exact x
            decoding. Causally patching perfect square bits into the final-only
            reducer yielded 0/400 correct T=1 examples at every checkpoint;
            both the square transformation and reducer transition fail under
            terminal-only training.
### 2026-08-15 — Exact-square final-only reducer isolation

```
CARD:       exact_square_prefix_reducer_e1_vs_e5
CHANGE:     bypass the squarer with exact square bits and train the unchanged
            prefix reducer from random initialization using only final T=1
            remainder labels; compare fixed-N E1 against varying-N E5
PREDICT:    E1 will learn a nontrivial fixed-modulus shortcut while E5 remains
            near zero; if E5 also learns, the prior failure was upstream square
            noise, while failure on both isolates the prefix reducer objective
```
### 2026-08-15 — Squarer trained through exact differentiable modulo

```
CARD:       exact_mod_oracle_square_identifiability_e1_e5
CHANGE:     remove the learned reducer and train the exact Neural GPU squarer
            only through an exact differentiable residue-distribution layer
            and final T=1 modular labels; compare fixed-N E1 and varying-N E5
PREDICT:    E1 can reach modular accuracy without literal square accuracy due
            to residue-class ambiguity; varying-N E5 should constrain the
            latent integer more strongly and produce more square information if
            reducer failure was masking an otherwise learnable squarer
```
RESULT:     reducer masking was rejected. E1 memorized sampled training
            residues (100% modular exact) but ended at 2% held-out modular and
            0% held-out square exact. E5 ended at 44.31% sampled-train modular,
            5.00% held-out modular, and 3.75% held-out square exact. At E5 step
            5,000, the final hidden state still decoded x at 82.25% exact but
            decoded x^2 at only 5.44% exact. Perfect modulo therefore does not
            rescue final-label-only learning of the squaring transition.

### 2026-08-15 — Exact-modulo squarer identifiability ablations

```
CARD:       exact_mod_pretrained_control
CHANGE:     initialize the same N-blind binary squarer from the audited direct
            square checkpoint and evaluate it through exact modulo
PREDICT:    held-out modular exact will track direct square exact near 99%,
            proving the interface and oracle evaluation are not the blocker

CARD:       exact_mod_pretrained_finetune
CHANGE:     continue the perfect pretrained squarer using final modular labels
            through exact modulo, with no square supervision
PREDICT:    square accuracy will remain near 100% if the correct circuit is a
            stable basin; collapse would show the modular objective conflicts
            with rather than merely fails to discover literal squaring

CARD:       exact_mod_informative_curriculum
CHANGE:     sample only x^2 < N rows for the first 1,000 updates, then restore
            the full E5 distribution
PREDICT:    square exact will rise early but decay or plateau after the switch
            because only 4.1% of E5 T=1 rows provide this identifying signal

CARD:       exact_mod_paired_multin
CHANGE:     construct every batch from same-x pairs with distinct N
PREDICT:    held-out square/modular exact will exceed the random-batch baseline
            if simultaneous multi-modulus constraints improve credit assignment

CARD:       exact_mod_stochastic_consistency
CHANGE:     add agreement between two dropout views of the N-blind squarer
PREDICT:    it may preserve x features but will not materially identify x^2,
            because consistency removes noise without choosing the right function

CARD:       exact_mod_digitized_bottleneck
CHANGE:     anneal Bernoulli temperature and penalize non-binary probabilities
PREDICT:    direct square exact may improve if diffuse bit distributions caused
            the failure; otherwise it will harden an incorrect residue shortcut
```
RESULT:     pretrained control confirmed at 100% square/modular exact on all
            400 held-out E5 T=1 rows. Final-only fine-tuning drifted to 76.5%
            at step 3,000 and recovered to 94.5% under LR cooldown, so the
            correct basin is reachable but not intrinsically preserved.
RESULT:     informative curriculum confirmed a material but incomplete gain:
            28.25% held-out square and 28.50% modular exact versus baseline
            3.75%/5.00%; the 65-row privileged phase changes the basin.
RESULT:     paired multi-N refuted at 0% square / 0.5% modular; oversampling
            repeated-x rows and simultaneous congruences did not identify x^2.
RESULT:     stochastic consistency refuted at 0% square / 1.0% modular; it made
            the shortcut stable rather than selecting multiplication.
RESULT:     digitization refuted at 0% square / 0.5% modular; sharpening bit
            probabilities hardened the wrong latent function.

### 2026-08-15 — Legal no-wrap square-anchor curriculum

```
CARD:       legal_nowrap_anchor_square_reduce
CHANGE:     on T=1 rows whose input bit lengths guarantee no modular wrap,
            use the evaluator-provided final bits to supervise the N-blind
            squarer directly for the first 20% of wall time, then train the
            complete learned reducer while retaining a 0.5 square anchor
PREDICT:    fixed-N E10 should remain within 10 points of the 41.13% anchor;
            varying-N E5 should exceed its 0.54% hosted baseline if the legal
            anchor recruits transferable squaring, but may remain below 10%
            because the learned reducer still receives terminal supervision
```
RESULT:     invalid for E10 and negative for E5 — E10 scored 8.92%, but N=403
            made the no-wrap mask empty and the first 20% branch returned zero
            loss, so that run discarded one fifth of its training budget. E5
            scored 0.50% after only 825 updates, far below the roughly 10k
            updates required by the supervised squarer.

```
CARD:       legal_nowrap_square_only_throughput
CHANGE:     during the first 20% no-wrap curriculum, skip the unused learned
            reduction phase and train only one square macrostep; retain the
            identical full model, anchor, and loss for the remaining 80%
PREDICT:    E5 should process materially more than 825 updates and exceed 0.50%
            if curriculum throughput was binding; staying at chance despite a
            large update gain refutes compute as the main Easy bottleneck
```

```
CARD:       legal_nblind_square_nowrap_throughput
CHANGE:     remove N from the square phase entirely, preserving immutable N
            injection immediately before the learned reducer; combine with the
            already registered square-only no-wrap throughput curriculum
PREDICT:    varying-N E5 should exceed 0.50% if N-specific co-adaptation was
            suppressing the shared square function; fixed-N may regress because
            the previous 8.92% signal could exploit N-visible shortcuts
```

### 2026-08-16 — Matched binary processor: exact-square reduction arm

```
CARD:       binary_workstate_exact_square_reduction
CHANGE:     initialize the shared binary work-state processor from exact x^2
            bits plus immutable N bits; use only final residue-bit labels
PREDICT:    seen-N validation should exceed 80% and unseen-N exact should
            exceed the prior 18.69% reduction diagnostic if immutable binary
            context and the Muon/dropout processor can learn generic reduction;
            unseen-N below 25% falsifies reduction capacity as currently built
```
RESULT:     refuted — validation-selected exact was 5.92% on unseen-x/seen-N,
            3.56% on seen-x/unseen-N, and 6.80% on unseen-x/unseen-N; even
            train exact was only 4.04%, so this processor/optimizer did not fit
            exact-square reduction at the matched 5.12M-example budget.

### 2026-08-16 — Matched binary processor: fused x,N arm

```
CARD:       binary_workstate_fused_t1
CHANGE:     replace only the exact-square source bits with x bits, keeping the
            split, processor, optimizer, update count, budget, and final
            residue labels identical
PREDICT:    fused unseen-N exact will trail reduction-only early; exceeding 5%
            is material evidence of joint transition learning, while chance
            performance alongside a successful reducer localizes the failure
            to multiplication/reduction credit assignment
```
RESULT:     refuted — validation-selected fused exact reached only 2.04% on
            unseen-x/seen-N, 2.06% on seen-x/unseen-N, and 1.74% on the joint
            unseen-x/unseen-N audit. Exact-square input improved the same
            processor but did not solve it, and both arms collapsed during the
            same Muon phase.

### 2026-08-16 — Exact-square reduction with AdamW

```
CARD:       binary_workstate_exact_square_adamw
CHANGE:     replace flattened-convolution Muon plus scalar AdamW with one
            AdamW optimizer over all parameters; everything else is identical
PREDICT:    the step-1,500 collapse should disappear and validation should stay
            above 5.92%; exceeding 25% unseen-N exact would identify optimizer
            instability as the dominant prior failure, while stable low fit
            would falsify Muon collapse as the main bottleneck
```
RESULT:     partially confirmed — AdamW eliminated collapse and improved exact
            to 14.56% unseen-x/seen-N, 10.84% seen-x/unseen-N, and 15.00%
            joint unseen. It missed the 25% gate and fit only 11.86% of train,
            so Muon instability was important but not the dominant bottleneck.

### 2026-08-16 — Binary reduction width 192

```
CARD:       binary_workstate_adamw_width192
CHANGE:     increase channels from 128 to 192; retain constant AdamW 3e-4 and
            every other matched exact-square reduction setting
PREDICT:    examples/second will fall by roughly 40-60%, but 10k-step exact
            should exceed 14.56% validation and 15.00% joint unseen if capacity
            limits the stable 128-channel curve; no accuracy gain refutes width
            as the immediate bottleneck
```
RESULT:     partially confirmed — width 192 improved train/validation/
            seen-x-unseen-N/joint-unseen exact to 15.06/18.10/14.40/18.30%,
            versus 11.86/14.56/10.84/15.00% at width 128. It took 1,591.7
            seconds versus 680.1 seconds (2.34x slower), and briefly collapsed
            at step 8,500 before recovering. Width improves the 10k-step
            endpoint, but not wall-clock learning speed or the 25% gate.

### 2026-08-16 — Binary reduction AdamW warmup-cosine

```
CARD:       binary_workstate_adamw_warmup_cosine
CHANGE:     at 128 channels, replace constant AdamW 3e-4 with 500-step warmup
            to 1e-3 followed by cosine decay to 1e-4 at step 10k
PREDICT:    faster early fit without Muon's collapse should exceed 14.56%
            validation at matched steps; a late decline or lower final score
            refutes this aggressive schedule
```
RESULT:     refuted — train/validation/seen-x-unseen-N/joint-unseen exact was
            9.86/11.96/8.30/12.20%. The schedule learned faster during its
            500-step warmup, but failed to compound that lead and finished
            below constant AdamW 3e-4 on every exact metric.

### 2026-08-16 — Binary reduction AdamW warmup-inverse-sqrt

```
CARD:       binary_workstate_adamw_inverse_sqrt
CHANGE:     at 128 channels, use 500-step warmup to AdamW 1e-3 followed by
            inverse-square-root decay, ending near 2.24e-4 at step 10k
PREDICT:    sustained learning should beat warmup-cosine late and exceed the
            14.56% constant-LR anchor; instability or no improvement falsifies
            schedule shape as the next limiting factor
```
RESULT:     refuted — train/validation/seen-x-unseen-N/joint-unseen exact was
            11.24/13.40/9.92/13.94%. It beat cosine late, but not constant
            AdamW 3e-4. Schedule shape is not the immediate limiting factor at
            this budget.

### 2026-08-16 — Full fused T=1 at fixed N=403

```
CARD:       binary_workstate_fused_fixed_n403
CHANGE:     give the full binary work-state processor x bits rather than exact
            square bits and hold N fixed at 403; use a deterministic 70/15/15
            split of all x in [0, 402], width 128, 44 updates, 9% dropout, and
            constant AdamW 3e-4 for 10,000 steps
PREDICT:    train exact should exceed 95% because the finite fixed-N mapping is
            learnable or memorizable; held-out-x validation above 25% is
            material function-learning evidence and above 80% is a strong
            fixed-N T=1 pass. Train above 95% with audit below 25% diagnoses
            memorization; train below 80% diagnoses fused optimization or
            capacity failure even after modulus variation is removed.
```
RESULT:     memorization branch confirmed — train exact reached 95.39% at
            step 1,000 and 100% at step 1,250, then stayed perfect through
            step 10,000. Held-out-x validation never exceeded 1/60 (1.67%),
            already attained at step 1, and ended at 0/60. Validation selected
            step 1; its untouched audit was 1/61 (1.64%), also chance-scale.
            Fixed N removes the optimization failure but permits a lookup
            solution rather than recruiting modular squaring.

### 2026-08-16 — Full fused varying-N with AdamW

```
CARD:       binary_workstate_fused_varying_n_adamw
CHANGE:     in the matched fused x,N arm, replace Muon warmdown plus scalar
            AdamW with constant AdamW 3e-4 over all parameters; retain the
            seed-74 x/N splits, width 128, 44 updates, 9% dropout, and 5.12M
            example budget
PREDICT:    the post-step-1,000 collapse should disappear and validation exact
            should exceed the Muon arm's 2.04%; above 5% is material fused
            signal and above 10% is a strong lead. Stable train below 5% or
            joint-unseen below 2% refutes optimizer collapse as the main fused
            bottleneck.
```
RESULT:     confirmed — AdamW eliminated the Muon collapse and raised
            train/validation/seen-x-unseen-N/joint-unseen exact from
            2.41/2.04/2.06/1.74% to 8.75/7.60/6.94/7.92%. It cleared the 5%
            material-signal gate but missed 10%; stable low train fit leaves
            full-transition underfitting as the dominant bottleneck.

### 2026-08-16 — Full fused varying-N AdamW at 20k steps

```
CARD:       binary_workstate_fused_varying_n_adamw_20k
CHANGE:     extend only the matched fused AdamW budget from 10,000 to 20,000
            steps (5.12M to 10.24M examples); restart deterministically and
            retain all architecture, split, optimizer, seed, and evaluation
            settings
PREDICT:    validation and joint-unseen exact should exceed 10% if ordinary
            optimization budget is still material. Validation above 12% is
            strong compute-limited evidence; below 9.1% (less than +1.5 points
            over 7.60%) indicates practical saturation and argues for an
            architectural rather than longer-training change.
```
RESULT:     unclear — the 20k run improved validation from the prior 7.60% to
            9.62% (+2.02 points) and unseen-N audit to 10.62%, so it cleared
            the +1.5-point practical-gain boundary. Validation and joint-unseen
            (9.16%) both narrowly missed 10%, and the rerun's first 10k differed
            materially from the anchor despite the same seed, making small
            endpoint gains trajectory-sensitive rather than a clean scaling law.

### 2026-08-16 — Full fused varying-N width 256

```
CARD:       binary_workstate_fused_varying_n_width256
CHANGE:     increase only hidden channels from 128 to 256, approximately
            quadrupling recurrent convolution parameters; retain constant
            AdamW 3e-4, seed-74 splits, 44 updates, 9% dropout, and the 10,000
            step / 5.12M-example budget
PREDICT:    train exact should exceed 12% and validation plus joint-unseen
            should exceed 10% if raw representational capacity is the dominant
            10k-step bottleneck. Validation below 9.6% or train below 10%
            refutes a large capacity increase as an efficient immediate fix.
```
RESULT:     confirmed — width 256 raised selected train/validation/
            seen-x-unseen-N/joint-unseen exact to 15.38/13.74/11.22/13.56%,
            clearing every preregistered capacity gate versus width 128's
            8.75/7.60/6.94/7.92%. It reached the narrow 20k validation region
            by 6k-7.5k steps but took 2,171.5 seconds total, so capacity improves
            step efficiency much more than wall-clock efficiency.

### 2026-08-16 — Tuned flattened-Muon LR screen

```
CARD:       binary_workstate_fused_muon_lr001
CHANGE:     at width 128, replace constant AdamW with flattened-convolution
            Muon at lr 0.001, momentum 0.95, weight decay 0.1, and 250-step
            warmup; screen for 3,000 steps using validation only
PREDICT:    stable official-default scaling should avoid the old lr-0.02
            collapse but may trail AdamW's 2.98% validation at step 3,000;
            above 3.5% is a promotion signal
```
RESULT:     refuted — the best checkpoint was step 1,500 at 0.15% train and
            0.50% validation exact; the final step was 0.50% train and 0.24%
            validation. The run was stable but drastically underpowered versus
            AdamW, so lr 0.001 is rejected without opening either audit set.

```
CARD:       binary_workstate_fused_muon_lr003
CHANGE:     change only tuned Muon peak lr from 0.001 to 0.003
PREDICT:    faster matrix learning should exceed 4.0% validation at step 3,000
            without collapse; below the 0.001 arm refutes the higher scale
```
RESULT:     confirmed — the step-3,000 checkpoint reached 4.68% train and
            4.50% validation exact, versus AdamW's matched 2.46%/2.98% and the
            lr-0.001 arm's best 0.15%/0.50%. This is a material per-step gain;
            audit sets remain unopened until the screen winner is promoted.

```
CARD:       binary_workstate_fused_muon_lr006
CHANGE:     change only tuned Muon peak lr from 0.003 to 0.006
PREDICT:    this approximates RMS-matched scaling and should lead the screen
            above 5% if Muon's preconditioning fits the recurrent convolutions;
            a sharp train/validation drop diagnoses overshoot
```
RESULT:     confirmed — the step-3,000 checkpoint reached 6.16% train and
            6.24% validation exact, leading lr 0.003 by 1.74 validation points
            and AdamW by 3.26 points. Training remained stable, so lr 0.006 is
            promoted to the full budget; audit sets remained unopened.

### 2026-08-16 — Tuned Muon full-budget promotion

```
CARD:       binary_workstate_fused_tuned_muon_full
CHANGE:     extend the winning width-128 Muon lr-0.006 configuration from
            3,000 to 10,000 steps and open the two audit sets once at the
            validation-selected checkpoint; all other settings remain fixed
PREDICT:    selected validation should exceed AdamW's 7.60% by at least 2
            points and both unseen-N audits should exceed the AdamW anchors
            (6.94% and 7.92%). Validation above 12% would show that Muon's
            early step-efficiency gain persists rather than merely arriving
            sooner at the same plateau.
```
RESULT:     confirmed — the validation-best final checkpoint reached
            16.84/18.14/14.60/18.40% train/validation/seen-x-unseen-N/
            joint-unseen exact in 681.0 seconds. It clears every gate and
            improves all splits over AdamW by 8--11 points. A sharp step-6,000
            dip recovered, so constant lr 0.006 is effective but noisy.

### 2026-08-16 — Width 256 plus tuned Muon factorial combination

```
CARD:       binary_workstate_fused_width256_tuned_muon
CHANGE:     combine exactly the two independently successful factors: hidden
            width 256 and flattened-convolution Muon lr 0.006, momentum 0.95,
            weight decay 0.1, 250-step warmup; retain the same data, seed,
            44 tied updates, dropout, batch, final-label loss, and 10k budget
PREDICT:    if capacity and optimizer preconditioning are complementary,
            selected train and validation exact should both exceed 22%, with
            both unseen-N audits above 18%. Validation below the better
            isolated arm's 18.14% would show a harmful interaction; exceeding
            27% would demonstrate a super-additive practical combination.
```
RESULT:     confirmed — the validation-best final checkpoint reached
            22.09/22.84/18.38/22.50% train/validation/seen-x-unseen-N/
            joint-unseen exact. It clears all complementarity gates, including
            the first unseen-N gate by 0.38 points, but misses the 27%
            super-additivity threshold. Runtime was 2,055.9 seconds, so the
            combination wins per step and endpoint while width-128 Muon remains
            the wall-clock-efficiency winner.

### 2026-08-17 — Frozen square representation probe

```
CARD:       fused_width256_tuned_muon_square_probe
CHANGE:     freeze the validation-selected width-256 tuned-Muon fused
            checkpoint and train only linear readouts from its final work tape
            to literal 22-bit x squared; include a shared local readout and an
            11-bit x-decoding control, with no processor or residue-head update
PREDICT:    global whole-square exact will remain below 25% while x exact
            exceeds 75%, showing that the 22.84% residue improvement does not
            require a linearly explicit square. Square exact above 80% refutes
            this diagnosis and localizes the remaining failure to reduction;
            25--80% is evidence of a partial square representation, not a gate.
```
RESULT:     confirmed — independently selected global-square exact was 3.10%
            on unseen-x/seen-N validation and 3.02% on the joint-unseen audit,
            while the x control reached 84.82% and 78.50%. The local shared
            square readout reached only 1.22% and 0.84%. The final work tape
            preserves x but does not expose a linearly explicit literal square;
            this fires the preregistered kill for further fused width/optimizer
            tuning as the main line. This is correlational and does not exclude
            a nonlinear or transient square code.

### 2026-08-17 — H13 bit-serial prefix-of-x residue state

```
CARD:       binary_prefix_residue_h13
CHANGE:     keep width 256, one tied 3x3 ConvGRU, 44 total updates, dropout,
            tuned Muon, data, seed, batch, and final-only residue BCE from the
            branch-best fused card; replace the all-at-once x tape with an
            MSB-to-LSB schedule of one prompt-visible x bit per four updates
PREDICT:    if all-at-once credit assignment is the main architectural
            bottleneck, selected unseen-x/seen-N exact will exceed 25% and both
            unseen-N audits will exceed 20%. Validation below 15% refutes this
            specific bit schedule; 15--25% is inconclusive. Prefix readouts are
            post-selection diagnostics only and must not affect checkpointing.
```

RESULT:     refuted — validation selected step 8,000 at 16.88/6.14/4.76/5.58%
            train/validation/seen-x unseen-N/joint-unseen exact, versus the
            all-at-once reference's 22.09/22.84/18.38/22.50%. By step 10,000
            train reached 20.50% while validation fell to 5.74%, so serializing
            x did not prevent shortcut fitting. The shared final readout was no
            better than an always-zero predictor after the first prefix and did
            not trace a stable prefix residue. The <15% kill fires.

### 2026-08-17 — Frozen H13 per-prefix state probe

```
CARD:       binary_prefix_residue_h13_state_probe
CHANGE:     freeze H13's validation-selected checkpoint and train independent
            global linear readouts from its work+scratch lanes after each input
            bit to the corresponding prefix residue; include a prefix-value
            control, with no processor or final residue-head updates
PREDICT:    final-prefix residue exact will remain below 15% and no steps 3--10
            will expose residue above 50%, while prefix value will exceed 75%
            on most steps. A final residue above 25% or a contiguous >50%
            intermediate band would instead license a readout/curriculum repair.
```

RESULT:     mixed and curriculum-licensing — final-prefix residue remained at
            6.32% validation and 6.06% joint unseen, confirming that the full
            transition is absent. However residue was linearly exact through
            five bits, reached 99.38/85.50% at six bits and 70.32/44.38% at
            seven bits on validation/joint unseen. Prefix value remained
            100% through seven bits and 97.02% joint unseen at eight bits. The
            predicted lack of a contiguous >50% band is refuted; the cliff
            localizes to first wrapping/longer depth and licenses length
            curriculum rather than another width or optimizer card.

### 2026-08-17 — H13 significant-bit length curriculum

```
CARD:       binary_prefix_residue_h13_length_curriculum
CHANGE:     keep H13's width, cell, dropout, optimizer, seed, and final-only
            BCE; consume only prompt-visible significant x bits and admit
            provided training rows by maximum x bit length 4,5,...,11, leaving
            the final 4,500 steps on the full distribution
PREDICT:    because frozen H13 states expose residue at 99.38/85.50% through
            six bits, direct final-label training on real short rows should
            preserve that basin while moving the wrap frontier. Promotion
            requires >22.84% validation and both unseen-N audits above their
            18.38/22.50% baselines; >25% validation is strong. Below 10%
            validation refutes this curriculum schedule.
RESULT:     Refuted. Validation selected step 9,000 at 22.56/5.84/3.82/4.32%
            train/validation/seen-x-unseen-N/joint-unseen exact. Full-length
            11-bit exact was 1.64/0.47/0.28/0.09%. The curriculum restored
            training fit but did not move the wrap/generalization frontier.
```

### 2026-08-17 — Width-256 fused binary work-state hosted E5

CARD:       binary_workstate_fused_width256_tuned_muon_e5
CHANGE:     Translate the locally strongest width-256, 44-update fused binary
            work-state model with tuned Muon into the evaluator interface and
            run it on varying-N E5.
PREDICT:    The model will validate and exceed the prior varying-N E5 result of
            0.54%, because exact binary representation and the locally improved
            optimizer/capacity pair produce nontrivial unseen-N T=1 learning;
            less than 1% or an evaluation failure falsifies promotion.
### 2026-08-17 — Width-128 fused hosted-throughput control

CARD:       binary_workstate_fused_width128_tuned_muon_e5
CHANGE:     Reduce only hidden channels from 256 to 128 in the corrected fused
            binary work-state E5 submission.
PREDICT:    Easy completed updates increase by at least 2.5x over 169 and final
            train exact becomes nonzero; fewer than 423 updates or zero final
            train exact refutes width as the primary hosted bottleneck.
### 2026-08-17 — Eleven-update fused hosted-throughput control

CARD:       binary_workstate_fused_width128_updates11_e5
CHANGE:     Reduce only tied recurrent updates from 44 to 11 in the width-128
            fused E5 model.
PREDICT:    Easy completes at least 1,200 updates and obtains nonzero final
            train exact plus mean evaluation above the width-128 card's 0.08%;
            failure of either condition means saved steps do not compensate
            for lost transport depth.
### 2026-08-17 — Six-update fused hosted-throughput control

CARD:       binary_workstate_fused_width128_updates6_e5
CHANGE:     Reduce only tied recurrent updates from 11 to 6.
PREDICT:    Easy completes at least 1,300 updates and improves final train exact
            beyond 2.3%; evaluation above 0.21% would retain six updates,
            while lower evaluation despite faster fitting identifies lost
            message-passing depth.
### 2026-08-17 — Fused update/reset gate throughput control

CARD:       binary_workstate_fused_width128_updates11_fused_gates_e5
CHANGE:     Fuse only the update/reset convolutions into one 2C-output call in
            the retained width-128, 11-update model.
PREDICT:    Completed Easy updates improve at least 15% beyond 936 without
            reducing final train exact below 2.3%; evaluation at or above
            0.21% retains the implementation.
### 2026-08-17 — Batch-256 fused hosted-throughput control

CARD:       binary_workstate_fused_width128_updates11_batch256_e5
CHANGE:     Reduce only training batch from 512 to 256.
PREDICT:    Optimizer updates increase at least 40% beyond 936 while processed
            examples remain at least 70% of 479,232; retain only if evaluation
            exceeds 0.21% or loss improves materially without losing OOD-N.
### 2026-08-17 — Batch-128 fused hosted-throughput control

CARD:       binary_workstate_fused_width128_updates11_batch128_e5
CHANGE:     Reduce only training batch from 256 to 128.
PREDICT:    Optimizer updates exceed 2,000 but processed examples fall below
            batch 256; retain only if mean evaluation exceeds 0.54% and both
            test and OOD-N remain nonzero.
### 2026-08-17 — Warmup-100 fused Easy control

CARD:       binary_workstate_fused_width128_updates11_batch256_warmup100_e5
CHANGE:     Reduce only Muon warmup from 250 to 100 optimizer updates.
PREDICT:    Final train exact exceeds 2.7% and mean evaluation exceeds 0.54%
            because less of the short run is spent below the tuned LR; rising
            loss or loss of either split refutes the shorter warmup.
## 2026-08-17 — binary global-attention transition (T=1)

- **Question:** Is the fused T=1 ceiling caused by the tied 3x3 ConvGRU forcing
  modulus-wide comparison and subtraction through slow local transport?
- **One change:** Replace the 44-update local ConvGRU transition with a
  weight-tied global self-attention transition over immutable x bits, immutable
  N bits, and a binary workspace. Keep the same deterministic 11-bit varying-N
  rows, split seed 74, final-residue-bit supervision, validation selection, and
  tuned Muon family.
- **Prediction:** At matched 5.12M examples, global attention will exceed the
  current fused width-256 validation exact score of 22.84% and the unseen-N
  audits of 18.38% / 22.50%. If validation remains below 20%, global transport
  is not the main bottleneck and this branch is rejected.
- **Gate:** Promote only if validation is above 25% and both untouched unseen-N
  audits are above 20%. Audit does not select checkpoints.

## 2026-08-17 — binary global-attention width 256 (T=1)

- **Question:** Did the global-attention arm regress because it had only 200,577
  parameters, rather than because attention is a poor transition bias?
- **One change:** Increase width from 128 to 256. Keep eight tied updates, four
  heads, the same 5.12M examples, data, seed, loss, Muon settings, validation
  selection, and untouched audits.
- **Prediction:** Train exact will exceed 12% and validation will exceed 10%.
  Below 8% validation rejects attention capacity as a productive direction;
  above 20% keeps the architecture alive for optimizer testing.

## 2026-08-17 — wide-kernel recurrent transport (T=1)

- **Question:** Can the successful ConvGRU bias retain its learnability while
  obtaining modulus-wide transport in the 11 microsteps that fit the hosted
  budget?
- **One architecture change:** Replace each 3x3 gate convolution with a 3x7
  convolution. Use 11 rather than 44 updates as the intended matched-throughput
  operating point. Keep width 256, data, seed, final-bit loss, tuned Muon, and
  selection policy fixed.
- **Prediction:** The larger receptive field will reach at least 15% validation
  exact at 5.12M examples despite using one quarter as many updates. Above 20%
  retains it as a faster transition; below 10% rejects the trade.

## 2026-08-17 — parameter-matched wide-kernel transition (T=1)

- **Trigger:** The width-256 / 3x7 model reached 27.52% train exact but only
  5.08% held-out-x validation at step 3,000, indicating fast memorization.
- **One change:** Reduce channels from 256 to 168. This changes persistent state
  from about 4.13M to 1.78M parameters, matching the original width-256 / 3x3
  ConvGRU while keeping the 3x7 transport, 11 updates, optimizer, data, and
  5.12M-example budget fixed.
- **Prediction:** The train-validation gap at step 3,000 will fall by at least
  half and final validation will exceed the width-256 wide-kernel arm. Promote
  above 15%; reject below 8%.

## 2026-08-17 — local-kernel 22-update transition (T=1)

- **Question:** Can we remove half the original recurrence without removing the
  local algorithmic bias that the 3x7 arm lost?
- **One change:** Reduce updates from 44 to 22 in the retained width-256 / 3x3
  ConvGRU. Keep its 1.77M parameters, seed, data, dropout, final-bit loss,
  tuned Muon, and 5.12M-example budget fixed.
- **Prediction:** Validation exact will remain above 18% while wall time falls
  by at least 40%. Above 20% retains 22 updates as the new transition anchor;
  below 15% says the original 44 steps encode necessary round-trip transport.

## 2026-08-17 — local-kernel 33-update transition (T=1)

- **Trigger:** The 22-update model matched the 44-update model through about
  3M examples but plateaued around 14% validation, showing that one 22-position
  traversal is insufficient.
- **One change:** Use 33 rather than 22 recurrent updates. Keep the width-256 /
  3x3 cell, data, seed, dropout, final-bit loss, tuned Muon, and 5.12M-example
  budget fixed.
- **Prediction:** Final validation will exceed 19% and wall time will stay at
  least 18% below the 44-step reference. Promote only if validation exceeds
  18% and the two unseen-N audits each exceed 16%.

## 2026-08-17 — official local E5 transfer of 33 updates

- **One change:** In the existing width-256 fused competition submission,
  reduce only internal transition updates from 44 to the locally promoted 33.
- **Prediction:** The official local E5 runner will complete successfully,
  process more than the 169 updates recorded for the 44-step hosted card, and
  produce nonzero T=1 accuracy on both seen and unseen N. Failure, zero on
  either T=1 profile, or fewer than 200 optimizer updates rejects transfer.
- **Boundary:** This is a local public-data run only. Do not spend online quota.

## 2026-08-17 — width-128 / 33-update T=1 control

- **One change:** Reduce channels from 256 to 128 in the promoted 33-update,
  3x3 ConvGRU. Keep data, seed, loss, dropout, Muon, and 5.12M examples fixed.
- **Prediction:** Validation exact will exceed 16%, both unseen-N audits will
  exceed 13%, and wall time will stay below 800 seconds. Promote only if all
  three accuracy gates pass.

## 2026-08-17 — official local E5 transfer of width 128 / 33 updates

- **One change:** Reduce only channels from 256 to 128 in the validated
  33-update competition-interface candidate.
- **Prediction:** The official local E5 run will exceed 200 optimizer updates,
  beat the width-256/33 mean exact score of 0.25%, and produce nonzero T=1
  accuracy on both seen and unseen N. All three conditions are required.
- **Boundary:** Local public data only; do not spend online quota.

## 2026-08-17 — full-window T=1 gradients at batch 256

- **One change:** Increase `T1_FRACTION` from 0.50 to 1.00 in the width-128,
  33-update, batch-256 official E5 candidate. The architecture, optimizer,
  input rows, and final-label binary loss are unchanged.
- **Prediction:** Both T=1 profiles will be nonzero, with at least two correct
  examples on one profile, while completed updates remain above 450. This card
  optimizes the first certification rung, not mean mixed-T score.
- **Boundary:** Local public data only; do not spend online quota.

## 2026-08-17 — width-128 / 33-update / batch-128 official E5

- **One change:** Reduce only training batch size from 256 to 128.
- **Prediction:** Completed updates will exceed 850. Promotion requires mean
  exact above the batch-256 arm's 0.2917% and nonzero T=1 on both seen and
  unseen N; otherwise batch 256 remains the official-E5 anchor.
- **Boundary:** Local public data only; do not spend online quota.

## 2026-08-17 — width-128 / 33-update / batch-256 official E5

- **One change:** Reduce only training batch size from 512 to 256 in the
  width-128 / 33-update official candidate.
- **Prediction:** The local E5 runner will exceed 450 updates, beat 0.125% mean
  exact, and produce nonzero T=1 accuracy on both seen and unseen N. All three
  conditions are required for promotion.
- **Boundary:** Local public data only; do not spend online quota.

## 2026-08-17 — half-scale initialization on the promoted T=1 machine

- **Question:** Does the tied ConvGRU begin with dynamics that are too large or
  saturated to discover a reusable local transition efficiently?
- **One change:** Multiply every learned parameter by `0.5` immediately after
  initialization in the promoted width-128, 33-update direct T=1 harness. Keep
  the deterministic data/splits, seed, final-residue-bit loss, dropout, tuned
  Muon, batch 512, and 5.12M-example budget unchanged.
- **Prediction:** Validation exact will exceed the 16.42% anchor and neither
  unseen-N audit will fall below 13%. Reject if validation is below 14%, or if
  training is materially slower at matched examples.
- **Selection:** Choose only on held-out-x/seen-N validation; open the two
  unseen-N audits once after selecting the best validation checkpoint.

## 2026-08-17 — Muon warmup plus cosine decay on the promoted T=1 machine

- **Trigger:** Half-scale initialization learned quickly but oscillated after
  about 6,500 steps while tuned Muon remained at its peak learning rate.
- **One change:** Replace the anchor's post-warmup constant Muon rate of
  `0.006` with cosine decay to `0.001` by step 10,000. Restore the anchor's
  original initialization; keep all data, architecture, seed, dropout, batch,
  loss, weight decay, and example budget unchanged.
- **Prediction:** Validation exact will exceed 17.0%, both unseen-N audits will
  exceed 14%, and the validation curve will not drop by more than 1.5 points
  after its first 12% checkpoint. Promote only if validation and both audits
  beat the unchanged anchor (`16.42%`, `13.62%`, `16.80%`).

## 2026-08-17 — delayed Muon decay after transition discovery

- **Trigger:** Full-horizon cosine decay fell below the anchor by step 5,000;
  the half-scale run's instability began only around step 6,500.
- **One change:** Hold tuned Muon at `0.006` after warmup through step 6,500,
  then cosine-decay to `0.001` over the final 3,500 steps. Restore every other
  anchor setting, including ordinary initialization.
- **Prediction:** Match the anchor through step 6,500, then exceed 17.0%
  validation. Promote only if validation and both unseen-N audits beat the
  unchanged anchor; reject if validation stays below 16.42%.

## 2026-08-17 — exact-square isolation at the promoted capacity

- **Question:** Is the centered-quotient cliff caused mainly by squaring a
  large centered operand, or by reducing the resulting 22-bit square?
- **One change:** Feed the exact 22-bit `x*x` source tape instead of the 11-bit
  `x` tape to the unchanged width-128, 33-update local model. No intermediate
  labels or traces are supplied; the target remains final residue bits only.
- **Prediction:** If squaring is the bottleneck, validation should exceed 30%.
  If reduction is the bottleneck, exact-square validation will remain below
  20% and retain the quotient-depth cliff. The latter result pivots research
  toward a deeper/faster reducer rather than a larger squarer.

## 2026-08-17 — exact-square reducer with 44 local updates

- **Trigger:** The 33-update reducer collapses abruptly when the centered
  quotient reaches 32, consistent with a recurrent transport horizon.
- **One change:** Increase only local recurrent updates from 33 to 44 in the
  width-128 exact-square isolation. Keep data, seed, tuned Muon, constant rate,
  dropout, loss, batch, and 5.12M-example budget fixed.
- **Prediction:** If clock depth is causal, validation will exceed 20% and the
  `q=32..63` bucket will exceed 5% exact. If validation remains below 18% and
  `q=32..63` remains below 2%, reject longer recurrence as a standalone fix.

## 2026-08-17 — exact-square reducer with 55 local updates

- **Trigger:** Moving from 33 to 44 updates moved the near-exact raw-quotient
  frontier from `q<16` to `q<32`, exactly one quotient bit per 11 clocks.
- **One change:** Increase only recurrent updates from 44 to 55. Keep the
  width-128 exact-square data, seed, optimizer, loss, dropout, batch, and
  5.12M-example budget unchanged.
- **Prediction:** Validation will exceed 23%, raw `q=32..63` exact will exceed
  50%, and `q>=64` will remain below 10%. Passing this shape, even if aggregate
  accuracy is modest, confirms a linear recurrent-clock law and directs the
  next architecture toward faster bit transport.

## 2026-08-17 — cyclic dilated transport in the exact-square reducer

- **Trigger:** 33→44→55 clocks progressively moved the quotient frontier but
  made matched-example runtime grow from 549s to 837s to 1065s.
- **One change:** At the original 33 clocks, apply the same tied 3x3 ConvGRU
  weights with horizontal dilations cycling `1,2,4,8`. The weights, parameter
  count, data, seed, optimizer, dropout, loss, batch, and example budget remain
  matched to the 33-clock exact-square reducer.
- **Prediction:** Validation will exceed 23%, raw `q=32..63` will exceed 40%,
  and runtime after compilation will stay below the 55-clock arm. Reject if
  validation is below 18% or the `q=32..63` bucket remains below 10%.

## 2026-08-17 — local reducer plus sparse fast messages

- **Trigger:** Replacing local convolution with dilation cycling regressed,
  showing that long-range reach cannot displace local arithmetic updates.
- **One change:** Restore the ordinary tied 3x3 cell for all 33 clocks. Before
  every fourth clock only, add a fixed `0.25` residual message from horizontal
  offsets cycling `2,4,8`; the operation adds no learned parameters. Keep the
  exact-square data and all training settings unchanged.
- **Prediction:** Preserve at least 17% validation while raising raw
  `q=32..63` above the ordinary 33-clock reducer's `0.27%`; promote only if
  validation exceeds 18% and `q=32..63` exceeds 10%.

## 2026-08-17 — zero-initialized learned fast-message gates

- **Trigger:** Fixed `0.25` sparse messages corrupted local arithmetic and
  strongly underfit, so fast communication must be optional and selective.
- **One change:** Replace the three fixed message scales with learned scalar
  gates initialized exactly at zero. The model therefore starts as the
  ordinary local reducer; all message timing/distances and training settings
  otherwise match the fixed sparse-message card.
- **Prediction:** Validation will recover above 16% and at least one gate will
  move beyond absolute `0.01`. Promote only if validation exceeds 18% and raw
  `q=32..63` exceeds 10%; a near-zero gate result says this message form is not
  useful, while opened gates without a frontier gain indicate a shortcut.

## 2026-08-17 — dedicated scratch-lane fast messages

- **Trigger:** Learned gates opened, but mixing transported state into every
  lane left the quotient frontier unchanged. Lane 3 is otherwise unused.
- **One change:** On every fourth clock, write the learned, zero-initialized
  shifted message from work lane 2 into scratch lane 3 only. Keep the ordinary
  local cell, message distances/timing, parameter count delta, data, and all
  optimizer settings matched to the learned mixed-message card.
- **Prediction:** Validation will exceed 18% and raw `q=32..63` will exceed
  10%, while shallower quotient buckets remain above 95%. Otherwise reject
  sparse fast messages in this form and redesign the reducer's state machine.

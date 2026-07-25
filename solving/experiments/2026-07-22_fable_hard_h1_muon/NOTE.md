# fable_hard_h1_muon

**CHANGE:** replaced fable_hard_h1's flat-lr=3e-4 WarmupSchedule with Muon (hidden weight matrices) + AdamW (embeddings/norms/biases) hybrid, wall-clock schedule.

**RESULT:** confirmed, strongly — Muon is the right call here.

**Detail:** 1,595,648 params, e5, 60s, 3,616 steps. Full train convergence (loss ~1e-5, 100% exact-match) by step ~2800 — dramatically faster than the AdamW-only variant, which was still at loss 1.82 / 7% train accuracy after 4,311 steps in the same window. **ood hit 2.0%** — the best number on e5 all session (previous best on any e5-family run today: 0.75% test on the flat RoPE anchor). test 0.42%.

**m5 (Medium, 600s): flat.** 27,759 steps, loss stuck ~2.1-2.2 the entire run, train accuracy never above ~1%, test 0.12% / ood 0.07% — the same flat pathology as the *original* broken-optimizer run, on a card that just crushed e5. Unexpected: Muon didn't just fail to help on m5, it failed as completely as the un-fixed lr=3e-4 flat schedule did.

Leading hypothesis, unconfirmed: m5's T range goes up to 8 vs. e5's up to 3 — the effective backprop depth through the unrolled recurrence is up to 2 tied layers × 8 outer loops = 16, vs. e5's ≤6. The harness applies a single global `grad_clip=1` across all parameters before any optimizer sees the gradient (`benchmark/runner.py` `_train`, unconditional, not submission-controlled). If one part of a deeper m5 unroll dominates the pre-clip gradient norm, the clip could wash out signal to the matrices Muon actually orthogonalizes, everywhere else in the network. Muon's lr=0.02 was only ever exercised at e5's shallow depth — not necessarily calibrated for m5's deeper one. Untested: rerun m5 with `scripts_local/monitor_train.py` (weight/grad norm visibility, already built) to see whether grad norm is pinned at the clip ceiling the whole run, which would support this mechanism directly.

Metrics: stdout only. Log: `solving/RESEARCH_LOG.md` (pending).

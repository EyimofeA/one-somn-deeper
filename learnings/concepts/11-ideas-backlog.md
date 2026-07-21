# Ideas backlog

Three columns so we do not mix taste, evidence, and literature. Append-only; mark status when we try something.

## Yours (principal)

| Idea | Why | Status |
|------|-----|--------|
| Learned / adaptive T (or early stop) so compute follows T | Fixed K wastes or under-spends | tried soft-ACT (incomplete); full ACT+ponder still open |
| Must generalize mod across many N, not one fixed N | e5 collapse | active constraint |
| Efficiency under 60s clock | Always | active |
| Diffusion on output / iterative answer refine | Later experiment | queued |
| Model-size scaling laws (score vs params under clock) | When we grow past d≈32 | queued — not ponder-τ |
| Reserve 10 Easy/day for Claude Code + critique | Collaboration | **active today** |
| T²MLR / middle-only recurrence (Cai et al.) | Tweet + arXiv 2607.15178 | **tried** midloop depth — e1 0.83% / e5 0.79% vs UT; reject |

## Ours (from measurements)

| Idea | Evidence | Status |
|------|----------|--------|
| Small width + tied loops beat wide shallow on Easy e1 | d32×K4 = 5.5% vs d64 K1 = 1.3% | adopted as reference |
| Always gate on e1 **and** e5 | 5.5% → 0.79% same weights | adopted |
| K-sweep at d=32 before adaptive halt | Only know K=1 vs 4 well at d=32 | **done** — e1 peak K=2 (6.2%); e5 peak K=4 (0.8%) |
| N-conditioning inside the same looped block | e5 needs many rings ℤ/Nℤ | **tried** — FiLM e1≈flat, e5 worse (0.29% vs 0.80%); reject **this recipe**, not the goal — FiLM is a smooth affine modulation of a *pooled* N, but mod-N reduction is a hard **digit-local** operation. Rejecting FiLM is not evidence against N-conditioning |
| Cross-attn / dedicated N tokens into block | FiLM pool may be too lossy | queued — **highest-EV untested axis**. Do it place-aligned (each X digit attends N's digits by place id, see `16-representation-vs-throughput.md`), not pooled |
| ACT / adaptive halt on tied loops | Your idea; K-sweep done | **tried** — soft ACT K_max=8: e1 3.83% (worse), e5 0.79% (≈K4); not promoted |

## Researched (prior art)

| Idea | Pointer | Status |
|------|---------|--------|
| Weight-tied block + ACT / halt | Universal Transformer; Graves ACT | soft-ACT tried; **UT depth-emb fixed-K next** |
| Extra loops at eval for harder inputs | Looped / recurrent-depth Transformers | candidate |
| Iterative latent updates for algorithms | Algorithmic reasoning / vertical CoT notes | background |
| UT depth/timestep embedding each loop | Dehghani et al. 2018; we omitted this | **tried** — UT K2 e1 6.50%; UT K4 e5 **1.00%** (new e5 best) |
| UT improvements (eval-only extra loops; halt bias; LR) | Karpathy: before aux | **tried** train2/eval4 — e1 **6.83%** (new best), e5 0.42% (worse) |
| Aux — *label-aware* intermediate supervision | Not expressible through the API | **blocked, structurally.** `forward` receives only `(input_ids, attention_mask)` and `training_loss` only `(loss_logits, loss_labels, auxiliary)` — both already flattened to valid target tokens. No `target_positions`, so a per-position intermediate target cannot be addressed. This is an API-shape limit, *not* "metrics are hidden" |
| Aux — *label-free* regularisers | Reachable today via the `auxiliary` return | **available, untried** — `forward` returns `(logits, auxiliary)` and `auxiliary` is passed straight to `training_loss`, so ponder cost, per-loop consistency ‖h_k − h_{k−1}‖, or entropy are all differentiable and in reach |

## Not doing (for now)

- Untied deep stacks like an LLM (burns params, wrong bias for “same step again”)
- **Scaling width to spend the 500M budget** — tested, lost badly: d=128 scored e1 2.00% vs d=32's 6.17% on an otherwise identical card. The 60s clock is the binding constraint, not parameters; there are ~4 orders of magnitude of unused budget and reaching for them does not help
- Easy vanity 100% via arithmetic circuits / hardwired solvers (Discord: trivial + Hard-broken)
- Hard/Medium until principal greenlights after e5 movement
- Treating e1-only wins as done
- Banking Hard on intermediate-step aux loss without re-checking final rules

(Update 2026-07-21: e5 moved to 1.00% with UT K4 — Medium still principal-gated. Discord meta: `14-discord-beta-meta.md`.)

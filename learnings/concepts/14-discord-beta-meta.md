# 14 — Discord / beta meta (GPU MODE #one-layer-deeper)

Working digest of organizer + participant chat (pasted 2026-07-21). Not a rules substitute — verify against README/website when something must be binding. Beta: rules still moving; LBs have been wiped when loopholes close.

## Calendar / logistics

| Claim | Source |
|-------|--------|
| Deadline TBD; “likely end of August” | Mark Saroufim, 2026-07-19 |
| Teams allowed | Mark |
| Submissions private during; public when competition concludes | Mark |
| Winners may be asked to open-source for repro; case-by-case | Mark |
| Prize TBD (“love of the game” / interview chat) | Mark |
| Eval GPUs: H100 currently | mcleish7 / Mark |

## What organizers say they want

- Architecture and/or **optimizer** ideas under a small param ceiling relative to the GPU. [SOURCED Discord mcleish7]
- Recurrence encouraged; more attention/compute complexity also welcome if arch/optim based. [SOURCED Discord mcleish7]
- Reward-hacking during **beta** is explicitly invited to refine rules. [SOURCED Discord Mark]
- Hard LB is the only ranked track; Easy/Medium are practice.

## Easy/Medium vs Hard (critical)

| Fact | Source |
|------|--------|
| Easy/Medium high score can be **100%** via task-specific / hardwired solvers | az; apaz (“trivial to design weights that solve exactly”) |
| A Hard entry that hardwired digit add/mul + `r² mod N` looped T times was called out as a cheat | Lord Tr0lldem0rt self-report; az: “100% shouldn’t be allowed” |
| Hard was edited so **hard-coding the Easy/Medium algorithm no longer works** | mcleish7 — “slight variant… mismatch secret”; LB resets after patches |
| Hard is still “extremely close” to Easy/Medium — just far enough to break exact Easy solvers | mcleish7 |
| No **data augmentation** on Hard (leaderboard) runs | mcleish7 → website rules |

**Implication for us:** Easy mean ~6% is not “losing to people who learned better” if they shipped solvers. The research score that matters is **Hard under the mismatched recurrence**, with a model that still trains.

## Cheating / grey (Discord refinements)

Already forbidden in README: hard-coded weights; hard-coded algorithm in forward; data inspection; custom train loop.

Discord clarifications / debates:

- **Task-specific solver in weights** (one-hot digits + FFN lookup tables for arithmetic, recurrent counter for T) = cheat for the intended competition. [SOURCED Discord]
- **Decomposing halt so a layer only learns “count to T”** while the rest is programmatic = grey → organizers treating as loophole territory; az proposed hidden eval recurrence (e.g. cube instead of square) to kill this class. Hard mismatch is the partial answer.
- **Supervise hidden states to match intermediate computation steps:** asked by Kham; mcleish7 said fine under *current* beta rules and encouraged trying craziest ideas — **not a promise it stays legal**. Log as grey; do not bank a Hard strategy on it without re-checking.
- Forward that **branches on T** without computing the answer: asked (Asher); no clear public ruling in the paste — treat as grey.

## Papers named in-channel

| ID | Title (verified) | Relevance claimed in Discord |
|----|------------------|------------------------------|
| [2501.19215](https://arxiv.org/abs/2501.19215) | Strassen Attention… Compositionality in Transformers | One-layer softmax TF limits on composition; Strassen attention as fix |
| [2602.21371](https://arxiv.org/abs/2602.21371) | Interleaved Head Attention (IHA) | Cross-head mixing for multi-step / compositional attention; cited alongside 2501.19215 |

Also ambient: looped Transformers / latent thoughts literature (e.g. Saunshi-style looping papers) matches the competition thesis — not Discord-unique.

## Our posture (sandbox)

1. Keep **learning** submissions (UT/tied loops, optim) as the main line — that is what Hard is designed to reward.
2. Do **not** optimize Easy for 100% with arithmetic circuits unless explicitly probing the cheat boundary for notes.
3. Before Hard: re-pull repo + re-read rules; assume Easy-perfect ≠ Hard-ready.
4. Optional grey experiment (principal call): intermediate-state aux loss on Easy only — document as beta-grey.

## Still open after this paste

- Exact Hard mismatch (organizers keep secret) — we must not reverse-engineer via forbidden means.
- Whether intermediate supervision survives into final rules.
- Official Easy/Medium public LBs (participants track informally; apaz: not important vs Hard).

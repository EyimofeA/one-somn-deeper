Read CLAUDE.md and FULL_TRANSCRIPT.md in this folder before doing anything else.

Context: I'm competing in the One Layer Deeper ML competition (Hard tier — iterated
modular squaring, recurrence may be altered). GitHub handle EyimofeA, repo
one-somn-deeper, competition handle mof. Ranked ~#10-11 at 0.03%, board leader 0.40%.

Current state: submission_v2.py fixes a training-stability bug (confidence-gated
quantization hardening instead of wallclock-gated, which caused an irreversible
collapse) but has NOT been shown to solve the actual problem. Independent real-GPU
runs show every architecture tried so far — including this family — converges to the
digit-marginal floor (~2.17 test loss, ~0.75% exact-match even at T=1, single step,
no depth composition involved). The register/loop machinery is not obviously the
issue; the one-step modular-reduction map itself has not been shown learnable for an
unseen modulus.

First task, in order:
1. Get my real repo checked out here (or tell me what you need from me to do that —
   I have `gh` / git access on this machine).
2. Run smoke_test.py against submission_v2.py in this repo's actual environment to
   confirm the harness replica matches the real evaluator (flag any discrepancy).
3. Build the P2 grokking ladder described in FULL_TRANSCRIPT.md: fixed single N
   (unseen x) -> multi-N seen at train (unseen x) -> held-out N, all at T=1 only.
   This is the actual gate. Do not treat any further Hard-tier architecture work as
   informative until rung 3 clears roughly 5% exact-match locally.
4. Report back per-rung results with the diagnostics from the transcript's D1-D7 list
   (per-position digit accuracy, loop-state decoding, epsilon split by seen/held-out
   N) — not just pass/fail.

Budget note: I have limited API budget left. Keep context lean — summarize/compact
after each rung resolves rather than carrying full logs forward, and don't re-read
PRIMARY_SOURCES.md or FULL_TRANSCRIPT.md in full more than once unless something
specific needs re-checking.

Do not fire a Hard-tier submission without telling me first and getting a go-ahead —
it's 1/day and monotone-best-kept, so it's low-risk, but I want to know when it
happens.

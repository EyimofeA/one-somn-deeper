# File-organization audit

## Judgment

The repository is navigable because `AGENTS.md` supplies a strict read order,
but the filesystem itself no longer communicates a single source of truth.
Do not broadly move files now: many research-log citations and historical
commits depend on the current paths. Repair discovery first, then archive in a
separate mechanical change.

## Measured shape

- 345 tracked files under `solving/`, 197 under `diagnostics/`, and 76 in the
  frozen `research_packet_2026-08-05/` snapshot.
- 51 dated experiment directories.
- 44 Python files directly in the top level of `diagnostics/`.
- 34 distinct `submission.py` files across the tree.
- 38 experiment `NOTE.md` files and 26 experiment `config.json` files.

## Main problems

### 1. The declared experiment layout and actual layout disagree

[`experiments/LAYOUT.md`](experiments/LAYOUT.md) says dated directories should
contain measured artifacts only and explicitly says not to copy `NOTE.md`,
`config.json`, or `submission.py` there. The actual tree has dozens of each.
The current practice is more reproducible than the declared policy, but the
contradiction makes every new card ambiguous.

**Recommendation:** amend `LAYOUT.md` in a dedicated policy commit: allow a
compact `NOTE.md`, `config.json`, source snapshot, and small reports when they
are the only durable provenance; keep checkpoints and bulky metrics ignored.
Do not pretend Git commits alone retain remote run configuration.

### 2. “Submission” has three meanings

- `competition/submissions/`: upstream/live checkout examples plus temporary
  validation copies;
- `solving/submissions/`: owner-curated upload candidates;
- `solving/experiments/*/submission.py`: historical per-card source snapshots.

This is the highest-risk ambiguity because an agent can upload a stale or
research-only file while believing it selected the active candidate.

**Recommendation:** retain all three roots but require a `CARD.md` containing
source SHA-1, validation pin, hosted job IDs, legality status, and promotion
status for every directory in `solving/submissions/`. Rename or delete only
temporary copies after exact provenance reconciliation.

### 3. `STATUS.md` is both dashboard and archive

The living status begins with current results but then contains a long
historical narrative already present in `RESEARCH_LOG.md`. Stale states are
therefore easy to quote as current.

**Recommendation:** cap the opening `Now` section at one screen: current
question, last decisive result, active compute/job, next registered gate, and
current upload candidate. Keep the existing historical body temporarily, then
replace sections with anchored links to `RESEARCH_LOG.md` once citations are
checked.

### 4. The frozen research packet pollutes ordinary search

`research_packet_2026-08-05/` is a valuable immutable handoff, but duplicates
source, status, reports, and even a Git bundle. Ordinary `rg` searches return
both live and frozen claims.

**Recommendation:** add a root `.rgignore` for the packet, checkpoints,
virtual environments, caches, and ignored run artifacts. Researchers can
still search the snapshot explicitly with `rg --no-ignore <term>
research_packet_2026-08-05`.

### 5. `diagnostics/` has become a flat chronology

The original documented suite is cleanly grouped into `data/`, `models/`,
`configs/`, and `tests/`, but 44 later research scripts sit at its root. Names
carry chronology and mechanism, yet shared primitives and one-off drivers are
mixed together.

**Recommendation:** do not move them mid-campaign. Add a generated
`diagnostics/INDEX.md` with columns for capability, canonical/one-off status,
latest evidence, and superseded-by path. After the competition, migrate
canonical primitives into `diagnostics/models/` and run drivers into
`diagnostics/scripts/` in one citation-rewrite commit.

## Priority order

1. Submission provenance cards.
2. One-screen `STATUS.md` opening.
3. `.rgignore` for frozen/ignored trees.
4. Diagnostics index.
5. Post-competition physical reorganization.

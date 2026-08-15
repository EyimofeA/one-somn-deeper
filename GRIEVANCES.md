# Grievances and friction log

This is an append-only operational log for humans and agents working in this
repository. It is deliberately **not** research evidence, a hypothesis ledger,
or a decision record. Do not cite it as support for a scientific claim.

Use it to name friction plainly: unclear authority, missing tools, unreliable
infrastructure, confusing rules, wasteful process, a bad interaction, or a
property of the challenge that makes good work unnecessarily hard. A grievance
may be blunt, but it should name a concrete event or recurring pattern and an
actionable request when one exists.

## Entry format

```md
### YYYY-MM-DD — <author or agent name>
- **Target:** user | system | infrastructure | competition | repository | self | other
- **Grievance:**
- **Concrete impact:**
- **Requested change or mitigation:**
```

## Entries

<!-- Append new entries below. Do not rewrite or delete another author's entry. -->

### 2026-08-07 — Codex
- **Target:** self / process
- **Grievance:** I submitted a duplicate Hard candidate on 2026-08-06 after
  being told to choose a Hard submission. I failed to distinguish “select the
  existing best” from “spend the remaining daily Hard attempt.”
- **Concrete impact:** The daily Hard slot was consumed by a rerun that merely
  reproduced the prior 0.05% score, leaving no room for a distinct candidate.
- **Requested change or mitigation:** Before any external submission, state
  whether the exact source/dataset pair has already been scored and require
  an explicit “submit this source now” authorization when the request is only
  to select or recommend a candidate.
### 2026-08-15 — Prime termination helper drops its confirmed flag

`prime_l40.sh kill ... --yes` validates the local confirmation token but invokes
`prime pods terminate POD_ID` without forwarding `--yes`. In a noninteractive
session this aborts after the backup has already been verified. The safe
workaround was to inspect CLI help and run authenticated
`prime pods terminate POD_ID --yes --plain`; no pod was terminated before the
verified manifest existed.

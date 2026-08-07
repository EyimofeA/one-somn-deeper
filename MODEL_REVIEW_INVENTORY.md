# Model-review inventory

Updated 2026-08-05. This inventory separates source material from project evidence.

## Available review/chat sources

| Source | Type | Coverage | Status |
|---|---|---|---|
| `claude code fable/FULL_TRANSCRIPT.md` | Claude research chat | Hard strategy, tied token-register recurrence, held-out-modulus diagnosis, RNS proposal | In corpus |
| `claude code fable/PRIMARY_SOURCES.md` | Claude handoff/source packet | Rules, evaluator contract, historical metrics and assumptions | In corpus; source material, not model opinion |
| `review_packets/latent_state_second_opinion_2026-08-05/GPT_PRO_REVIEW_PROMPT.md` | GPT Pro review prompt | Adversarial review request for latent-state VDF hypothesis | Prompt only; review response absent |
| `review_packets/latent_state_second_opinion_2026-08-05/{README,STATUS,FAILED_EXPERIMENTS}.md` | Project-authored packet | Current evidence and failure table supplied to reviewers | In corpus; project evidence |
| `research_packet_2026-08-05/` | Consolidated research packet | Logs, hypotheses, reports, code, cited papers | In corpus; project evidence |
| `claude code fable/{START_PROMPT.md,PRIMARY_SOURCES.md}` | Claude inputs | Problem framing and constraints | In corpus |

## Not found locally

No complete ChatGPT Pro response, Sol, Gemini, Grok, or other external research-chat transcript was found. The two `pi-session-*.html` exports contain only export chrome and no conversation body. These sources must be supplied/imported before they can be treated as review evidence.

## Evidence anchors

- Current competition outcomes: `solving/submissions/SUBMISSION_EXECUTION_REPORT.md` and `runs/fable_tcap_adamw_{easy_e1,medium_m1}/result.json`.
- Controlled recurrence evidence: `review_packets/latent_state_second_opinion_2026-08-05/STATUS.md` and `research_packet_2026-08-05/reports/`.
- Legal candidate source: `solving/submissions/fable_tcap_adamw/submission.py`.

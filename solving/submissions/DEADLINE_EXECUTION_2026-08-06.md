# Deadline execution — 2026-08-06

The selection rule is expected leaderboard value from direct hosted evidence.
The serial arithmetic diagnostics are scientific evidence, not legal
competition candidates: they use generated transition traces unavailable to the
official submission training contract.

| Candidate | Tier / dataset | Architecture and configuration | Best local / hosted evidence | Runtime | Decision / risk |
| --- | --- | --- | --- | --- | --- |
| Fable T-cap AdamW, batch 512 | Easy e1 | 256-wide tied transformer register; train depth cap 16, eval cap 64; AdamW 3e-3, batch 512 | Hosted 2026-08-04: 8.50%; current hosted `1b06d008-89e0-4a49-90d4-f2589a969ed6`: **8.00%** | 60 s train + evaluator-owned evaluation | Submitted. Strongest direct Easy evidence; no certified rung, so score is prompt/statistical rather than algorithmic. |
| Fable T-cap AdamW, batch 256 | Medium m5 | Exact historical Fable source; same model, optimizer, and depth caps as above, batch 256 | Hosted historical `aa699c3f`: 0.25%; current `60510147-ed5b-4944-97b5-ddce0340b883`: **0.17%** (0.20% test / 0.10% OOD). Recent batch-512 m1: 0.03%. | 600 s train + evaluation | Submitted and completed. Best directly observed Medium family; narrow absolute margin and m5-specific evidence. |
| UT K4, clamped cosine | Medium m5 | 32-wide, four tied UT blocks; AdamW, batch 256; non-restarting cosine | 0.20% historical mean | 600 s | Not submitted: dominated by Fable 0.25% historical result. |
| UT K4 + STE bottleneck | Medium m5 | Same small UT with discrete inter-loop bottleneck | 0.17% historical mean | 600 s | Not submitted: directly refuted against its parent. |
| Final-label recurrent VDF | Easy / Hard | Tied learned LSD-first SquareCell then ReduceCell, dynamic T | Easy local: 0.50%; Hard hosted: 0.04% | 60 s / 3,600 s | Not submitted today: scientifically relevant but dominated on observed competition score. |
| Fable v2 confidence-gated register | Hard h1 | 256-wide tied register with confidence-gated recurrence | Hosted `602bf7f1-eab7-46c2-91e8-e4a4a010f9d7`: **0.0467%** mean; current VDF Hard: 0.0367% | 3,600 s | Submitted as `c377241f-528a-43b4-b1ac-3aab139543a3`. Risk: neither prior candidate certifies T=1; exact score is weak. |

## Provenance

- Easy source: `solving/submissions/fable_tcap_adamw/submission.py`, commit
  `871fe4f` at submission time.
- Medium source: `solving/experiments/2026-07-25_fable_tcap_adamw/submission.py`.
  Its only code difference from the Easy source is `batch_size=256` instead of
  `512`.
- Hard source: `solving/experiments/2026-08-04_deadline/hard_fable_v2/submission.py`,
  SHA-1 `d16e4ec932d4bea38bf1bff4c74ac2ab2bb0d838`.
- Both sources passed `one-layer validate` immediately before upload.
- Hosted job details are recorded in the research log and may be retrieved by
  the IDs above.

## Competition transfer screen (Track B)

| Candidate transfer | Coding time | Training time | Chance of beating selected candidate today | Decision |
| --- | ---: | ---: | ---: | --- |
| Global latent recurrent workspace | 90 min | 60 s Easy / 600 s Medium plus local validation | <5% | Do not run. It has a small synthetic signal (17.29% unseen-N T=1 versus 9.35% control), but no public-scale or official-wrapper evidence. |
| Better recurrent input initialization | 45 min | 60–600 s plus local validation | <3% | Do not run. Controlled fixed-N workspace evidence did not establish a full VDF gain. |
| Trace-inspired intermediate loss | 30 min | 60–600 s | 0% legal | Do not run. It requires generated internal targets unavailable under the official final-label contract. |
| Final-label T curriculum | 15 min | 60–600 s | <1% | Do not run. Directly refuted by the final-label curriculum controls. |

## Hard decision gate

Do not spend the next Hard attempt on a new research model. The historically
stronger legal Fable v2 source is selected and is being submitted before the
cut-off, rather than the final-label recurrent VDF source. The job needs a
full 3,600-second training allowance plus its evaluator-owned evaluation
window.

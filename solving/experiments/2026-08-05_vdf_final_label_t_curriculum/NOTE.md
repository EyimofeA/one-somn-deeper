# Final-label T curriculum

Author: Codex

This is a legal-objective diagnostic on public e1, where official train rows
already contain T=1, T=2, and T=3. The model always executes the input's exact
T. Only the custom token loss is staged over existing rows: first T=1, then
T≤2, then T≤3. It adds no intermediate target, trace label, solver, data row,
or phase-specific transition weight.

The current experiment is not a Medium candidate: public m1 contains only
T=4/8/16, so literal T=1 curriculum cannot be tested there without changing
the public training distribution.

## Result

Local L40 e1 run, seed 74, 60.1 seconds: 461 updates, 3.33% test exact and
0% OOD exact (1.67% mean). The seen-N depth ladder is T=1 5.2632% (2/38), then
0% for T=2,4,8,16,32,64; the OOD-N ladder is at most 0.3906%. This fails the
promotion criterion. It exposes a small one-step effect but no composed or
OOD mechanism. Artifact excluded from Git:
`runs/vdf_final_label_t_curriculum_e1/competition_report.md`.

## Architecture audit (e1, same local L40, one seed)

| Model | Updates | Test | OOD | Seen T=1 | Seen T=2 | Certified T |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A. Direct final-output Transformer | 697 | 2.00% | 1.00% | 2.6316% | 2.6316% | none |
| B. Tied Square→Reduce VDF | 434 | 3.33% | 0% | 0% | 0% | none |
| C. Tied VDF + final-label curriculum | 461 | 3.33% | 0% | 5.2632% | 0% | none |

All models are far below a certified rung. The direct model's apparent 1% OOD
win is not a reusable depth signal: its seen ladder has an isolated 10.5263%
at T=32 while T=1–16/64 remain near zero. Curriculum creates a weak T=1
signal but it disappears at T=2. The diagnostic trace-loss control likewise
has zero held-out final exact at T=1/2/3 despite better in-batch fitting.
Per-T reports and train/dynamics SVG charts are retained under Git-ignored
`runs/vdf_{architecture_audit_direct_transformer,final_label_t_curriculum_e1,square_reduce_dynamic_depth_e1}/`.

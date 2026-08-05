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

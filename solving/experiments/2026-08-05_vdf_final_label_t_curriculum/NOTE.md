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

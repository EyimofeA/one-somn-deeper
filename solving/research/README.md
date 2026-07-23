# Active research code

This directory holds canonical implementations for mechanisms that are still
being tested. Git commits preserve exact code and configuration history.
`solving/RESEARCH_LOG.md` is the sole durable per-experiment narrative;
experiment directories retain only measured artifacts worth keeping.

For the carry-normalization diagnostic, edit only `carry_scan.py`. New local
cards use a tiny wrapper:

```python
from solving.research.carry_scan import CarryScanSettings, build_submission

SUBMISSION = build_submission(CarryScanSettings(num_prototypes=64))
```

Before an upload, materialize the active artifact:

```sh
python scripts/freeze_carry_scan_submission.py \
  --num-prototypes 64 \
  --output solving/submissions/<card>/submission.py
```

Record the source commit and freezer arguments in `solving/RESEARCH_LOG.md`.
The generated file is self-contained; the research runner never depends on an
implicit package being present on the competition host.

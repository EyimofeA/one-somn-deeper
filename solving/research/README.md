# Active research code

This directory holds canonical implementations for mechanisms that are still
being tested. Experiment folders remain immutable records: their
`submission.py` is a frozen, standalone artifact for the exact run.

For the carry-normalization diagnostic, edit only `carry_scan.py`. New local
cards use a tiny wrapper:

```python
from solving.research.carry_scan import CarryScanSettings, build_submission

SUBMISSION = build_submission(CarryScanSettings(num_prototypes=64))
```

Before an upload or final card freeze, materialize it:

```sh
python scripts/freeze_carry_scan_submission.py \
  --num-prototypes 64 \
  --output solving/experiments/<card>/submission.py
```

The generated file is self-contained; the research runner never depends on an
implicit package being present on the competition host.

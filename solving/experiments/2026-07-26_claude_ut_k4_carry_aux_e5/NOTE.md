# UT K4 aggregate carry auxiliary

Question: does the strong offline carry-supervision result transfer to hosted
variable-N e5 through a cheap two-scalar auxiliary head?

Results:

- Python target generation + STE: 0.75% mean, 1,349 steps (`8e5457ad`).
- Tensorized target generation + STE: 0.58% mean, 2,179 steps (`4eff4824`).
- Tensorized target generation + continuous state: 0.38% mean (`16ebb553`).

Classification: refuted for hosted use. Tensorization repaired throughput, but
aggregate carry statistics are too coarse and continuous state did not recover
the 1.00% UT-K4 e5 champion.

# 2026-08-17 Hard deadline candidate

- Hard job: `a1132421-ef3b-4261-b145-13c68ef34f8c`
- Uploaded file: [`submission.py`](submission.py)
- Exact SHA-1: `b1773edcc356201737972e31adb49d6d8c62b856`
- Accepted before the UTC cutoff; zero Hard attempts remained afterward.
- Result: **0.02333% mean exact** (UI: 0.02%); no certified rung.
- Seen-`N` T=1: **0/768**; OOD-`N` T=1: **0/768**.
- Training: 109,143 updates; final loss **0.693338871**.
- Run page:
  <https://onelayerdeeper.ai/submissions/a1132421-ef3b-4261-b145-13c68ef34f8c>

## Selection basis

This was the strongest completed varying-modulus hosted candidate from the
2026-08-17 throughput sweep, not a single speculative research delta and not a
previous Hard source. It combines the retained choices from the sweep:

- binary four-lane work state;
- width 128;
- 11 tied recurrent updates rather than 44;
- batch 256;
- tuned flattened-convolution Muon with scalar AdamW;
- final-label learning of the whole transition.

Hosted E5 job `1cfa42ed-a7fb-4850-a8f1-3750c750846e` completed 1,474 optimizer
updates and scored **0.5417% mean exact**: 7/1,200 test and 3/600 OOD, with
final train loss 0.43902. The later batch-128 job
`231c4362-837b-4789-890d-4afa536fd407` scored only 0.2083%, so it was rejected.

This evidence does not certify T=1 and does not imply a strong Hard result. It
only makes this source the least-bad novel candidate available at the explicit
deadline. The H13 research model was not submitted: its local full-transition
validation was 6.14%, below its kill gate, and it was not translated or hosted.

## Outcome

Hard refuted the promotion. Across the three full evaluation profiles, the
model got 2/9,999 test examples, 2/10,002 OOD-`T` examples, and 3/10,002
OOD-`N,T` examples correct. Those isolated hits did not include any T=1 case,
so neither ladder certified. See [`metrics.json`](metrics.json).

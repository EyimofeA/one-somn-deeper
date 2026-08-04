# Easy submission report

## Candidate: `easy_serial_recurrent`

Status: submitted once; hosted result pending.

The candidate receives only `(N, x, T)` prompt tokens. A shared arithmetic cell
combines bidirectional learned attention with a right-to-left GRU scan. Learned
field and right-relative decimal-place embeddings preserve LSD alignment. The
same cell is applied up to four times; the parsed `T` chooses which learned
updates are active. Every output logit is produced from learned PyTorch modules.

No `x²`, modulus, quotient, factors, answer lookup, or diagnostic reducer input
is calculated in the forward pass. Public Easy only provides final labels, so
intermediate-state supervision is not permitted in the submission training
loop.

| Field | Result |
| --- | --- |
| Source validation | Pass; 6,367 bytes |
| Parameters | 21,152 persistent model-state elements |
| Local e1 training | 500 updates in 60.00 seconds on L40 |
| Local e1 test exact | 1.33% (2/150) |
| Local e1 OOD exact | 0.00% (0/100) |
| Local e1 mean exact | 0.67% |
| Historical e1 reference | 6.80% mean; its ranking value is documented as invalid |

The candidate is legal and is being submitted once as authorized, but the
local result gives no evidence it will improve the historical Easy score. Its
expected leaderboard value is therefore low. The tested source and output are
archived on the L40 at `~/somn-taskb/easy_serial_recurrent/e1/`.

Hosted Easy e1 submission: `dfac516b-2989-4fc1-beb5-0f1407174d42`.

Hosted result: **1.00%** mean exact; no certified T or OOD-N T rung. This is
slightly above the L40 local mean (0.67%) but remains below the historical
6.80% e1 reference, so it is not promoted as the primary Easy source.

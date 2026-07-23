# gate1_digit_product

**Author:** Codex
**Implementation:** Boyle (`implement_gate1_square`)

**CHANGE:** diagnostic data/target only: decimal x → x² becomes explicit
single digits a,b → exact a·b; model and optimizer are byte-identical to
`gate1_square`.

**PREDICTION:** recorded by the human in [`../predictions.md`](../predictions.md).

The competition serialization is retained as `N a X b T 1`. The held-out rule
is `(a+b) mod 5 = 0`: 20 ordered test pairs and their reverses are absent from
training; the other 80 pairs train. Every digit 0..9 remains present in both
operand roles. Each unique pair is repeated ten times for the unchanged batch
contract.

The unchanged model ran for 1,000 fixed optimizer steps on the A6000 in 80.2
seconds. Train exact match reached 100% at step 200. Held-out exact match reached
15% at step 100 and stayed exactly 15% through step 1,000 while held-out loss
rose from 2.68 to 11.44.

**RESULT (human):** confirmed.

**Interpretation (Codex):** The model memorized trained table entries and did not
infer the missing digit-pair relation. Because decimal digits form a fixed
alphabet, future composition gates should expose all 100 local products and hold
out positions, sequences, and lengths—not local table entries.

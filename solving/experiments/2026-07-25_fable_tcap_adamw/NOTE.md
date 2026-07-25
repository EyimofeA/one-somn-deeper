# fable_tcap_adamw

**CHANGE (one variable framing):** last Hard Fable card (`fable_hard_h1_muon`) but
(1) train-time T loop cap = 16, full parsed T at eval (≤64), (2) AdamW+wallclock
instead of Muon, (3) `eval_batch_size=1024`.

**Why:** muon Hard timed out under T≤64 unroll; muon also flatlined m5. New Hard
rank needs eval-time depth on the 1..64 ladder without paying 64 loops every train step.

**RESULT:** Medium m5 **0.25%** mean (`aa699c3f`, test 0.20 / ood 0.30, 22194 steps). Hard h1 **succeeded** (no timeout) at **0.03%** mean exact (`f4246e70`, ~60 min). Timeout fix confirmed; accuracy still floor vs prior Hard shots (~0.03–0.05%).

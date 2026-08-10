# Canonical smooth worst-digit loss

Status: completed; OOD-N gate refuted.

This is the exact canonical register architecture and schedule.  The only
change is the within-example reduction of valid output-token cross entropy:
instead of a mean, use a temperature-0.5 log-mean-exp.  This gives the worst
predicted digit more gradient without adding arithmetic structure or reading
anything beyond the permitted `TokenLossBatch` fields.

Promotion gate: at least 8/512 seen-N and 5/512 OOD-N exact at T=1.  Stop after
one local run if either side fails; this card is not permitted a hosted submit.

The run reached the seen-N boundary exactly at 8/512, but only 1/512 OOD-N.
Mean exact was 0.5833% (10/1200 test, 2/600 OOD).  Concentrating gradient on
the worst digit therefore sharpened seen-modulus fitting without learning a
modulus-general reduction rule.

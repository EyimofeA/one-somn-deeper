# Canonical local ConvGLU with four tied microsteps

Status: completed; refuted.

The exact strong local-Conv card applies one translation-shared neighbor update
per modular transition.  This control reuses that same learned block four times,
allowing information to move four adjacent LSD-first slots without adding
parameters, an arithmetic schedule, or intermediate labels.

Gate: at least 8/512 seen-N and 5/512 OOD-N T=1.  One local run only; no hosted
submission without a separate promotion decision.

The four-step card completed 1,491 updates and scored 0.3750% mean exact.
T=1 was 3/512 seen-N and 2/512 OOD-N.  Additional tied propagation distance
does not recover a reusable rule and costs about 12% of the one-step card's
update count.

# Post-Hard T=1 sprint summary

All numbers below are one-seed local public-e5 runner results unless marked
hosted.  Every architecture/loss card was preregistered before its run.

| Card | Updates | Mean exact | T=1 seen | T=1 OOD-N | Decision |
|---|---:|---:|---:|---:|---|
| local ConvGLU k=1 | 1,695 | 0.6667% | 5/512 | 1/512 | reject |
| local scale trajectory | 1,600 | 0.8333% | 5/512 | 2/512 | reject; branch active |
| smooth worst-digit loss | 1,760 | 0.5833% | 8/512 | 1/512 | reject; OOD gate |
| full-time T=1 | 1,612 | 0.1667% | 4/512 | 0/512 | reject; memorizes |
| modulus-length group DRO | 1,685 | 0.7500% | 2/512 | 1/512 | reject |
| local ConvGLU k=4 | 1,491 | 0.3750% | 3/512 | 2/512 | reject |
| structured-tape residual | 1,362 | 0.7917% | 3/512 | 2/512 | reject; underfit |
| direct structured tape | 1,953 | 0.4167% | 2/512 | 1/512 | reject |
| direct tape, 300 sec | 9,640 | 0.2083% | 2/512 | 0/512 | reject |
| direct tape, no position | 1,864 | 0.5000% | 3/512 | 1/512 | reject |

No card certified any depth rung or earned a hosted promotion.  The exact
machine-readable values are in [`results.json`](results.json); individual cards
contain source and raw runner logs.

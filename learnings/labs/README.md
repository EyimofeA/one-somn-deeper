# Interactive learning labs

Run Lesson 1 on CPU:

```bash
source .venv-learning/bin/activate
python learnings/labs/neural_gpu_01_state_and_cell.py
```

The lesson demonstrates tensor layout, local gated convolution, tied recurrence,
loss, and backward gradient flow. Do not use a GPU for these labs.

Run Lesson 2:

```bash
python learnings/labs/neural_gpu_02_signal_propagation.py
```

This uses a fixed teaching convolution to show locality and receptive-field
growth. It is deliberately not trained and is not a competition architecture.

Run Lesson 3:

```bash
python learnings/labs/neural_gpu_03_learn_the_transition.py
```

This trains a three-weight local convolution on fresh random tapes and shows
whether optimization discovers the reusable shift-right rule.

Run Lesson 4:

```bash
python learnings/labs/neural_gpu_04_length_extrapolation.py
```

This trains only on length-eight tapes, then directly measures transfer to
lengths 16, 64, and 256.

Run Lesson 5:

```bash
python learnings/labs/neural_gpu_05_rule_vs_compute_depth.py
```

This holds the learned local rule fixed while independently changing tape
length and the number of recurrent executions.

Run Lesson 6:

```bash
python learnings/labs/neural_gpu_06_vanishing_and_exploding.py
```

This shows how a tiny per-step contraction or expansion compounds in both the
forward state and the backward gradient.

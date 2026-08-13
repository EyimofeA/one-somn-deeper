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

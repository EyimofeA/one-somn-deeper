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

Run Lesson 7:

```bash
python learnings/labs/neural_gpu_07_decodable_is_not_used.py
```

This constructs a perfectly decodable carry lane that is disconnected from the
answer, then erases it to demonstrate the difference between representation and
causal use.

Run Lesson 8:

```bash
python learnings/labs/neural_gpu_08_causal_vs_terminal.py
```

This compares terminal-only and causally consumed message paths using the tens
column of `38²`, including the gradient path back to the message.

Run Lesson 9:

```bash
python learnings/labs/neural_gpu_09_learning_messages_from_final_loss.py
```

This trains an unlabeled message writer using only a later answer target. It
shows why causal consumption creates a credit path without explicitly
supervising the message as carry.

Run Lesson 10:

```bash
python learnings/labs/neural_gpu_10_shortcuts_and_identifiability.py
```

This adds a second shortcut path. Several seeds reach perfect final loss while
learning different—and usually unintended—internal message decompositions.

Run Lesson 11:

```bash
python learnings/labs/neural_gpu_11_regularization_is_not_semantics.py
```

This gives the shortcut path greater leverage and shows that weight decay
prefers small parameters, not integer weights or intended message semantics.

Run Lesson 12:

```bash
python learnings/labs/neural_gpu_12_shortcut_fails_ood.py
```

This creates a feature that perfectly predicts training labels but reverses on
OOD data, then compares a two-path model with one forced through the invariant
algorithm feature.

Run Lesson 13:

```bash
python learnings/labs/neural_gpu_13_product_formation.py
```

This decomposes a two-digit square into product columns and compares an affine
digit mixer with a model containing a generic bilinear interaction.

Run Lesson 14:

```bash
python learnings/labs/neural_gpu_14_accumulation_vs_carry.py
```

This constructs two different broken squares: one misses cross-product
accumulation while carrying correctly; the other forms products correctly but
does not propagate carry.

Run Lesson 15:

```bash
python learnings/labs/neural_gpu_15_modular_reduction.py
```

This holds a square fixed while changing `N`, exposes quotient-boundary changes,
and compares correct linear-time repeated subtraction with a shifted schedule.

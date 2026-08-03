---
name: data-inspector
description: Inspect and explore competition datasets. Look at data shapes, distributions, edge cases, and generated samples. Use when the user wants to "look at the data", understand what the model sees, check for label issues, or explore the problem structure visually.
---

# Data Inspector

## Goal

Make the data visible — not just as shapes in a config file, but as actual numbers the model processes. Every research question starts here.

## What to do

### 1. Find the data

Data generators live in `competition/data/`. Key files:
- `competition/data/factory.py` — dispatches by task
- `competition/data/squaring_mod.py` — the modular squaring task
- `competition/data/config.py` — task configs (T, N, K values)

### 2. Generate a sample

```bash
cd competition && python -c "
from data.factory import create_task
task = create_task('squaring_mod', n=323, k=2)
batch = task.generate_batch(4)
print('x shape:', batch.x.shape)
print('y shape:', batch.y.shape)
print('x[0]:', batch.x[0])
print('y[0]:', batch.y[0])
"
```

### 3. Show the user, don't just compute

When reporting data to the user:
- Show actual numbers, not just shapes
- Describe what patterns you see
- Flag anything surprising (all zeros? repeats? weird distributions?)
- Use `studio_repl_send` for quick interactive exploration

### 4. Check common failure modes

- **OOD length**: generate T=4 vs T=8, compare distributions
- **Digit distribution**: are digits uniform? Any bias?
- **Carry patterns**: how often do carries occur?
- **Answer entropy**: for small N, are there few unique answers?

### 5. Write findings

If you discover something notable → write to `learnings/concepts/` with a citation to the exact data sample that shows it.

## Key questions this skill answers

- "What does the input actually look like?"
- "Is there a pattern in the data the model could exploit?"
- "What changes between Easy and Hard?"
- "Show me a worked example of the recurrence."
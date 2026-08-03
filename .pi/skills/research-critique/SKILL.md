---
name: research-critique
description: Get multiple perspectives on a research direction by asking different models the same question and comparing their critiques. Use when the user wants to "ask multiple models", "get a second opinion", "critique this approach", or evaluate a research idea from different angles.
---

# Multi-Model Research Critique

## Why

One model tells you what you want to hear. Three models disagree and expose blind spots. The disagreement is the signal.

## How (since pi doesn't natively multi-model)

### Method A: Sequential switching (simplest)

1. Ask the current model: *"Here's my research direction. Critique it. Find the weakest assumption."*
2. Press Ctrl+P to cycle to a different model
3. Ask the same question. Note where they agree and disagree.
4. Cycle to a third model. Same question.
5. Synthesize: what did all three flag? What did only one notice?

### Method B: Fork per model

1. Ask model A for critique
2. `/fork` to create a branch
3. Switch models, ask model B
4. `/fork` again
5. Switch models, ask model C
6. Navigate branches with `/tree` to compare

Method A is faster. Method B preserves each model's response cleanly.

### Method C: Sub-agent delegation (experimental)

When pi adds sub-agent support with model selection, delegate to sub-agents with different models. Not yet practical.

## Recommended model trio

| Role | Model | Why |
|------|-------|-----|
| Depth | `deepseek-v4-pro` | Best reasoning for mechanism critique |
| Breadth | `gpt-5.6-luna` or `gpt-5.5-pro` | Different training, spots different gaps |
| Skeptic | `kimi-k3` or `inkling` | Third architecture, often contrarian |

## What to ask

Not "is this a good idea?" — too vague. Instead:

1. *"What is the weakest assumption in this approach?"*
2. *"Under what conditions would this fail completely?"*
3. *"What simpler experiment would falsify this faster?"*
4. *"What prior work contradicts this direction?"*

## Output

After the round, produce a short synthesis:
- **Agreement**: what all models flagged
- **Disagreement**: where they split
- **Blind spot**: what none of them mentioned
- **Decision**: proceed, modify, or abandon

Don't trust any single model. Trust the intersection of their critiques.
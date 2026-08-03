# GPU Run Analysis — L40S Local Experiments

![wd](/Users/eadebayo/Developer/one%20somn%20deeper/solving/experiments/figures/gpu_wd_grokking.png)

![budget](/Users/eadebayo/Developer/one%20somn%20deeper/solving/experiments/figures/gpu_budget_scale.png)

![medium](/Users/eadebayo/Developer/one%20somn%20deeper/solving/experiments/figures/medium_pregrok_plateau.png)

## What this means

- Weight decay is the grokking knob — wd=0.1 generalizes, wd=1.0 never fits
- Double GPU budget = higher peak, same collapse to overfit
- Medium stops 6k steps before the grokking transition at ~64k

The L40S is the only place to study grokking properly. Competition clocks stop before the interesting thing happens.
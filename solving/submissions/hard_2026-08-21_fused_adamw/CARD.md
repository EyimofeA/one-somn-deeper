# Deadline fused AdamW candidate

Full-model, recurrence-agnostic binary work state. There is no squarer,
reducer, or arithmetic intermediate target. One change from the prior fused
Muon Hard model: use constant AdamW to avoid the observed long-run Muon
collapse.

## Deadline selection

| Candidate | E5 test | E5 OOD | Mean | Updates | Final train loss |
|---|---:|---:|---:|---:|---:|
| AdamW 3e-4 | 0.20% | 1.00% | **0.60%** | 1,261 | 0.611 |
| AdamW 1e-3 | 0.50% | 0.20% | 0.30% | 1,284 | 0.463 |

The faster optimizer fit training better but generalized worse, so the exact
3e-4 source was restored. Its SHA-1 is
`8db168bef3847543c5ce8fb1ee545abd0ac868a6`.

- Easy 3e-4: `c09b14f7-f445-4ca7-b765-cb9ec6ea7f7e`
- Easy 1e-3: `771b4b7b-abea-4383-aa34-ba492600f52d`
- Hard H1: `9e4e7618-55a9-4524-bb4b-e2c45e37db8b` (running)

# Scripts

| File | Role |
|------|------|
| `osmn` | CLI: `gpu start` / `gpu kill` / `gpu status` for rented boxes |
| `gpu_box/` | Implementation of GPU bootstrap + cleanup |
| `extrapolation_curve.py` | **Frozen measurement** once written — every card’s `curve.png` comes from this file only. See `RESEARCH_PROTOCOL.md` §9. |

## GPU box (`osmn`)

```bash
./scripts/osmn gpu start ubuntu@IP
./scripts/osmn gpu kill --wipe    # processes + wipe remote checkout; does not stop billing
./scripts/osmn gpu status
```

Agent skill: `.cursor/skills/osmn-gpu-box/`. Docs: `solving/experiments/OPS.md`.

Do not regenerate the plotting logic after freeze. Fix bugs with a versioned note in RESEARCH_LOG if unavoidable.

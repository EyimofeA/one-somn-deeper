"""Separate local transition fidelity from self-fed rollout drift."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from train_teacher_depth_reducer import DEPTHS, N_VALUE, N_WIDTH, STATE_WIDTH, ReducerCell, digits


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # Reconstruct the registered held-out remainder split exactly.
    train_remainders = set(random.Random(args.seed).sample(range(N_VALUE), 800))
    heldout = random.Random(args.seed + 1).sample(
        [r for r in range(N_VALUE) if r not in train_remainders], 256
    )
    model = ReducerCell().to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.eval()
    n = torch.tensor([digits(N_VALUE, N_WIDTH)] * len(heldout), dtype=torch.long, device=args.device)
    report: dict[str, list[dict]] = {}
    for depth in (5, 10, 50, 100):
        free = torch.tensor(
            [digits(r + depth * N_VALUE, STATE_WIDTH) for r in heldout], dtype=torch.long, device=args.device
        )
        rows = []
        for step in range(1, depth + 1):
            remaining = depth - step + 1
            true_input = torch.tensor(
                [digits(r + remaining * N_VALUE, STATE_WIDTH) for r in heldout], dtype=torch.long, device=args.device
            )
            target = torch.tensor(
                [digits(r + (remaining - 1) * N_VALUE, STATE_WIDTH) for r in heldout], dtype=torch.long, device=args.device
            )
            teacher = model(n, true_input).argmax(dim=-1)
            free = model(n, free).argmax(dim=-1)
            teacher_correct = teacher == target
            free_correct = free == target
            rows.append({
                "step": step,
                "teacher_exact": float(teacher_correct.all(dim=-1).float().mean()),
                "teacher_token": float(teacher_correct.float().mean()),
                "free_exact": float(free_correct.all(dim=-1).float().mean()),
                "free_token": float(free_correct.float().mean()),
            })
        report[str(depth)] = rows
    out = Path(args.out)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

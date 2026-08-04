"""Aggregate autonomous learned-halting metrics across quotient-depth buckets."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from train_teacher_depth_reducer import N_VALUE, N_WIDTH, STATE_WIDTH, ReducerCell, digits


@torch.no_grad()
def measure(model: ReducerCell, remainders: list[int], depth: int, device: str, max_steps: int) -> dict:
    n = torch.tensor([digits(N_VALUE, N_WIDTH)] * len(remainders), dtype=torch.long, device=device)
    state = torch.tensor([digits(r + depth * N_VALUE, STATE_WIDTH) for r in remainders], dtype=torch.long, device=device)
    target = torch.tensor([digits(r, STATE_WIDTH) for r in remainders], dtype=torch.long, device=device)
    active = torch.ones(len(remainders), dtype=torch.bool, device=device)
    stopped_at = torch.full((len(remainders),), -1, dtype=torch.long, device=device)
    for iteration in range(max_steps + 1):
        stop = model.stop_logits(n, state).sigmoid() >= 0.5
        newly_stopped = active & stop
        stopped_at[newly_stopped] = iteration
        active = active & ~stop
        if not active.any() or iteration == max_steps:
            break
        next_state = model(n, state).argmax(dim=-1)
        state = torch.where(active[:, None], next_state, state)
    exact = (state == target).all(dim=-1)
    correct_depth = stopped_at == depth
    return {
        "remainder_exact": float(exact.float().mean()),
        "halting_accuracy": float(correct_depth.float().mean()),
        "mean_iterations": float(torch.where(stopped_at < 0, torch.full_like(stopped_at, max_steps), stopped_at).float().mean()),
        "stopped_early": float((stopped_at.ge(0) & (stopped_at < depth)).float().mean()),
        "stopped_late": float((stopped_at > depth).float().mean()),
        "failed_to_stop": float((stopped_at < 0).float().mean()),
        "wrong_remainder_after_correct_depth": float((correct_depth & ~exact).float().mean()),
    }


def mean(rows: list[dict]) -> dict:
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-depth", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    train = set(random.Random(args.seed).sample(range(N_VALUE), 800))
    heldout = random.Random(args.seed + 1).sample([r for r in range(N_VALUE) if r not in train], 256)
    model = ReducerCell(with_stop_head=True).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.eval()
    by_depth = {str(q): measure(model, heldout, q, args.device, args.max_depth + 20) for q in range(args.max_depth + 1)}
    groups = {"q=0": [0], "q=1": [1], "q=2-3": [2, 3], "q=4-9": list(range(4, 10)), "q=10-99": list(range(10, 100)), "q=100": [100]}
    report = {"by_depth": by_depth, "by_quotient_bucket": {name: mean([by_depth[str(q)] for q in qs]) for name, qs in groups.items()}}
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["by_quotient_bucket"]), flush=True)


if __name__ == "__main__":
    main()

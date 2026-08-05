"""Aggregate autonomous learned-halting metrics across quotient-depth buckets."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from train_teacher_depth_reducer import N_VALUE, N_WIDTH, STATE_WIDTH, ReducerCell, digits


@torch.no_grad()
def measure(model: ReducerCell, n_value: int, remainders: list[int], depth: int, device: str, max_steps: int) -> dict:
    n = torch.tensor([digits(n_value, N_WIDTH)] * len(remainders), dtype=torch.long, device=device)
    state = torch.tensor([digits(r + depth * n_value, STATE_WIDTH) for r in remainders], dtype=torch.long, device=device)
    target = torch.tensor([digits(r, STATE_WIDTH) for r in remainders], dtype=torch.long, device=device)
    one_step_target = torch.tensor(
        [digits(r + max(depth - 1, 0) * n_value, STATE_WIDTH) for r in remainders],
        dtype=torch.long, device=device,
    )
    teacher = model(n, state).argmax(dim=-1)
    teacher_correct = teacher == one_step_target
    q_known_state = state.clone()
    for _ in range(depth):
        q_known_state = model(n, q_known_state).argmax(dim=-1)
    q_known_exact = (q_known_state == target).all(dim=-1)
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
        "teacher_one_step_exact": float(teacher_correct.all(dim=-1).float().mean()),
        "teacher_one_step_token": float(teacher_correct.float().mean()),
        "q_known_remainder_exact": float(q_known_exact.float().mean()),
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
    ap.add_argument("--min-depth", type=int, default=0)
    ap.add_argument("--n-values", type=int, nargs="+", default=[N_VALUE])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    model = ReducerCell(with_stop_head=True).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.eval()
    if args.min_depth < 0 or args.min_depth > args.max_depth:
        raise ValueError("invalid depth range")
    depths = list(range(args.min_depth, args.max_depth + 1))
    by_modulus = {}
    for index, n_value in enumerate(args.n_values):
        train = set(random.Random(args.seed + 10_000 * index).sample(range(n_value), 800))
        heldout = random.Random(args.seed + 1 + 10_000 * index).sample(
            [r for r in range(n_value) if r not in train], 256
        )
        by_depth = {
            str(q): measure(model, n_value, heldout, q, args.device, args.max_depth + 20)
            for q in depths
        }
        by_modulus[str(n_value)] = {
            "by_depth": by_depth,
            "by_quotient_bucket": {f"q={args.min_depth}-{args.max_depth}": mean(list(by_depth.values()))},
        }
    report = {"by_modulus": by_modulus}
    if len(args.n_values) == 1:
        report.update(by_modulus[str(args.n_values[0])])
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["by_modulus"]), flush=True)


if __name__ == "__main__":
    main()

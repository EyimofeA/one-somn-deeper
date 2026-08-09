"""Research-only learned shifted-divisor reduction gate.

The learned forward passes contain no arithmetic. Python arithmetic constructs
and scores supervised examples. The fixed high-to-low shift evaluator makes
this a mechanism diagnostic, not submission-legal evidence.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from train_comparator_reducer import SerialComparator
from train_serial_subtractor import WIDTH, SerialSubtractor, digits, semiprimes


def digit_row(value: int) -> list[int]:
    if value >= 10**WIDTH:
        raise ValueError(f"{value} exceeds SERIAL_WIDTH={WIDTH}")
    return digits(value)[::-1]


def transition_rows(moduli: list[int], seed: int, residuals: int, max_shift: int):
    rows = []
    for modulus_index, modulus in enumerate(moduli):
        for shift in range(max_shift + 1):
            divisor = modulus * 10**shift
            rng = random.Random(seed + 100_003 * modulus_index + 997 * shift)
            for k in range(10):
                for _ in range(residuals):
                    residual = rng.randrange(divisor)
                    state = k * divisor + residual
                    target = state if k == 0 else state - divisor
                    rows.append((digit_row(state), digit_row(divisor), digit_row(target), int(k > 0), shift, k))
    random.Random(seed + 71).shuffle(rows)
    return rows


def tensor_batch(rows, device: str):
    state = torch.tensor([row[0] for row in rows], dtype=torch.long, device=device)
    divisor = torch.tensor([row[1] for row in rows], dtype=torch.long, device=device)
    target = torch.tensor([row[2] for row in rows], dtype=torch.long, device=device)
    label = torch.tensor([row[3] for row in rows], dtype=torch.float32, device=device)
    return state, divisor, target, label


@torch.no_grad()
def transition_metrics(comparator, subtractor, rows, device: str, batch_size: int = 4096):
    totals = {shift: {"n": 0, "comparator": 0, "subtract_n": 0, "subtract": 0} for shift in sorted({row[4] for row in rows})}
    for offset in range(0, len(rows), batch_size):
        chunk = rows[offset : offset + batch_size]
        state, divisor, target, _ = tensor_batch(chunk, device)
        compare = comparator(state, divisor).sigmoid() >= 0.5
        subtract = subtractor(state, divisor).argmax(dim=-1)
        exact = (subtract == target).all(dim=-1)
        for index, row in enumerate(chunk):
            bucket = totals[row[4]]
            bucket["n"] += 1
            bucket["comparator"] += int(compare[index].item() == bool(row[3]))
            if row[3]:
                bucket["subtract_n"] += 1
                bucket["subtract"] += int(exact[index].item())
    return {
        str(shift): {
            "examples": bucket["n"],
            "comparator_accuracy": bucket["comparator"] / bucket["n"],
            "subtraction_exact": bucket["subtract"] / bucket["subtract_n"],
        }
        for shift, bucket in totals.items()
    }


@torch.no_grad()
def autonomous_metrics(comparator, subtractor, moduli: list[int], seed: int, max_shift: int, per_modulus: int, device: str):
    quotients = (0, 1, 9, 10, 99, 100, 999, 1_000, 9_999, 999_999, 99_999_999)
    report = {}
    for quotient in quotients:
        states, divisors_by_shift, targets = [], [[] for _ in range(max_shift + 1)], []
        for modulus_index, modulus in enumerate(moduli):
            rng = random.Random(seed + 900_001 * modulus_index + quotient)
            for _ in range(per_modulus):
                remainder = rng.randrange(modulus)
                states.append(digit_row(quotient * modulus + remainder))
                targets.append(digit_row(remainder))
                for shift in range(max_shift + 1):
                    divisors_by_shift[shift].append(digit_row(modulus * 10**shift))
        state = torch.tensor(states, dtype=torch.long, device=device)
        target = torch.tensor(targets, dtype=torch.long, device=device)
        applied = torch.zeros(len(states), dtype=torch.long, device=device)
        for shift in range(max_shift, -1, -1):
            divisor = torch.tensor(divisors_by_shift[shift], dtype=torch.long, device=device)
            for _ in range(9):
                active = comparator(state, divisor).sigmoid() >= 0.5
                candidate = subtractor(state, divisor).argmax(dim=-1)
                state = torch.where(active[:, None], candidate, state)
                applied += active.long()
        report[str(quotient)] = {
            "examples": len(states),
            "remainder_exact": float((state == target).all(dim=-1).float().mean()),
            "scheduled_comparator_calls": 9 * (max_shift + 1),
            "mean_subtractions_selected": float(applied.float().mean()),
            "expected_subtractions": sum(map(int, str(quotient))),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--residuals", type=int, default=16)
    parser.add_argument("--max-shift", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if WIDTH < 14:
        raise ValueError("run with SERIAL_WIDTH=14")

    all_moduli = semiprimes(args.seed, 64)
    train_moduli, test_moduli = all_moduli[:48], all_moduli[48:]
    train = transition_rows(train_moduli, args.seed, args.residuals, args.max_shift)
    unseen = transition_rows(test_moduli, args.seed + 1_000_000, args.residuals, args.max_shift)
    comparator = SerialComparator().to(args.device)
    subtractor = SerialSubtractor().to(args.device)
    weights = torch.load(args.init_checkpoint, map_location=args.device, weights_only=True)
    comparator.load_state_dict(weights["comparator"])
    subtractor.load_state_dict(weights["subtractor"])
    optimizer = torch.optim.AdamW(list(comparator.parameters()) + list(subtractor.parameters()), lr=3e-4, weight_decay=0.01)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "classification": "RESEARCH-ONLY; fixed shifted-divisor schedule requires Rule-7 audit",
        "width": WIDTH, "train_moduli": train_moduli, "test_moduli": test_moduli,
        "max_shift": args.max_shift, "residuals_per_quotient_digit": args.residuals,
        "steps": args.steps, "batch_size": args.batch_size,
        "init_checkpoint": args.init_checkpoint, "train_examples": len(train), "unseen_examples": len(unseen),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    initial = {
        "transition": transition_metrics(comparator, subtractor, unseen, args.device),
        "autonomous": autonomous_metrics(comparator, subtractor, test_moduli, args.seed + 2_000_000, args.max_shift, 16, args.device),
    }
    (out / "initial_report.json").write_text(json.dumps(initial, indent=2) + "\n")
    print(json.dumps({"initial": initial}), flush=True)

    start = time.perf_counter()
    with (out / "metrics.jsonl").open("w") as log:
        for step in range(1, args.steps + 1):
            offset = ((step - 1) * args.batch_size) % len(train)
            batch = [train[(offset + index) % len(train)] for index in range(args.batch_size)]
            state, divisor, target, label = tensor_batch(batch, args.device)
            comparator_logits = comparator(state, divisor)
            positive = label.bool()
            subtraction_logits = subtractor(state[positive], divisor[positive])
            loss_compare = F.binary_cross_entropy_with_logits(comparator_logits, label)
            loss_subtract = F.cross_entropy(subtraction_logits.reshape(-1, 10), target[positive].reshape(-1))
            loss = loss_compare + loss_subtract
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(comparator.parameters()) + list(subtractor.parameters()), 1.0)
            optimizer.step()
            if step % 400 == 0 or step == args.steps:
                record = {"step": step, "loss": float(loss.detach()), "compare_loss": float(loss_compare.detach()), "subtract_loss": float(loss_subtract.detach()), "steps_per_sec": step / (time.perf_counter() - start)}
                log.write(json.dumps(record) + "\n"); log.flush(); print(json.dumps(record), flush=True)

    comparator.eval(); subtractor.eval()
    report = {
        "transition": transition_metrics(comparator, subtractor, unseen, args.device),
        "autonomous": autonomous_metrics(comparator, subtractor, test_moduli, args.seed + 2_000_000, args.max_shift, 64, args.device),
        "elapsed_seconds": time.perf_counter() - start,
        "parameters": sum(p.numel() for p in comparator.parameters()) + sum(p.numel() for p in subtractor.parameters()),
    }
    (out / "eval_report.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save({"comparator": comparator.state_dict(), "subtractor": subtractor.state_dict()}, out / "reducer.pt")
    print(json.dumps({"final": report}), flush=True)


if __name__ == "__main__":
    main()

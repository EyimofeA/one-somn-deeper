"""Boundary-only repair for the research shifted long-division comparator."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from train_comparator_reducer import SerialComparator
from train_serial_subtractor import WIDTH, SerialSubtractor, semiprimes
from train_shifted_long_division_reducer import autonomous_metrics, digit_row, transition_metrics, transition_rows


def boundary_rows(moduli: list[int], seed: int, residuals: int, max_shift: int):
    rows = []
    for modulus_index, modulus in enumerate(moduli):
        for shift in range(max_shift + 1):
            divisor = modulus * 10**shift
            rng = random.Random(seed + 200_003 * modulus_index + 991 * shift)
            for _ in range(residuals):
                # Long division's leading-zero case is D-N+r, not a uniform
                # draw from [0,D); it becomes vanishingly rare at large p.
                negative = divisor - modulus + rng.randrange(modulus)
                positive = divisor + rng.randrange(modulus)
                rows.append((digit_row(negative), digit_row(divisor), 0))
                rows.append((digit_row(positive), digit_row(divisor), 1))
    random.Random(seed + 23).shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--residuals", type=int, default=128)
    parser.add_argument("--max-shift", type=int, default=8)
    parser.add_argument("--uniform-rehearsal", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if WIDTH < 14:
        raise ValueError("run with SERIAL_WIDTH=14")

    moduli = semiprimes(args.seed, 64)
    train_moduli, test_moduli = moduli[:48], moduli[48:]
    train = boundary_rows(train_moduli, args.seed, args.residuals, args.max_shift)
    if args.uniform_rehearsal:
        uniform = transition_rows(train_moduli, args.seed, max(16, args.residuals // 4), args.max_shift)
        train.extend((row[0], row[1], row[3]) for row in uniform)
        random.Random(args.seed + 97).shuffle(train)
    unseen_transitions = transition_rows(test_moduli, args.seed + 1_000_000, 16, args.max_shift)
    comparator = SerialComparator().to(args.device)
    subtractor = SerialSubtractor().to(args.device)
    weights = torch.load(args.init_checkpoint, map_location=args.device, weights_only=True)
    comparator.load_state_dict(weights["comparator"])
    subtractor.load_state_dict(weights["subtractor"])
    subtractor.eval()
    for parameter in subtractor.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(comparator.parameters(), lr=1e-4, weight_decay=0.01)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({
        "classification": "RESEARCH-ONLY; boundary repair; subtractor frozen",
        "steps": args.steps, "batch_size": args.batch_size, "residuals": args.residuals,
        "max_shift": args.max_shift, "init_checkpoint": args.init_checkpoint,
        "uniform_rehearsal": args.uniform_rehearsal,
        "train_examples": len(train), "boundary_definition": "negative=D-N+r; positive=D+r",
    }, indent=2) + "\n")
    start = time.perf_counter()
    with (out / "metrics.jsonl").open("w") as log:
        for step in range(1, args.steps + 1):
            offset = ((step - 1) * args.batch_size) % len(train)
            batch = [train[(offset + index) % len(train)] for index in range(args.batch_size)]
            state = torch.tensor([row[0] for row in batch], dtype=torch.long, device=args.device)
            divisor = torch.tensor([row[1] for row in batch], dtype=torch.long, device=args.device)
            label = torch.tensor([row[2] for row in batch], dtype=torch.float32, device=args.device)
            loss = F.binary_cross_entropy_with_logits(comparator(state, divisor), label)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(comparator.parameters(), 1.0); optimizer.step()
            if step % 200 == 0 or step == args.steps:
                record = {"step": step, "loss": float(loss.detach()), "steps_per_sec": step / (time.perf_counter() - start)}
                log.write(json.dumps(record) + "\n"); log.flush(); print(json.dumps(record), flush=True)
    comparator.eval()
    report = {
        "transition": transition_metrics(comparator, subtractor, unseen_transitions, args.device),
        "autonomous": autonomous_metrics(comparator, subtractor, test_moduli, args.seed + 2_000_000, args.max_shift, 64, args.device),
        "elapsed_seconds": time.perf_counter() - start,
    }
    (out / "eval_report.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save({"comparator": comparator.state_dict(), "subtractor": subtractor.state_dict()}, out / "reducer.pt")
    print(json.dumps({"final": report}), flush=True)


if __name__ == "__main__":
    main()

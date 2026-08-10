"""Carry-supervised follow-up for the frozen multi-lane Neural GPU.

The backbone, data split, main digit loss, optimizer, and update budget match
the answer-only baseline. The only intervention is an auxiliary linear head
and MSE loss for normalized carry-in/carry-out at each output column.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from train_multilane_neural_gpu_square import (
    MultiLaneNeuralGPU,
    evaluate,
    make_split,
    tensors,
)


def raw_carries(rows: list[tuple]) -> torch.Tensor:
    targets = []
    for _, _, target_digits, value in rows:
        source = [int(character) for character in f"{value:04d}"][::-1]
        carry = 0
        columns = []
        emitted = []
        for position in range(8):
            diagonal = sum(
                source[left] * source[right]
                for left in range(4)
                for right in range(4)
                if left + right == position
            )
            carry_in = carry
            total = diagonal + carry_in
            emitted.append(total % 10)
            carry = total // 10
            columns.append((carry_in, carry))
        assert carry == 0
        assert emitted == target_digits
        targets.append(columns)
    return torch.tensor(targets, dtype=torch.float32)


class CarrySupervisedNeuralGPU(MultiLaneNeuralGPU):
    def __init__(self, tape_width: int, hidden: int, lanes: int, microsteps: int):
        super().__init__(tape_width, hidden, lanes, microsteps)
        self.carry_head = nn.Linear(hidden, 2)

    def forward_with_carry(self, x_digits: torch.Tensor, n_digits: torch.Tensor):
        features = self.forward_features(x_digits, n_digits)
        return self.head(features), self.carry_head(features)


@torch.no_grad()
def frozen_audit(model, rows, carries, mean, std, device):
    model.eval()
    correct_by_position = torch.zeros(model.tape_width, device=device)
    carry_sq_error = 0.0
    count = 0
    for start in range(0, len(rows), 1024):
        chunk = rows[start : start + 1024]
        x, n, target = tensors(chunk, device)
        logits, carry_prediction = model.forward_with_carry(x, n)
        correct_by_position += (logits.argmax(-1) == target).sum(0)
        normalized = (carries[start : start + len(chunk)].to(device) - mean) / std
        carry_sq_error += float(F.mse_loss(carry_prediction, normalized, reduction="sum"))
        count += normalized.numel()
    return {
        "digit_accuracy_lsd_to_msd": [
            float(value / len(rows)) for value in correct_by_position
        ],
        "carry_mse_normalized": carry_sq_error / count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=20260810)
    parser.add_argument("--input-width", type=int, default=4)
    parser.add_argument("--tape-width", type=int, default=8)
    parser.add_argument("--train-values", type=int, default=8000)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lanes", type=int, default=6)
    parser.add_argument("--microsteps", type=int, default=16)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--carry-weight", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    train_rows, test_rows = make_split(
        args.split_seed, args.input_width, args.tape_width, args.train_values
    )
    train_carries = raw_carries(train_rows)
    test_carries = raw_carries(test_rows)
    carry_mean = train_carries.mean(dim=(0, 1)).to(args.device)
    carry_std = train_carries.std(dim=(0, 1)).clamp_min(1e-6).to(args.device)

    model = CarrySupervisedNeuralGPU(
        args.tape_width, args.hidden, args.lanes, args.microsteps
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    train_x, train_n, train_target = tensors(train_rows, args.device)
    normalized_train_carries = (
        train_carries.to(args.device) - carry_mean
    ) / carry_std

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "metrics.jsonl").open("w") as metrics_file:
        for step in range(1, args.steps + 1):
            model.train()
            indices = torch.randint(
                len(train_rows), (args.batch_size,), generator=generator
            ).to(args.device)
            logits, carry_prediction = model.forward_with_carry(
                train_x[indices], train_n[indices]
            )
            main_loss = F.cross_entropy(
                logits.flatten(0, 1), train_target[indices].flatten()
            )
            carry_loss = F.mse_loss(
                carry_prediction, normalized_train_carries[indices]
            )
            loss = main_loss + args.carry_weight * carry_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                record = {
                    "step": step,
                    "total_loss": float(loss.detach()),
                    "main_loss": float(main_loss.detach()),
                    "carry_loss": float(carry_loss.detach()),
                    "train": evaluate(model, train_rows, args.device),
                    "unseen_x": evaluate(model, test_rows, args.device),
                }
                metrics_file.write(json.dumps(record) + "\n")
                metrics_file.flush()
                print(json.dumps(record), flush=True)

    report = {
        "seed": args.seed,
        "split_seed": args.split_seed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "carry_weight": args.carry_weight,
        "carry_mean": carry_mean.tolist(),
        "carry_std": carry_std.tolist(),
        "metrics": {
            "train": evaluate(model, train_rows, args.device),
            "unseen_x": evaluate(model, test_rows, args.device),
        },
        "audit": {
            "train": frozen_audit(
                model, train_rows, train_carries, carry_mean, carry_std, args.device
            ),
            "unseen_x": frozen_audit(
                model, test_rows, test_carries, carry_mean, carry_std, args.device
            ),
        },
    }
    (out / "eval_report.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save(model.state_dict(), out / "final.pt")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

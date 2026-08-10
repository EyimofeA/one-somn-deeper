"""Generic multi-lane local grid capability test for raw decimal squaring.

The model forward contains only learned local mixing, lane mixing, tied GRU
updates, and digit decoding. Arithmetic is used only to construct labels.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def digits(value: int, width: int) -> list[int]:
    return [int(character) for character in f"{value:0{width}d}"][::-1]


def make_split(seed: int, input_width: int, tape_width: int, train_values: int):
    values = list(range(10**input_width))
    random.Random(seed).shuffle(values)
    train_x, test_x = values[:train_values], values[train_values:]

    def rows(xs):
        return [
            (
                digits(value, input_width) + [0] * (tape_width - input_width),
                [0] * tape_width,
                digits(value * value, tape_width),
                value,
            )
            for value in xs
        ]

    return rows(train_x), rows(test_x)


class LocalGridCell(nn.Module):
    def __init__(self, hidden: int, lanes: int):
        super().__init__()
        self.left = nn.Linear(hidden, hidden, bias=False)
        self.self_ = nn.Linear(hidden, hidden, bias=False)
        self.right = nn.Linear(hidden, hidden, bias=False)
        self.lane_mix = nn.Linear(lanes, lanes, bias=False)
        self.norm = nn.LayerNorm(hidden)
        self.gru = nn.GRUCell(hidden, hidden)

    def forward(self, state: torch.Tensor, immutable: torch.Tensor) -> torch.Tensor:
        left = F.pad(state[:, :-1], (0, 0, 0, 0, 1, 0))
        right = F.pad(state[:, 1:], (0, 0, 0, 0, 0, 1))
        local = self.left(left) + self.self_(state) + self.right(right)
        across_lanes = self.lane_mix(state.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        update = self.norm(local + across_lanes + immutable)
        shape = state.shape
        return self.gru(update.reshape(-1, shape[-1]), state.reshape(-1, shape[-1])).reshape(shape)


class MultiLaneNeuralGPU(nn.Module):
    def __init__(self, tape_width: int, hidden: int, lanes: int, microsteps: int):
        super().__init__()
        self.tape_width = tape_width
        self.hidden = hidden
        self.lanes = lanes
        self.microsteps = microsteps
        self.digit = nn.Embedding(10, hidden)
        self.x_inject = nn.Linear(hidden, lanes * hidden)
        self.n_inject = nn.Linear(hidden, lanes * hidden)
        self.roles = nn.Parameter(torch.randn(lanes, hidden) * 0.02)
        self.boundaries = nn.Parameter(torch.randn(2, lanes, hidden) * 0.02)
        self.cell = LocalGridCell(hidden, lanes)
        self.output_norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 10)

    def forward_features(self, x_digits: torch.Tensor, n_digits: torch.Tensor) -> torch.Tensor:
        batch, positions = x_digits.shape
        x = self.x_inject(self.digit(x_digits)).reshape(batch, positions, self.lanes, self.hidden)
        n = self.n_inject(self.digit(n_digits)).reshape(batch, positions, self.lanes, self.hidden)
        immutable = x + n + self.roles[None, None]
        immutable = immutable.clone()
        immutable[:, 0] = immutable[:, 0] + self.boundaries[0]
        immutable[:, -1] = immutable[:, -1] + self.boundaries[1]
        state = torch.zeros_like(immutable)
        for _ in range(self.microsteps):
            state = self.cell(state, immutable)
        return self.output_norm(state[:, :, 0])

    def forward(self, x_digits: torch.Tensor, n_digits: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x_digits, n_digits))


def tensors(rows, device):
    return (
        torch.tensor([row[0] for row in rows], dtype=torch.long, device=device),
        torch.tensor([row[1] for row in rows], dtype=torch.long, device=device),
        torch.tensor([row[2] for row in rows], dtype=torch.long, device=device),
    )


@torch.no_grad()
def evaluate(model, rows, device, batch_size=1024):
    model.eval()
    exact = digits_correct = examples = 0
    loss_sum = 0.0
    for start in range(0, len(rows), batch_size):
        x, n, target = tensors(rows[start : start + batch_size], device)
        logits = model(x, n)
        prediction = logits.argmax(-1)
        exact += int((prediction == target).all(-1).sum())
        digits_correct += int((prediction == target).sum())
        examples += target.shape[0]
        loss_sum += float(F.cross_entropy(logits.flatten(0, 1), target.flatten(), reduction="sum"))
    return {
        "examples": examples,
        "exact": exact,
        "exact_accuracy": exact / examples,
        "digit_accuracy": digits_correct / (examples * model.tape_width),
        "cross_entropy": loss_sum / (examples * model.tape_width),
    }


def main():
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
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    train_rows, test_rows = make_split(
        args.split_seed, args.input_width, args.tape_width, args.train_values
    )
    model = MultiLaneNeuralGPU(
        args.tape_width, args.hidden, args.lanes, args.microsteps
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    train_x, train_n, train_target = tensors(train_rows, args.device)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "metrics.jsonl").open("w") as metrics_file:
        for step in range(1, args.steps + 1):
            model.train()
            indices = torch.randint(
                len(train_rows), (args.batch_size,), generator=generator
            ).to(args.device)
            logits = model(train_x[indices], train_n[indices])
            loss = F.cross_entropy(
                logits.flatten(0, 1), train_target[indices].flatten()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                record = {
                    "step": step,
                    "batch_loss": float(loss.detach()),
                    "train": evaluate(model, train_rows, args.device),
                    "unseen_x": evaluate(model, test_rows, args.device),
                }
                metrics_file.write(json.dumps(record) + "\n")
                metrics_file.flush()
                print(json.dumps(record), flush=True)

    report = {
        "seed": args.seed,
        "split_seed": args.split_seed,
        "input_width": args.input_width,
        "tape_width": args.tape_width,
        "train_values": len(train_rows),
        "test_values": len(test_rows),
        "hidden": args.hidden,
        "lanes": args.lanes,
        "microsteps": args.microsteps,
        "steps": args.steps,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "optimizer": {
            "name": "AdamW",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        "metrics": {
            "train": evaluate(model, train_rows, args.device),
            "unseen_x": evaluate(model, test_rows, args.device),
        },
    }
    (out / "eval_report.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save(model.state_dict(), out / "final.pt")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

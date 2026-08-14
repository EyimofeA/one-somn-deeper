"""Decimal adaptation of the user-supplied shared-cell Neural GPU."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from train_transformer_multiplication_factorial import carry_count, split_examples


INPUT_DIGITS, OUTPUT_DIGITS = 2, 4


def digits(value: int, width: int) -> list[int]:
    return [(value // (10**position)) % 10 for position in range(width)]


def rows(examples):
    return [(digits(a, INPUT_DIGITS), digits(b, INPUT_DIGITS),
             digits(a * b, OUTPUT_DIGITS), a, b) for a, b in examples]


class ConvGRUCell2D(nn.Module):
    def __init__(self, channels: int = 128, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.update = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.reset = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.candidate = nn.Conv2d(channels, channels, kernel_size, padding=padding)

    def forward(self, hidden):
        update = torch.sigmoid(self.update(hidden))
        reset = torch.sigmoid(self.reset(hidden))
        candidate = torch.tanh(self.candidate(reset * hidden))
        return (1 - update) * hidden + update * candidate


class DecimalNeuralGPUMultiplier(nn.Module):
    def __init__(self, channels: int = 128, grid_height: int = 4, num_steps: int = 14):
        super().__init__()
        self.channels, self.grid_height, self.num_steps = channels, grid_height, num_steps
        self.digit_embedding = nn.Embedding(10, channels)
        self.left_marker = nn.Parameter(torch.randn(channels) * 0.02)
        self.right_marker = nn.Parameter(torch.randn(channels) * 0.02)
        self.cell = ConvGRUCell2D(channels)
        self.readout = nn.Conv1d(channels, 10, 1)

    def make_initial_state(self, left_digits, right_digits):
        batch = left_digits.shape[0]
        hidden = torch.zeros(batch, self.channels, self.grid_height, OUTPUT_DIGITS,
                             device=left_digits.device)
        hidden[:, :, 0, :INPUT_DIGITS] = (
            self.digit_embedding(left_digits).transpose(1, 2) + self.left_marker[None, :, None]
        )
        hidden[:, :, 1, :INPUT_DIGITS] = (
            self.digit_embedding(right_digits).transpose(1, 2) + self.right_marker[None, :, None]
        )
        return hidden

    def forward(self, left_digits, right_digits):
        hidden = self.make_initial_state(left_digits, right_digits)
        for _ in range(self.num_steps):
            hidden = self.cell(hidden)
        return self.readout(hidden[:, :, 0, :]).transpose(1, 2)


def tensors(items, device):
    return (torch.tensor([row[0] for row in items], device=device),
            torch.tensor([row[1] for row in items], device=device),
            torch.tensor([row[2] for row in items], device=device))


def bucket_report(counts):
    return {str(key): {"correct": value[0], "total": value[1],
                       "accuracy": value[0] / value[1]}
            for key, value in sorted(counts.items(), key=lambda item: str(item[0]))}


@torch.no_grad()
def evaluate(model, data, device, batch_size=512):
    model.eval()
    exact = total = digit_correct = digit_total = 0
    positions = [0] * OUTPUT_DIGITS
    length_buckets = defaultdict(lambda: [0, 0])
    carry_buckets = defaultdict(lambda: [0, 0])
    for start in range(0, len(data), batch_size):
        chunk = data[start:start + batch_size]
        left, right, target = tensors(chunk, device)
        prediction = model(left, right).argmax(-1)
        matches = prediction == target
        exact_rows = matches.all(-1)
        exact += int(exact_rows.sum())
        total += len(chunk)
        digit_correct += int(matches.sum())
        digit_total += matches.numel()
        for position in range(OUTPUT_DIGITS):
            positions[position] += int(matches[:, position].sum())
        for row, correct in zip(chunk, exact_rows.tolist()):
            a, b = row[3], row[4]
            for buckets, key in ((length_buckets, len(str(a * b))),
                                 (carry_buckets, carry_count(a, b))):
                buckets[key][0] += int(correct)
                buckets[key][1] += 1
    return {"exact_numerical_accuracy": exact / total,
            "digit_accuracy": digit_correct / digit_total,
            "digit_accuracy_lsd_first": [value / total for value in positions],
            "by_product_digit_length": bucket_report(length_buckets),
            "by_decimal_carry_columns": bucket_report(carry_buckets),
            "examples": total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_examples, test_examples = split_examples(args.seed)
    train, test = rows(train_examples), rows(test_examples)
    model = DecimalNeuralGPUMultiplier().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    curve = []
    for step in range(1, args.steps + 1):
        indices = torch.randint(len(train), (256,), generator=generator).tolist()
        left, right, target = tensors([train[index] for index in indices], args.device)
        model.train()
        logits = model(left, right)
        loss = F.cross_entropy(logits.reshape(-1, 10), target.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0:
            train_metrics = evaluate(model, train, args.device)
            test_metrics = evaluate(model, test, args.device)
            record = {"step": step, "loss": float(loss.detach()),
                      "train_exact": train_metrics["exact_numerical_accuracy"],
                      "test_exact": test_metrics["exact_numerical_accuracy"]}
            curve.append(record)
            print(json.dumps({"type": "progress", **record}), flush=True)
    report = {"classification": "RESEARCH ONLY - decimal attached Neural GPU multiplication",
              "architecture": {"channels": 128, "grid_height": 4, "kernel": 3,
                               "shared_cgru_layers": 1, "recurrent_steps": 14},
              "representation": "two 2-digit LSD decimal operands in rows 0/1; four decimal outputs",
              "split": "same numeric unordered-pair split as prior baselines",
              "seed": args.seed, "steps": args.steps,
              "parameters": sum(parameter.numel() for parameter in model.parameters()),
              "curve": curve, "train": evaluate(model, train, args.device),
              "test": evaluate(model, test, args.device)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "eval_report.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save(model.state_dict(), args.out / "model.pt")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

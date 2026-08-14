"""Digit-serial GRU multiplication on a commutativity-safe 0..99 split."""
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


WIDTH, INPUT_WIDTH, HIDDEN = 4, 2, 64


def digits(value: int, width: int = WIDTH):
    return [int(character) for character in f"{value:0{width}d}"][::-1]


def rows(examples):
    return [
        (digits(a, INPUT_WIDTH) + [0, 0], digits(b, INPUT_WIDTH) + [0, 0], digits(a * b), a, b)
        for a, b in examples
    ]


class SerialMultiplier(nn.Module):
    def __init__(self):
        super().__init__()
        self.digit = nn.Embedding(10, HIDDEN)
        self.place = nn.Embedding(WIDTH, HIDDEN)
        self.pair = nn.Linear(2 * HIDDEN, HIDDEN)
        self.cell = nn.GRUCell(HIDDEN, HIDDEN)
        self.head = nn.Linear(HIDDEN, 10)

    def forward(self, left, right):
        hidden = torch.zeros(left.shape[0], HIDDEN, device=left.device)
        outputs = []
        for position in range(WIDTH):
            place = self.place.weight[position]
            pair = torch.tanh(self.pair(torch.cat((
                self.digit(left[:, position]) + place,
                self.digit(right[:, position]) + place,
            ), dim=-1)))
            hidden = self.cell(pair, hidden)
            outputs.append(self.head(hidden))
        return torch.stack(outputs, dim=1)


def tensors(items, device):
    return (
        torch.tensor([row[0] for row in items], device=device),
        torch.tensor([row[1] for row in items], device=device),
        torch.tensor([row[2] for row in items], device=device),
    )


def bucket_report(counts):
    return {str(k): {"correct": v[0], "total": v[1], "accuracy": v[0] / v[1]}
            for k, v in sorted(counts.items(), key=lambda item: str(item[0]))}


@torch.no_grad()
def evaluate(model, data, device, batch_size=1024):
    model.eval()
    exact = total = 0
    digit_correct = [0] * WIDTH
    operand_buckets = defaultdict(lambda: [0, 0])
    product_buckets = defaultdict(lambda: [0, 0])
    carry_buckets = defaultdict(lambda: [0, 0])
    predictions = {}
    for start in range(0, len(data), batch_size):
        chunk = data[start:start + batch_size]
        left, right, target = tensors(chunk, device)
        prediction = model(left, right).argmax(-1)
        matches = prediction == target
        for position in range(WIDTH):
            digit_correct[position] += int(matches[:, position].sum())
        for row, predicted, correct in zip(chunk, prediction.tolist(), matches.all(-1).tolist()):
            a, b = row[3], row[4]
            value = sum(digit * 10**position for position, digit in enumerate(predicted))
            predictions[(a, b)] = value
            exact += int(correct)
            total += 1
            for buckets, key in (
                (operand_buckets, f"{len(str(a))}x{len(str(b))}"),
                (product_buckets, len(str(a * b))),
                (carry_buckets, carry_count(a, b)),
            ):
                buckets[key][0] += int(correct)
                buckets[key][1] += 1
    comm_correct = comm_total = 0
    for (a, b), value in predictions.items():
        if a <= b and (b, a) in predictions:
            comm_correct += int(value == predictions[(b, a)])
            comm_total += 1
    return {
        "exact_numerical_accuracy": exact / total,
        "digit_accuracy_lsd_first": [value / total for value in digit_correct],
        "by_operand_digit_lengths": bucket_report(operand_buckets),
        "by_product_digit_length": bucket_report(product_buckets),
        "by_carry_columns": bucket_report(carry_buckets),
        "commutativity_consistency": comm_correct / comm_total,
        "examples": total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_examples, test_examples = split_examples(args.seed)
    train, test = rows(train_examples), rows(test_examples)
    model = SerialMultiplier().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(.9, .95), weight_decay=.01)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    curve = []
    for step in range(1, args.steps + 1):
        indices = torch.randint(len(train), (256,), generator=generator).tolist()
        sample = [train[index] for index in indices]
        left, right, target = tensors(sample, args.device)
        model.train()
        logits = model(left, right)
        loss = F.cross_entropy(logits.reshape(-1, 10), target.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 400 == 0:
            record = {"step": step, "loss": float(loss.detach())}
            curve.append(record)
            print(json.dumps({"type": "progress", **record}), flush=True)
    report = {
        "classification": "RESEARCH ONLY - fixed-width serial GRU multiplication",
        "architecture": "LSD-first paired digit scan plus two zero flush positions",
        "split": "same 80/20 unordered-pair groups as Transformer factorial",
        "seed": args.seed,
        "steps": args.steps,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "curve": curve,
        "train": evaluate(model, train, args.device),
        "test": evaluate(model, test, args.device),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "eval_report.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save(model.state_dict(), args.out / "model.pt")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

"""Direct MLP/Transformer baselines for unseen-x, unseen-N x^2 mod N.

Arithmetic is used only to construct synthetic labels and evaluation splits.
The learned forwards receive fixed-width decimal digits and emit digit logits.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def is_prime(value: int) -> bool:
    if value < 2:
        return False
    return all(value % divisor for divisor in range(2, math.isqrt(value) + 1))


def semiprimes(width: int) -> list[int]:
    lower, upper = 10 ** (width - 1), 10**width - 1
    primes = [value for value in range(2, upper + 1) if is_prime(value)]
    return sorted(
        {
            left * right
            for index, left in enumerate(primes)
            for right in primes[index:]
            if lower <= left * right <= upper
        }
    )


def digits(value: int, width: int) -> list[int]:
    return [int(character) for character in f"{value:0{width}d}"][::-1]


def rows_for_moduli(
    moduli: list[int], width: int, per_modulus: int | None, seed: int
) -> list[tuple[list[int], list[int], list[int], int, int]]:
    rows = []
    for index, modulus in enumerate(moduli):
        values = list(range(modulus))
        random.Random(seed + 104729 * index).shuffle(values)
        if per_modulus is not None:
            values = values[: min(per_modulus, len(values))]
        rows.extend(
            (
                digits(value, width),
                digits(modulus, width),
                digits((value * value) % modulus, width),
                value,
                modulus,
            )
            for value in values
        )
    return rows


def make_splits(width: int, split_seed: int, train_per_n: int, seen_test_per_n: int):
    moduli = semiprimes(width)
    random.Random(split_seed).shuffle(moduli)
    train_count = int(0.70 * len(moduli))
    validation_count = int(0.15 * len(moduli))
    train_moduli = moduli[:train_count]
    validation_moduli = moduli[train_count : train_count + validation_count]
    test_moduli = moduli[train_count + validation_count :]

    train_rows, seen_rows = [], []
    for index, modulus in enumerate(train_moduli):
        values = list(range(modulus))
        random.Random(split_seed + 65537 * index).shuffle(values)
        train_values = values[: min(train_per_n, len(values))]
        seen_values = values[
            min(train_per_n, len(values)) : min(train_per_n + seen_test_per_n, len(values))
        ]
        train_rows.extend(
            (digits(x, width), digits(modulus, width), digits((x * x) % modulus, width), x, modulus)
            for x in train_values
        )
        seen_rows.extend(
            (digits(x, width), digits(modulus, width), digits((x * x) % modulus, width), x, modulus)
            for x in seen_values
        )

    validation_rows = rows_for_moduli(validation_moduli, width, seen_test_per_n, split_seed + 1)
    test_rows = rows_for_moduli(test_moduli, width, None, split_seed + 2)
    return {
        "train": train_rows,
        "seen_n_unseen_x": seen_rows,
        "unseen_n_validation": validation_rows,
        "unseen_n_test": test_rows,
        "train_moduli": train_moduli,
        "validation_moduli": validation_moduli,
        "test_moduli": test_moduli,
    }


class MLP(nn.Module):
    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.width = width
        self.network = nn.Sequential(
            nn.Linear(20 * width, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 10 * width),
        )

    def forward(self, x_digits: torch.Tensor, n_digits: torch.Tensor) -> torch.Tensor:
        encoded = torch.cat((F.one_hot(x_digits, 10), F.one_hot(n_digits, 10)), dim=1)
        return self.network(encoded.flatten(1).float()).reshape(-1, self.width, 10)


class Transformer(nn.Module):
    def __init__(self, width: int, hidden: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.digit = nn.Embedding(10, hidden)
        self.role = nn.Embedding(3, hidden)
        self.position = nn.Embedding(3 * width, hidden)
        self.output_queries = nn.Parameter(torch.randn(width, hidden) * 0.02)
        block = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=4 * hidden,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers, norm=nn.LayerNorm(hidden))
        self.head = nn.Linear(hidden, 10)

    def forward(self, x_digits: torch.Tensor, n_digits: torch.Tensor) -> torch.Tensor:
        batch = x_digits.shape[0]
        positions = self.position.weight.unsqueeze(0)
        x = self.digit(x_digits) + self.role.weight[0]
        n = self.digit(n_digits) + self.role.weight[1]
        queries = self.output_queries.unsqueeze(0).expand(batch, -1, -1) + self.role.weight[2]
        state = torch.cat((x, n, queries), dim=1) + positions
        return self.head(self.encoder(state)[:, -self.width :])


def to_tensors(rows, device):
    return (
        torch.tensor([row[0] for row in rows], dtype=torch.long, device=device),
        torch.tensor([row[1] for row in rows], dtype=torch.long, device=device),
        torch.tensor([row[2] for row in rows], dtype=torch.long, device=device),
    )


@torch.no_grad()
def evaluate(model, rows, device, batch_size=4096):
    model.eval()
    exact, digits_correct, examples, loss_sum = 0, 0, 0, 0.0
    for start in range(0, len(rows), batch_size):
        x, n, target = to_tensors(rows[start : start + batch_size], device)
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
        "digit_accuracy": digits_correct / (examples * target.shape[1]),
        "cross_entropy": loss_sum / (examples * target.shape[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("mlp", "transformer"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=20260810)
    parser.add_argument("--width", type=int, default=3)
    parser.add_argument("--train-per-n", type=int, default=64)
    parser.add_argument("--seen-test-per-n", type=int, default=64)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    splits = make_splits(args.width, args.split_seed, args.train_per_n, args.seen_test_per_n)
    if args.arch == "mlp":
        model = MLP(args.width, args.hidden).to(args.device)
    else:
        model = Transformer(args.width, args.hidden, args.layers, args.heads).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    train_x, train_n, train_target = to_tensors(splits["train"], args.device)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.jsonl"
    curve_sets = {
        name: rows[: min(4096, len(rows))]
        for name, rows in splits.items()
        if name in ("train", "seen_n_unseen_x", "unseen_n_test")
    }
    with metrics_path.open("w") as metrics_file:
        for step in range(1, args.steps + 1):
            model.train()
            indices = torch.randint(len(train_x), (args.batch_size,), generator=generator).to(args.device)
            logits = model(train_x[indices], train_n[indices])
            loss = F.cross_entropy(logits.flatten(0, 1), train_target[indices].flatten())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                record = {"step": step, "train_batch_loss": float(loss)}
                record.update({name: evaluate(model, rows, args.device) for name, rows in curve_sets.items()})
                metrics_file.write(json.dumps(record) + "\n")
                metrics_file.flush()
                print(json.dumps(record), flush=True)

    final = {
        "arch": args.arch,
        "seed": args.seed,
        "width": args.width,
        "steps": args.steps,
        "train_per_n": args.train_per_n,
        "seen_test_per_n": args.seen_test_per_n,
        "split_seed": args.split_seed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "optimizer": {"name": "AdamW", "lr": args.lr, "weight_decay": args.weight_decay},
        "modulus_counts": {
            "train": len(splits["train_moduli"]),
            "validation": len(splits["validation_moduli"]),
            "test": len(splits["test_moduli"]),
        },
        "row_counts": {name: len(rows) for name, rows in splits.items() if name.endswith("x") or name in ("train", "unseen_n_validation", "unseen_n_test")},
        "metrics": {
            name: evaluate(model, rows, args.device)
            for name, rows in splits.items()
            if name in ("train", "seen_n_unseen_x", "unseen_n_validation", "unseen_n_test")
        },
    }
    (out / "eval_report.json").write_text(json.dumps(final, indent=2) + "\n")
    torch.save(model.state_dict(), out / "final.pt")
    print(json.dumps(final), flush=True)


if __name__ == "__main__":
    main()

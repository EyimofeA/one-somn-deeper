"""Final-label-only reduction diagnostic with an evaluation-time depth sweep.

The model receives the exact decimal digits of x**2 and N. Arithmetic is used
only to construct inputs and final labels; no quotient, comparison, borrow, or
intermediate reduction trace is supervised.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


WIDTH = 4
VOCAB = 10


def encode(value: int) -> list[int]:
    return [int(character) for character in f"{value:0{WIDTH}d}"][::-1]


def decode(tokens: list[int]) -> int:
    return sum(token * (10**index) for index, token in enumerate(tokens))


def moduli(seed: int = 0) -> list[int]:
    primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47)
    values = sorted(
        {p * q for index, p in enumerate(primes) for q in primes[index + 1 :] if 21 <= p * q <= 99}
    )
    return random.Random(seed).sample(values, 26)


def rows(moduli_list: list[int], train_split: bool) -> list[tuple[list[int], ...]]:
    output = []
    for modulus in moduli_list:
        values = list(range(modulus))
        random.Random(45 + modulus).shuffle(values)
        cut = int(0.8 * len(values))
        values = values[:cut] if train_split else values[cut:]
        for x in values:
            product = x * x
            target = product % modulus
            output.append((encode(modulus), encode(product), encode(target), target))
    return output


def batch(items, size, step, device):
    sample = [items[(step * size + index) % len(items)] for index in range(size)]
    n = torch.tensor([row[0] for row in sample], device=device)
    product = torch.tensor([row[1] for row in sample], device=device)
    y = torch.tensor([row[2] for row in sample], device=device)
    target = torch.tensor([row[3] for row in sample], device=device)
    return n, product, y, target


class ReductionTape(nn.Module):
    def __init__(self, d: int = 128):
        super().__init__()
        self.d = d
        self.token = nn.Embedding(VOCAB, 32)
        self.place = nn.Embedding(WIDTH, 32)
        self.n_local = nn.Sequential(nn.Linear(32, d), nn.LayerNorm(d), nn.GELU())
        self.product_local = nn.Sequential(nn.Linear(32, d), nn.LayerNorm(d), nn.GELU())
        self.init = nn.Sequential(nn.Linear(2 * d, d), nn.LayerNorm(d), nn.GELU())
        self.local_mix = nn.Linear(3 * d, d)
        self.cell = nn.GRUCell(2 * d, d)
        self.decoder = nn.Sequential(nn.Linear(d + 32, d), nn.GELU(), nn.Linear(d, VOCAB))

    def forward(self, n, product, recurrent_steps: int):
        positions = torch.arange(WIDTH, device=n.device)
        place = self.place(positions)
        n_state = self.n_local(self.token(n) + place)
        product_state = self.product_local(self.token(product) + place)
        hidden = self.init(torch.cat((n_state, product_state), dim=-1))
        for _ in range(recurrent_steps):
            zeros = torch.zeros_like(hidden[:, :1])
            left = torch.cat((zeros, hidden[:, :-1]), dim=1)
            right = torch.cat((hidden[:, 1:], zeros), dim=1)
            local = self.local_mix(torch.cat((left, hidden, right), dim=-1))
            hidden = self.cell(
                torch.cat((local, n_state), dim=-1).reshape(-1, 2 * self.d),
                hidden.reshape(-1, self.d),
            ).reshape_as(hidden)
        return self.decoder(torch.cat((hidden, place[None].expand(hidden.shape[0], -1, -1)), dim=-1))


@torch.no_grad()
def evaluate(model, data, recurrent_steps, device):
    model.eval()
    exact = token_correct = total = 0
    position_correct = torch.zeros(WIDTH, dtype=torch.long)
    for start in range(0, len(data), 512):
        sample = data[start : start + 512]
        n, product, y, target = batch(sample, len(sample), 0, device)
        prediction = model(n, product, recurrent_steps).argmax(dim=-1)
        matches = prediction == y
        token_correct += int(matches.sum())
        position_correct += matches.sum(dim=0).cpu()
        exact += sum(decode(row) == int(value) for row, value in zip(prediction.tolist(), target.tolist()))
        total += len(sample)
    return {
        "exact": exact / total,
        "token_accuracy": token_correct / (total * WIDTH),
        "position_accuracy_lsd_first": [int(value) / total for value in position_correct],
        "examples": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=1200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-steps", type=int, default=8)
    parser.add_argument("--eval-steps", type=int, nargs="+", default=(1, 2, 4, 8, 16, 32))
    parser.add_argument("--check-every", type=int, default=200)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--early-stop-exact", type=float, default=0.999)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    all_moduli = moduli()
    train_moduli, unseen_moduli = all_moduli[:18], all_moduli[18:]
    train = rows(train_moduli, True)
    held_out_x = rows(train_moduli, False)
    unseen_n = rows(unseen_moduli, True) + rows(unseen_moduli, False)

    model = ReductionTape().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, betas=(0.9, 0.95), weight_decay=0.05)
    started = time.monotonic()
    step = 0
    curve = []
    converged_checks = 0
    stopped_early = False
    while time.monotonic() - started < args.seconds:
        n, product, y, _ = batch(train, 512, step, device)
        model.train()
        logits = model(n, product, args.train_steps)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        step += 1
        if step == 1 or step % args.check_every == 0:
            train_metrics = evaluate(model, train, args.train_steps, device)
            record = {
                "step": step,
                "seconds": round(time.monotonic() - started, 2),
                "loss": float(loss.detach()),
                "last_batch_exact": float((logits.argmax(-1) == y).all(dim=-1).float().mean()),
                "full_train_exact": train_metrics["exact"],
            }
            curve.append(record)
            print(json.dumps({"type": "progress", **record}), flush=True)
            converged_checks = converged_checks + 1 if train_metrics["exact"] >= args.early_stop_exact else 0
            if converged_checks >= args.early_stop_patience:
                stopped_early = True
                break

    results = {}
    for recurrent_steps in args.eval_steps:
        results[str(recurrent_steps)] = {
            "train": evaluate(model, train, recurrent_steps, device),
            "held_out_x": evaluate(model, held_out_x, recurrent_steps, device),
            "unseen_N": evaluate(model, unseen_n, recurrent_steps, device),
        }

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out / "model.pt")
    report = {
        "classification": "RESEARCH ONLY - reduction from exact product with final remainder labels",
        "intermediate_supervision": False,
        "train_moduli": train_moduli,
        "unseen_moduli": unseen_moduli,
        "train_recurrent_steps": args.train_steps,
        "eval_recurrent_steps": args.eval_steps,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "updates": step,
        "seconds": time.monotonic() - started,
        "stopped_early": stopped_early,
        "early_stop_rule": {
            "metric": "full training exact accuracy",
            "threshold": args.early_stop_exact,
            "consecutive_checks": args.early_stop_patience,
            "check_every_updates": args.check_every,
        },
        "curve": curve,
        "depth_sweep": results,
    }
    (args.out / "eval_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

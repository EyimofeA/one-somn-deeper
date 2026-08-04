"""Clean learned square-then-reduce recurrent VDF diagnostic.

Synthetic arithmetic appears only in dataset construction and metric labels.
The forward transition consists entirely of learned LSD-first modules.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


WIDTH, HIDDEN = 4, 64


def digits(value): return [int(char) for char in f"{value:0{WIDTH}d}"][::-1]


def moduli(seed):
    primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47)
    values = sorted({p * q for index, p in enumerate(primes) for q in primes[index + 1:] if 21 <= p * q <= 99})
    return random.Random(seed).sample(values, 26)


def square_rows(modulus_values):
    return [(digits(value), digits(modulus), digits(value * value), value, modulus) for modulus in modulus_values for value in range(modulus)]


def reduction_rows(modulus_values):
    return [(digits(q * modulus + remainder), digits(modulus), digits((q - 1) * modulus + remainder), q > 0, q, remainder, modulus) for modulus in modulus_values for remainder in range(modulus) for q in range(modulus - 1)]


class Serial(nn.Module):
    def __init__(self, output):
        super().__init__()
        self.digit, self.place = nn.Embedding(10, HIDDEN), nn.Embedding(WIDTH, HIDDEN)
        self.pair, self.cell, self.head = nn.Linear(2 * HIDDEN, HIDDEN), nn.GRUCell(HIDDEN, HIDDEN), nn.Linear(HIDDEN, output)

    def encode(self, state, modulus):
        hidden, outputs = torch.zeros(state.shape[0], HIDDEN, device=state.device), []
        for position in range(WIDTH):
            place = self.place.weight[position]
            pair = torch.tanh(self.pair(torch.cat((self.digit(state[:, position]) + place, self.digit(modulus[:, position]) + place), -1)))
            hidden = self.cell(pair, hidden); outputs.append(hidden)
        return torch.stack(outputs, 1), hidden


class Square(nn.Module):
    def __init__(self): super().__init__(); self.serial = Serial(10)
    def forward(self, state, modulus): return self.serial.encode(state, modulus)[0] @ self.serial.head.weight.T + self.serial.head.bias


class Subtractor(nn.Module):
    def __init__(self): super().__init__(); self.serial = Serial(10)
    def forward(self, state, modulus): return self.serial.encode(state, modulus)[0] @ self.serial.head.weight.T + self.serial.head.bias


class Comparator(nn.Module):
    def __init__(self): super().__init__(); self.serial = Serial(1)
    def forward(self, state, modulus): return self.serial.encode(state, modulus)[1].squeeze(-1) @ self.serial.head.weight.squeeze(0) + self.serial.head.bias.squeeze(0)


def batch(rows, size, step):
    offset = (step * size) % len(rows)
    return rows[offset:offset + size] if offset + size <= len(rows) else rows[offset:] + rows[:(offset + size) % len(rows)]


def tensors(rows, device, target=2):
    return (torch.tensor([row[0] for row in rows], device=device), torch.tensor([row[1] for row in rows], device=device), torch.tensor([row[target] for row in rows], device=device))


def train_digits(model, rows, steps, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    for step in range(steps):
        state, modulus, target = tensors(batch(rows, 256, step), device)
        loss = F.cross_entropy(model(state, modulus).reshape(-1, 10), target.reshape(-1))
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()


def train_comparator(model, rows, steps, device):
    positive, negative = [row for row in rows if row[3]], [row for row in rows if not row[3]]
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    for step in range(steps):
        rng = random.Random(step)
        data = [rng.choice(positive if index % 2 else negative) for index in range(256)]
        state = torch.tensor([row[0] for row in data], device=device); modulus = torch.tensor([row[1] for row in data], device=device)
        label = torch.tensor([row[3] for row in data], dtype=torch.float32, device=device)
        loss = F.binary_cross_entropy_with_logits(model(state, modulus), label)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()


@torch.no_grad()
def reduce_state(subtractor, comparator, state, modulus):
    current, steps = state.clone(), torch.zeros(state.shape[0], dtype=torch.long, device=state.device)
    for _ in range(100):
        continue_ = comparator(current, modulus).sigmoid() >= .5
        updated = subtractor(current, modulus).argmax(-1)
        current = torch.where(continue_[:, None], updated, current)
        steps += continue_.long()
    return current, steps


@torch.no_grad()
def report(square, subtractor, comparator, modulus_values, device):
    sq = square_rows(modulus_values); red = reduction_rows(modulus_values)
    s, n, raw = tensors(sq, device)
    square_exact = float((square(s, n).argmax(-1) == raw).all(-1).float().mean())
    rs, rn, rt = tensors(red, device)
    subtract_exact = float((subtractor(rs, rn).argmax(-1) == rt).all(-1).float().mean())
    comp_exact = float(((comparator(rs, rn).sigmoid() >= .5) == torch.tensor([row[3] for row in red], device=device)).float().mean())
    predicted_raw = square(s, n).argmax(-1); one, reduce_steps = reduce_state(subtractor, comparator, predicted_raw, n)
    target_one = torch.tensor([digits((value * value) % modulus) for _, _, _, value, modulus in sq], device=device)
    one_exact = float((one == target_one).all(-1).float().mean())
    rollout = {}
    base = torch.tensor([digits(value) for modulus in modulus_values for value in range(modulus)], device=device)
    base_n = torch.tensor([digits(modulus) for modulus in modulus_values for _ in range(modulus)], device=device)
    expected = torch.tensor([digits(value) for modulus in modulus_values for value in range(modulus)], device=device)
    current = base.clone()
    for depth in range(1, 9):
        current, _ = reduce_state(subtractor, comparator, square(current, base_n).argmax(-1), base_n)
        expected_values = []
        for modulus in modulus_values:
            for value in range(modulus):
                for _ in range(depth): value = (value * value) % modulus
                expected_values.append(digits(value))
        expected = torch.tensor(expected_values, device=device)
        rollout[str(depth)] = float((current == expected).all(-1).float().mean())
    return {"square_exact": square_exact, "subtractor_exact": subtract_exact, "comparator_accuracy": comp_exact, "one_step_exact": one_exact, "mean_reduction_steps": float(reduce_steps.float().mean()), "rollout_exact": rollout, "examples": len(sq)}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", required=True); parser.add_argument("--seed", type=int, default=0); parser.add_argument("--steps", type=int, default=3000); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); all_moduli = moduli(args.seed); train_moduli, test_moduli = all_moduli[:18], all_moduli[18:]
    square, subtractor, comparator = Square().to(args.device), Subtractor().to(args.device), Comparator().to(args.device)
    reductions = reduction_rows(train_moduli)
    train_comparator(comparator, reductions, args.steps, args.device); train_digits(subtractor, [row for row in reductions if row[3]], args.steps, args.device); train_digits(square, square_rows(train_moduli), args.steps, args.device)
    square.eval(); subtractor.eval(); comparator.eval()
    result = {"width": WIDTH, "train_moduli": train_moduli, "test_moduli": test_moduli, "parameters": sum(p.numel() for m in (square, subtractor, comparator) for p in m.parameters()), "seen": report(square, subtractor, comparator, train_moduli, args.device), "unseen": report(square, subtractor, comparator, test_moduli, args.device)}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True); (out / "eval_report.json").write_text(json.dumps(result, indent=2) + "\n"); torch.save({"square": square.state_dict(), "subtractor": subtractor.state_dict(), "comparator": comparator.state_dict()}, out / "cell.pt"); print(json.dumps(result), flush=True)


if __name__ == "__main__": main()

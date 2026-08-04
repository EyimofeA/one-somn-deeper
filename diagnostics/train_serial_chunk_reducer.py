"""Learned serial reduction chunks; a diagnostic, never a submission path.

The forward predicts a five-way action and all next-state digits from learned
state. Arithmetic appears only while constructing synthetic supervision.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from train_serial_subtractor import WIDTH, SerialSubtractor, digits, semiprimes


ACTIONS = (0, 1, 2, 4, 8)


def action_index(quotient: int) -> int:
    return max(index for index, value in enumerate(ACTIONS) if value <= quotient)


def chunk_rows(moduli, seed, per_modulus, qs, *, heldout):
    result = []
    for index, modulus in enumerate(moduli):
        rng = random.Random(seed + 10_000 * index)
        sampled = set(rng.sample(list(range(modulus)), per_modulus))
        choices = list(range(modulus)) if heldout else list(sampled)
        for remainder in rng.sample(choices, per_modulus):
            for quotient in qs:
                action = ACTIONS[action_index(quotient)]
                current = quotient * modulus + remainder
                target = (quotient - action) * modulus + remainder
                if len(str(current)) > WIDTH or len(str(target)) > WIDTH:
                    raise ValueError("state exceeds configured decimal width")
                result.append((digits(current)[::-1], digits(modulus)[::-1], digits(target)[::-1], action_index(quotient)))
    random.Random(seed + int(heldout) + min(qs)).shuffle(result)
    return result


class SerialChunkReducer(nn.Module):
    """Existing serial digit reducer plus a learned discrete action head."""

    def __init__(self):
        super().__init__()
        self.digits = SerialSubtractor()
        self.action = nn.Linear(self.digits.cell.hidden_size, len(ACTIONS))

    def forward(self, state, modulus):
        hidden, final = self.digits.encode(state, modulus)
        return self.digits.head(hidden), self.action(final)


def tensors(rows, device):
    return tuple(torch.tensor([row[index] for row in rows], dtype=dtype, device=device) for index, dtype in ((0, torch.long), (1, torch.long), (2, torch.long), (3, torch.long)))


def batch(rows, batch_size, step):
    offset = ((step - 1) * batch_size) % len(rows)
    return [rows[(offset + index) % len(rows)] for index in range(batch_size)]


def expected_steps(quotient):
    steps = 0
    while quotient:
        quotient -= ACTIONS[action_index(quotient)]
        steps += 1
    return steps


@torch.no_grad()
def teacher_metrics(model, moduli, seed, per_modulus, quotients, device):
    report = {}
    for quotient in quotients:
        rows = chunk_rows(moduli, seed, per_modulus, range(quotient, quotient + 1), heldout=False)
        state, modulus, target, action = tensors(rows, device)
        digit_logits, action_logits = model(state, modulus)
        digit = digit_logits.argmax(dim=-1)
        selected = action_logits.argmax(dim=-1)
        report[str(quotient)] = {
            "next_state_exact": float((digit == target).all(dim=-1).float().mean()),
            "action_accuracy": float((selected == action).float().mean()),
            "joint_transition_exact": float(((digit == target).all(dim=-1) & (selected == action)).float().mean()),
            "per_lsd_position": [float((digit[:, position] == target[:, position]).float().mean()) for position in range(WIDTH)],
            "examples": len(rows),
        }
    return report


@torch.no_grad()
def autonomous_metrics(model, moduli, seed, per_modulus, quotients, max_steps, device):
    canonical = chunk_rows(moduli, seed, per_modulus, range(0, 1), heldout=False)
    remainder = torch.tensor([row[0] for row in canonical], dtype=torch.long, device=device)
    modulus = torch.tensor([row[1] for row in canonical], dtype=torch.long, device=device)
    report = {}
    for quotient in quotients:
        state = torch.tensor([digits(quotient * int("".join(map(str, row[1][::-1]))) + int("".join(map(str, row[0][::-1]))))[::-1] for row in canonical], dtype=torch.long, device=device)
        active = torch.ones(len(canonical), dtype=torch.bool, device=device)
        stopped = torch.full((len(canonical),), -1, dtype=torch.long, device=device)
        for step in range(max_steps + 1):
            digit_logits, action_logits = model(state, modulus)
            action = action_logits.argmax(dim=-1)
            stop = action == 0
            newly_stopped = active & stop
            stopped[newly_stopped] = step
            continuing = active & ~stop
            if step == max_steps:
                break
            state = torch.where(continuing[:, None], digit_logits.argmax(dim=-1), state)
            active = continuing
            if not active.any():
                break
        expected = expected_steps(quotient)
        report[str(quotient)] = {
            "remainder_exact": float((state == remainder).all(dim=-1).float().mean()),
            "exact_halt_step": float((stopped == expected).float().mean()),
            "early_stops": float((stopped.ge(0) & (stopped < expected)).float().mean()),
            "late_stops": float((stopped > expected).float().mean()),
            "non_stops": float((stopped < 0).float().mean()),
            "average_executed_steps": float(torch.where(stopped.ge(0), stopped, torch.full_like(stopped, max_steps + 1)).float().mean()),
            "expected_steps": expected,
            "examples": len(canonical),
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-train-q", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    all_moduli = semiprimes(args.seed, 64)
    train_moduli, test_moduli = all_moduli[:48], all_moduli[48:]
    train = chunk_rows(train_moduli, args.seed, 128, range(args.max_train_q + 1), heldout=False)
    model = SerialChunkReducer().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    for step in range(1, args.steps + 1):
        state, modulus, target, action = tensors(batch(train, args.batch_size, step), args.device)
        digit_logits, action_logits = model(state, modulus)
        loss = F.cross_entropy(digit_logits.reshape(-1, 10), target.reshape(-1)) + F.cross_entropy(action_logits, action)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    quotients = (0, 1, 2, 4, 5, 8, 10, 20, 50, 100, 1000)
    report = {
        "actions": ACTIONS,
        "width": WIDTH,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "teacher": teacher_metrics(model, test_moduli, args.seed, 128, quotients, args.device),
        "autonomous": autonomous_metrics(model, test_moduli, args.seed, 128, quotients, 140, args.device),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_report.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save(model.state_dict(), out / "chunk_reducer.pt")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

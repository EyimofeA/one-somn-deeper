"""Train only a chunk scheduler over a frozen learned unit reducer.

All state updates are repeated applications of the frozen learned unit model.
The controller emits an action class; no decimal arithmetic is implemented in
the forward path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from train_comparator_reducer import ComparatorReducer, SerialComparator
from train_serial_chunk_reducer import ACTIONS, action_index, batch, chunk_rows, expected_steps
from train_serial_subtractor import SerialSubtractor, digits, semiprimes


class FrozenUnitChunkController(nn.Module):
    def __init__(self, checkpoint, device):
        super().__init__()
        comparator = SerialComparator().to(device)
        subtractor = SerialSubtractor().to(device)
        comparator.load_state_dict(checkpoint["comparator"])
        subtractor.load_state_dict(checkpoint["subtractor"])
        self.unit = ComparatorReducer(comparator, subtractor)
        for parameter in self.unit.parameters():
            parameter.requires_grad_(False)
        self.action = nn.Linear(subtractor.cell.hidden_size, len(ACTIONS))

    def forward(self, state, modulus):
        _, final = self.unit.subtractor.encode(state, modulus)
        return self.action(final)

    @torch.no_grad()
    def unit_step(self, state, modulus):
        return self.unit(state, modulus)[0].argmax(dim=-1)


def tensors(rows, device):
    return tuple(torch.tensor([row[index] for row in rows], dtype=torch.long, device=device) for index in range(4))


@torch.no_grad()
def metrics(model, moduli, seed, per_modulus, quotients, device):
    report = {}
    for quotient in quotients:
        rows = chunk_rows(moduli, seed, per_modulus, range(quotient, quotient + 1), heldout=False)
        state, modulus, _, target_action = tensors(rows, device)
        remainder = torch.tensor([
            digits(int("".join(map(str, row[0][::-1]))) - quotient * int("".join(map(str, row[1][::-1]))))[::-1]
            for row in rows
        ], dtype=torch.long, device=device)
        action = model(state, modulus).argmax(dim=-1)
        action_accuracy = float((action == target_action).float().mean())

        current = torch.tensor([row[0] for row in rows], dtype=torch.long, device=device)
        for inner in range(ACTIONS[-1]):
            update = action > inner
            current = torch.where(update[:, None], model.unit_step(current, modulus), current)
        macro_exact = float((current == torch.tensor([row[2] for row in rows], dtype=torch.long, device=device)).all(dim=-1).float().mean())

        state = torch.tensor([row[0] for row in rows], dtype=torch.long, device=device)
        active = torch.ones(len(rows), dtype=torch.bool, device=device)
        stopped = torch.full((len(rows),), -1, dtype=torch.long, device=device)
        inner_total = torch.zeros(len(rows), dtype=torch.long, device=device)
        for outer in range(141):
            predicted = model(state, modulus).argmax(dim=-1)
            newly_stopped = active & (predicted == 0)
            stopped[newly_stopped] = outer
            active &= predicted != 0
            if outer == 140 or not active.any():
                break
            for inner in range(ACTIONS[-1]):
                update = active & (predicted > inner)
                state = torch.where(update[:, None], model.unit_step(state, modulus), state)
                inner_total += update.long()
        expected = expected_steps(quotient)
        report[str(quotient)] = {
            "action_accuracy": action_accuracy,
            "macro_transition_exact": macro_exact,
            "remainder_exact": float((state == remainder).all(dim=-1).float().mean()),
            "exact_outer_steps": float((stopped == expected).float().mean()),
            "average_outer_steps": float(torch.where(stopped.ge(0), stopped, torch.full_like(stopped, 141)).float().mean()),
            "average_inner_unit_steps": float(inner_total.float().mean()),
            "early_stops": float((stopped.ge(0) & (stopped < expected)).float().mean()),
            "late_stops": float((stopped > expected).float().mean()),
            "non_stops": float((stopped < 0).float().mean()),
            "expected_outer_steps": expected,
            "examples": len(rows),
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-reducer-checkpoint", required=True)
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
    checkpoint = torch.load(args.unit_reducer_checkpoint, map_location=args.device, weights_only=True)
    model = FrozenUnitChunkController(checkpoint, args.device).to(args.device)
    optimizer = torch.optim.AdamW(model.action.parameters(), lr=3e-4, weight_decay=.01)
    for step in range(1, args.steps + 1):
        state, modulus, _, action = tensors(batch(train, args.batch_size, step), args.device)
        loss = F.cross_entropy(model(state, modulus), action)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    report = {
        "actions": ACTIONS,
        "controller_parameters": sum(parameter.numel() for parameter in model.action.parameters()),
        "metrics": metrics(model, test_moduli, args.seed, 128, (0, 1, 2, 4, 5, 8, 10, 20, 50, 100, 1000), args.device),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_report.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save(model.action.state_dict(), out / "controller.pt")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

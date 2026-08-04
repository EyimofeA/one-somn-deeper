"""Test a learned one-call macro transition conditioned on a learned chunk code.

Unlike the refuted direct chunk decoder, the decoder receives the chosen chunk
bits on every digit update.  Arithmetic is used only to create synthetic labels.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from train_frozen_unit_threshold_controller import BITS, MAX_CHUNK, ThresholdController, bit_target, selected
from train_serial_subtractor import SerialSubtractor, digits, semiprimes


def chunk(quotient):
    return min(quotient, MAX_CHUNK)


def rows(moduli, seed, per_modulus, quotients):
    result = []
    for index, modulus in enumerate(moduli):
        rng = random.Random(seed + 10_000 * index)
        for remainder in rng.sample(list(range(modulus)), per_modulus):
            for quotient in quotients:
                value = quotient * modulus + remainder
                target = (quotient - chunk(quotient)) * modulus + remainder
                if len(str(value)) > 14: raise ValueError("state exceeds W=14")
                result.append((digits(value)[::-1], digits(modulus)[::-1], digits(target)[::-1], quotient, bit_target(quotient), digits(remainder)[::-1]))
    random.Random(seed + min(quotients)).shuffle(result)
    return result


class MacroDecoder(nn.Module):
    def __init__(self, unit_weights):
        super().__init__()
        unit = SerialSubtractor()
        unit.load_state_dict(unit_weights)
        self.digit, self.place, self.pair, self.cell, self.head = unit.digit, unit.place, unit.pair, unit.cell, unit.head
        self.action = nn.Linear(len(BITS), self.cell.hidden_size)
        nn.init.zeros_(self.action.weight); nn.init.zeros_(self.action.bias)

    def forward(self, state, modulus, action):
        hidden = torch.zeros(state.shape[0], self.cell.hidden_size, device=state.device)
        action_state = self.action(action.to(dtype=self.action.weight.dtype))
        output = []
        for position in range(14):
            place = self.place.weight[position]
            pair = torch.tanh(self.pair(torch.cat((self.digit(state[:, position]) + place, self.digit(modulus[:, position]) + place), dim=-1)))
            hidden = self.cell(pair + action_state, hidden)
            output.append(hidden)
        return self.head(torch.stack(output, dim=1))


def tensor(data, device):
    return (
        torch.tensor([row[0] for row in data], dtype=torch.long, device=device),
        torch.tensor([row[1] for row in data], dtype=torch.long, device=device),
        torch.tensor([row[2] for row in data], dtype=torch.long, device=device),
        torch.tensor([row[4] for row in data], dtype=torch.float32, device=device),
    )


def balanced(groups, batch_size, seed, step):
    rng = random.Random(seed + step)
    return [rng.choice(groups[index % len(groups)]) for index in range(batch_size)]


@torch.no_grad()
def evaluate(decoder, controller, moduli, seed, per_modulus, quotients, device):
    report = {}
    for quotient in quotients:
        data = rows(moduli, seed, per_modulus, range(quotient, quotient + 1))
        state, modulus, target, oracle = tensor(data, device)
        controller_bits = (controller(state, modulus).sigmoid() >= .5).float()
        predicted = decoder(state, modulus, controller_bits).argmax(dim=-1)
        report[str(quotient)] = {
            "controller_selected_k_accuracy": float((selected(controller(state, modulus)) == chunk(quotient)).float().mean()),
            "macro_transition_exact": float((predicted == target).all(dim=-1).float().mean()),
            "per_lsd_position": [float((predicted[:, position] == target[:, position]).float().mean()) for position in range(14)],
            "examples": len(data),
        }
    return report


@torch.no_grad()
def rollout(decoder, controller, moduli, seed, per_modulus, quotients, device):
    result = {}
    canonical = rows(moduli, seed, per_modulus, range(0, 1))
    remainder = torch.tensor([row[5] for row in canonical], dtype=torch.long, device=device)
    modulus = torch.tensor([row[1] for row in canonical], dtype=torch.long, device=device)
    for quotient in quotients:
        state = torch.tensor([digits(quotient * int("".join(map(str, row[1][::-1]))) + int("".join(map(str, row[5][::-1]))))[::-1] for row in canonical], dtype=torch.long, device=device)
        active, stopped = torch.ones(len(canonical), dtype=torch.bool, device=device), torch.full((len(canonical),), -1, dtype=torch.long, device=device)
        expected = (quotient + MAX_CHUNK - 1) // MAX_CHUNK
        for outer in range(max(70, expected + 2)):
            action = (controller(state, modulus).sigmoid() >= .5).float()
            count = selected(controller(state, modulus))
            newly = active & (count == 0); stopped[newly] = outer; active &= count != 0
            if not active.any(): break
            updated = decoder(state, modulus, action).argmax(dim=-1)
            state = torch.where(active[:, None], updated, state)
        result[str(quotient)] = {
            "remainder_exact": float((state == remainder).all(dim=-1).float().mean()),
            "exact_outer_steps": float((stopped == expected).float().mean()),
            "early_stops": float((stopped.ge(0) & (stopped < expected)).float().mean()),
            "non_stops": float((stopped < 0).float().mean()), "examples": len(canonical),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reducer-checkpoint", required=True); parser.add_argument("--controller-checkpoint", required=True)
    parser.add_argument("--out", required=True); parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=4000); parser.add_argument("--batch-size", type=int, default=512); parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    all_moduli = semiprimes(args.seed, 64); train_moduli, test_moduli = all_moduli[:48], all_moduli[48:]
    checkpoint = torch.load(args.reducer_checkpoint, map_location=args.device, weights_only=True)
    controller = ThresholdController(checkpoint, args.device, "positions").to(args.device)
    controller.bits.load_state_dict(torch.load(args.controller_checkpoint, map_location=args.device, weights_only=True)); controller.eval()
    for parameter in controller.parameters(): parameter.requires_grad_(False)
    decoder = MacroDecoder(checkpoint["subtractor"]).to(args.device)
    train = rows(train_moduli, args.seed, 128, range(101)); groups = [[row for row in train if chunk(row[3]) == value] for value in range(MAX_CHUNK + 1)]
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=3e-4, weight_decay=.01)
    for step in range(1, args.steps + 1):
        state, modulus, target, action = tensor(balanced(groups, args.batch_size, args.seed, step), args.device)
        loss = F.cross_entropy(decoder(state, modulus, action).reshape(-1, 10), target.reshape(-1))
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    decoder.eval()
    report = {"width": 14, "decoder_parameters": sum(p.numel() for p in decoder.parameters()), "metrics": evaluate(decoder, controller, test_moduli, args.seed, 128, (0, 1, 2, 4, 8, 16, 32, 100, 1000), args.device), "autonomous": rollout(decoder, controller, test_moduli, args.seed, 128, (0, 1, 2, 4, 8, 16, 32, 100, 1000), args.device)}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True); (out / "eval_report.json").write_text(json.dumps(report, indent=2) + "\n"); torch.save(decoder.state_dict(), out / "decoder.pt"); print(json.dumps(report), flush=True)


if __name__ == "__main__": main()

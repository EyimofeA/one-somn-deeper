"""Train a binary chunk-code controller over a frozen learned unit reducer.

The four learned bits encode the safe greedy chunk ``min(q, 15)``.  Each
chosen unit is still executed by the frozen learned reducer; no decimal
arithmetic is implemented in this model's forward path.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from train_comparator_reducer import ComparatorReducer, SerialComparator
from train_serial_subtractor import SerialSubtractor, digits, semiprimes


BITS = (1, 2, 4, 8)
MAX_CHUNK = sum(BITS)


def chunk_target(quotient):
    """Binary encoding of a safe, non-overshooting greedy chunk."""
    return min(quotient, MAX_CHUNK)


def bit_target(quotient):
    value = chunk_target(quotient)
    return [(value & bit) != 0 for bit in BITS]


def rows(moduli, seed, per_modulus, quotients):
    result = []
    for index, modulus in enumerate(moduli):
        rng = random.Random(seed + 10_000 * index)
        for remainder in rng.sample(list(range(modulus)), per_modulus):
            for quotient in quotients:
                state = quotient * modulus + remainder
                if len(str(state)) > 14:
                    raise ValueError("state exceeds W=14")
                result.append((digits(state)[::-1], digits(modulus)[::-1], digits(remainder)[::-1], quotient, bit_target(quotient)))
    random.Random(seed + min(quotients)).shuffle(result)
    return result


class ThresholdController(nn.Module):
    def __init__(self, checkpoint, device, features):
        super().__init__()
        comparator, subtractor = SerialComparator().to(device), SerialSubtractor().to(device)
        comparator.load_state_dict(checkpoint["comparator"])
        subtractor.load_state_dict(checkpoint["subtractor"])
        self.unit = ComparatorReducer(comparator, subtractor)
        for parameter in self.unit.parameters():
            parameter.requires_grad_(False)
        self.features = features
        feature_size = subtractor.cell.hidden_size if features == "final" else 14 * subtractor.cell.hidden_size
        self.bits = nn.Linear(feature_size, len(BITS))

    def forward(self, state, modulus):
        hidden, final = self.unit.subtractor.encode(state, modulus)
        features = final if self.features == "final" else hidden.reshape(state.shape[0], -1)
        return self.bits(features)

    @torch.no_grad()
    def unit_step(self, state, modulus):
        return self.unit(state, modulus)[0].argmax(dim=-1)


def tensors(data, device):
    state = torch.tensor([row[0] for row in data], dtype=torch.long, device=device)
    modulus = torch.tensor([row[1] for row in data], dtype=torch.long, device=device)
    target = torch.tensor([row[4] for row in data], dtype=torch.float32, device=device)
    return state, modulus, target


def balanced_batch(groups, batch_size, seed, step):
    rng = random.Random(seed + step)
    return [rng.choice(groups[index % len(groups)]) for index in range(batch_size)]


def selected(logits):
    return ((logits.sigmoid() >= .5).long() * torch.tensor(BITS, device=logits.device)).sum(dim=-1)


@torch.no_grad()
def run_units(model, state, modulus, counts):
    current = state.clone()
    for inner in range(MAX_CHUNK):
        current = torch.where((counts > inner)[:, None], model.unit_step(current, modulus), current)
    return current


@torch.no_grad()
def metrics(model, moduli, seed, per_modulus, quotients, device):
    report = {}
    for quotient in quotients:
        data = rows(moduli, seed, per_modulus, range(quotient, quotient + 1))
        state, modulus, target_bits = tensors(data, device)
        remainder = torch.tensor([row[2] for row in data], dtype=torch.long, device=device)
        logits = model(state, modulus)
        choices = selected(logits)
        target_chunk = chunk_target(quotient)
        macro = run_units(model, state, modulus, choices)

        current = state.clone()
        active = torch.ones(len(data), dtype=torch.bool, device=device)
        stopped = torch.full((len(data),), -1, dtype=torch.long, device=device)
        unit_total = torch.zeros(len(data), dtype=torch.long, device=device)
        expected = (quotient + MAX_CHUNK - 1) // MAX_CHUNK
        for outer in range(max(70, expected + 2)):
            choices = selected(model(current, modulus))
            newly = active & (choices == 0)
            stopped[newly] = outer
            active &= choices != 0
            if not active.any():
                break
            for inner in range(MAX_CHUNK):
                update = active & (choices > inner)
                current = torch.where(update[:, None], model.unit_step(current, modulus), current)
                unit_total += update.long()
        predicted_bits = logits.sigmoid() >= .5
        report[str(quotient)] = {
            "threshold_accuracy": [float((predicted_bits[:, index] == target_bits[:, index].bool()).float().mean()) for index in range(len(BITS))],
            "selected_k_accuracy": float((selected(logits) == target_chunk).float().mean()),
            "macro_transition_exact": float((macro == torch.tensor([digits((quotient - target_chunk) * int(''.join(map(str, row[1][::-1]))) + int(''.join(map(str, row[2][::-1]))))[::-1] for row in data], dtype=torch.long, device=device)).all(dim=-1).float().mean()),
            "remainder_exact": float((current == remainder).all(dim=-1).float().mean()),
            "q0_fixed_point_exact": float((current == state).all(dim=-1).float().mean()) if quotient == 0 else None,
            "exact_outer_steps": float((stopped == expected).float().mean()),
            "average_outer_steps": float(torch.where(stopped.ge(0), stopped, torch.full_like(stopped, max(70, expected + 2))).float().mean()),
            "average_inner_unit_steps": float(unit_total.float().mean()),
            "early_stops": float((stopped.ge(0) & (stopped < expected)).float().mean()),
            "late_stops": float((stopped > expected).float().mean()),
            "non_stops": float((stopped < 0).float().mean()),
            "expected_outer_steps": expected,
            "examples": len(data),
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
    parser.add_argument("--features", choices=("final", "positions"), default="final")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    all_moduli = semiprimes(args.seed, 64)
    train_moduli, test_moduli = all_moduli[:48], all_moduli[48:]
    train = rows(train_moduli, args.seed, 128, range(args.max_train_q + 1))
    groups = [[row for row in train if chunk_target(row[3]) == chunk] for chunk in range(MAX_CHUNK + 1)]
    checkpoint = torch.load(args.unit_reducer_checkpoint, map_location=args.device, weights_only=True)
    model = ThresholdController(checkpoint, args.device, args.features).to(args.device)
    optimizer = torch.optim.AdamW(model.bits.parameters(), lr=3e-4, weight_decay=.01)
    for step in range(1, args.steps + 1):
        state, modulus, target = tensors(balanced_batch(groups, args.batch_size, args.seed, step), args.device)
        loss = F.binary_cross_entropy_with_logits(model(state, modulus), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    report = {
        "bits": BITS, "max_chunk": MAX_CHUNK, "width": 14, "features": args.features,
        "controller_parameters": sum(parameter.numel() for parameter in model.bits.parameters()),
        "metrics": metrics(model, test_moduli, args.seed, 128, (0, 1, 2, 4, 8, 16, 32, 100, 1000), args.device),
    }
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "eval_report.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save(model.bits.state_dict(), out / "controller.pt")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

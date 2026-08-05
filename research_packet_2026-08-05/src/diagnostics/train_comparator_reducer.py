"""Comparator-gated learned serial reduction diagnostic; never a submission path."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from train_serial_subtractor import WIDTH, SerialSubtractor, canonical_rows, digits, rows, semiprimes, tensors


class SerialComparator(nn.Module):
    """A learned serial classifier for x >= N; no comparison in its forward."""

    def __init__(self, width=128):
        super().__init__()
        self.digit = nn.Embedding(10, width)
        self.place = nn.Embedding(WIDTH, width)
        self.pair = nn.Linear(2 * width, width)
        self.cell = nn.GRUCell(width, width)
        self.head = nn.Linear(width, 1)

    def forward(self, state, modulus):
        hidden = torch.zeros(state.shape[0], self.cell.hidden_size, device=state.device)
        for position in range(WIDTH):
            place = self.place.weight[position]
            x = torch.tanh(self.pair(torch.cat((self.digit(state[:, position]) + place, self.digit(modulus[:, position]) + place), dim=-1)))
            hidden = self.cell(x, hidden)
        return self.head(hidden).squeeze(-1)


def comparison_rows(moduli, seed, per_modulus):
    """Exactly balanced classes plus explicit N-1/N/N+1 boundary examples."""
    result, boundary = [], []
    for index, modulus in enumerate(moduli):
        rng = random.Random(seed + 10_000 * index)
        negative = rng.sample(list(range(modulus)), per_modulus)
        positive = [(rng.randrange(1, 21), rng.randrange(modulus)) for _ in range(per_modulus)]
        for value in negative:
            result.append((digits(value)[::-1], digits(modulus)[::-1], 0, False))
        for quotient, remainder in positive:
            result.append((digits(quotient * modulus + remainder)[::-1], digits(modulus)[::-1], 1, False))
        # Two explicit extras per class preserve exact 50/50 balance.
        for value in (modulus - 1, modulus - 2):
            result.append((digits(value)[::-1], digits(modulus)[::-1], 0, True))
        for value in (modulus, modulus + 1):
            result.append((digits(value)[::-1], digits(modulus)[::-1], 1, True))
    random.Random(seed + 55).shuffle(result)
    return result


def comparator_metrics(model, dataset, device):
    state = torch.tensor([row[0] for row in dataset], dtype=torch.long, device=device)
    modulus = torch.tensor([row[1] for row in dataset], dtype=torch.long, device=device)
    label = torch.tensor([row[2] for row in dataset], dtype=torch.long, device=device)
    boundary = torch.tensor([row[3] for row in dataset], dtype=torch.bool, device=device)
    pred = model(state, modulus).sigmoid() >= 0.5
    return {
        "accuracy": float((pred == label).float().mean()),
        "boundary_accuracy": float((pred[boundary] == label[boundary]).float().mean()),
        "n": len(dataset), "boundary_n": int(boundary.sum()),
        "negative_accuracy": float((pred[label == 0] == 0).float().mean()),
        "positive_accuracy": float((pred[label == 1] == 1).float().mean()),
    }


class ComparatorReducer(nn.Module):
    """Learned gate selects learned subtraction logits or an identity residual."""

    def __init__(self, comparator, subtractor):
        super().__init__()
        self.comparator = comparator
        self.subtractor = subtractor

    def forward(self, state, modulus):
        gate = self.comparator(state, modulus).sigmoid().view(-1, 1, 1)
        subtraction = self.subtractor(state, modulus).softmax(dim=-1)
        identity = F.one_hot(state, num_classes=10).float()
        return (gate * subtraction + (1 - gate) * identity).clamp_min(1e-8).log(), gate[:, 0, 0]


def transition_rows(moduli, seed, per_modulus, max_q):
    identity = canonical_rows(moduli, seed, per_modulus)
    subtraction = rows(moduli, seed, per_modulus, heldout=False, qs=range(1, max_q + 1))
    all_rows = identity + subtraction
    random.Random(seed + 88).shuffle(all_rows)
    return all_rows


@torch.no_grad()
def reducer_metrics(model, moduli, seed, per_modulus, max_q, max_steps, device):
    trace = transition_rows(moduli, seed, per_modulus, max_q)
    report = {}
    for quotient in range(max_q + 1):
        batch = [row for row in trace if (row[0] == row[2]) == (quotient == 0)] if quotient == 0 else rows(moduli, seed, per_modulus, heldout=False, qs=range(quotient, quotient + 1))
        state, modulus, target = tensors(batch, device)
        log_probs, gate = model(state, modulus)
        pred = log_probs.argmax(dim=-1)
        report[str(quotient)] = {"transition_exact": float((pred == target).all(dim=-1).float().mean()), "continue_accuracy": float(((gate >= .5) == (quotient > 0)).float().mean()), "examples": len(batch)}
    # Independent q=0 fixed-point check and autonomous terminal loop.
    canonical = canonical_rows(moduli, seed, per_modulus)
    remainder = torch.tensor([row[0] for row in canonical], dtype=torch.long, device=device)
    modulus = torch.tensor([row[1] for row in canonical], dtype=torch.long, device=device)
    fixed, _ = model(remainder, modulus)
    fixed_exact = float((fixed.argmax(dim=-1) == remainder).all(dim=-1).float().mean())
    auto = {}
    for quotient in range(max_q + 1):
        state = torch.tensor([digits(quotient * int("".join(map(str, row[1][::-1]))) + int("".join(map(str, row[0][::-1]))))[::-1] for row in canonical], dtype=torch.long, device=device)
        target = remainder
        active = torch.ones(len(canonical), dtype=torch.bool, device=device)
        stopped = torch.full((len(canonical),), -1, dtype=torch.long, device=device)
        for step in range(max_steps + 1):
            log_probs, gate = model(state, modulus)
            stop = gate < .5
            newly = active & stop
            stopped[newly] = step
            continuing = active & ~stop
            if step == max_steps:
                break
            state = torch.where(continuing[:, None], log_probs.argmax(dim=-1), state)
            active = continuing
        auto[str(quotient)] = {"remainder_exact": float((state == target).all(dim=-1).float().mean()), "exact_halt_step": float((stopped == quotient).float().mean()), "early_stops": float((stopped.ge(0) & (stopped < quotient)).float().mean()), "late_stops": float((stopped > quotient).float().mean()), "non_stops": float((stopped < 0).float().mean()), "examples": len(canonical)}
    return {"transition": report, "true_remainder_fixed_point_exact": fixed_exact, "autonomous": auto}


def batch(rows_, batch_size, step):
    offset = ((step - 1) * batch_size) % len(rows_)
    return [rows_[(offset + index) % len(rows_)] for index in range(batch_size)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("comparator", "reducer"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--comparator-checkpoint")
    parser.add_argument("--subtractor-checkpoint")
    parser.add_argument("--reducer-checkpoint", help="Resume a qualified comparator/subtractor pair without changing its architecture.")
    parser.add_argument("--max-train-q", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    moduli = semiprimes(args.seed, 64); train_moduli, test_moduli = moduli[:48], moduli[48:]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    if args.stage == "comparator":
        train, unseen = comparison_rows(train_moduli, args.seed, 128), comparison_rows(test_moduli, args.seed, 128)
        model = SerialComparator().to(args.device); optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
        for step in range(1, args.steps + 1):
            data = batch(train, args.batch_size, step)
            state = torch.tensor([row[0] for row in data], dtype=torch.long, device=args.device); modulus = torch.tensor([row[1] for row in data], dtype=torch.long, device=args.device); label = torch.tensor([row[2] for row in data], dtype=torch.float32, device=args.device)
            loss = F.binary_cross_entropy_with_logits(model(state, modulus), label); optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        model.eval(); report = {"seen": comparator_metrics(model, train, args.device), "unseen": comparator_metrics(model, unseen, args.device), "parameters": sum(p.numel() for p in model.parameters())}
        torch.save(model.state_dict(), out / "comparator.pt")
    else:
        comparator = SerialComparator().to(args.device)
        subtractor = SerialSubtractor().to(args.device)
        if args.reducer_checkpoint:
            weights = torch.load(args.reducer_checkpoint, map_location=args.device, weights_only=True)
            comparator.load_state_dict(weights["comparator"])
            subtractor.load_state_dict(weights["subtractor"])
        else:
            if not args.comparator_checkpoint or not args.subtractor_checkpoint: raise ValueError("stage reducer needs --reducer-checkpoint or both component checkpoints")
            comparator.load_state_dict(torch.load(args.comparator_checkpoint, map_location=args.device, weights_only=True))
            subtractor.load_state_dict(torch.load(args.subtractor_checkpoint, map_location=args.device, weights_only=True))
        model = ComparatorReducer(comparator, subtractor).to(args.device); train = transition_rows(train_moduli, args.seed, 128, args.max_train_q); optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
        for step in range(1, args.steps + 1):
            data = batch(train, args.batch_size, step); state, modulus, target = tensors(data, args.device); continue_label = torch.tensor([row[0] != row[2] for row in data], dtype=torch.float32, device=args.device)
            log_probs, gate = model(state, modulus); loss = F.nll_loss(log_probs.reshape(-1, 10), target.reshape(-1)) + F.binary_cross_entropy(gate, continue_label)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        model.eval(); report = reducer_metrics(model, test_moduli, args.seed, 128, 100, 110, args.device)
        torch.save({"comparator": comparator.state_dict(), "subtractor": subtractor.state_dict()}, out / "reducer.pt")
    (out / "eval_report.json").write_text(json.dumps(report, indent=2) + "\n"); print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

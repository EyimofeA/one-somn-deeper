"""Held-out-modulus q=1 reduction with a fully learned digit-serial cell.

Labels use ``r = u - n`` only while constructing examples.  The model forward
contains embeddings, a GRU, and categorical digit heads—no arithmetic,
comparison, borrow rule, or lookup table.  Processing is LSD-to-MSD so a
learned recurrent state can represent a borrow-like dependency.
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


WIDTH = 6


def digits(value: int) -> list[int]:
    return [int(c) for c in f"{value:0{WIDTH}d}"]


def is_prime(value: int) -> bool:
    return value >= 2 and all(value % divisor for divisor in range(2, int(value**0.5) + 1))


def semiprimes(seed: int, count: int) -> list[int]:
    primes = [value for value in range(31, 100) if is_prime(value)]
    values = sorted({a * b for index, a in enumerate(primes) for b in primes[index:] if 1000 <= a * b <= 9999})
    return random.Random(seed).sample(values, count)


def rows(moduli: list[int], seed: int, per_modulus: int, heldout: bool, qs: range = range(1, 2)) -> list[tuple[list[int], list[int], list[int]]]:
    result = []
    for index, modulus in enumerate(moduli):
        rng = random.Random(seed + 10_000 * index)
        all_remainders = list(range(modulus))
        train = set(rng.sample(all_remainders, per_modulus))
        choices = [r for r in all_remainders if r not in train] if heldout else list(train)
        for remainder in rng.sample(choices, per_modulus):
            for q in qs:
                current = q * modulus + remainder
                target = (q - 1) * modulus + remainder
                if len(str(current)) > WIDTH or len(str(target)) > WIDTH:
                    raise ValueError("state exceeds configured decimal width")
                result.append((digits(current)[::-1], digits(modulus)[::-1], digits(target)[::-1]))
    random.Random(seed + int(heldout)).shuffle(result)
    return result


def canonical_rows(moduli: list[int], seed: int, per_modulus: int) -> list[tuple[list[int], list[int], list[int]]]:
    """Learned identity targets r -> r for sampled canonical states."""
    result = []
    for index, modulus in enumerate(moduli):
        rng = random.Random(seed + 10_000 * index)
        sampled = rng.sample(list(range(modulus)), per_modulus)
        for remainder in sampled:
            state = digits(remainder)[::-1]
            result.append((state, digits(modulus)[::-1], state))
    random.Random(seed + 41).shuffle(result)
    return result


@torch.no_grad()
def wrong_canonical_recovery_rows(checkpoint: str, moduli: list[int], seed: int, per_modulus: int, max_q: int, steps: int, device: str) -> list[tuple[list[int], list[int], list[int]]]:
    """Frozen-model generated s<N, s!=r states, labeled only with their true r."""
    base = canonical_rows(moduli, seed, per_modulus)
    modulus = torch.tensor([row[1] for row in base], dtype=torch.long, device=device)
    target = torch.tensor([row[0] for row in base], dtype=torch.long, device=device)
    powers = torch.tensor([10**i for i in range(WIDTH)], device=device)
    modulus_value = (modulus * powers).sum(dim=-1)
    teacher = SerialSubtractor().to(device)
    teacher.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    teacher.eval()
    result, seen = [], set()
    for quotient in range(max_q + 1):
        state = torch.tensor([digits(quotient * int("".join(map(str, row[1][::-1]))) + int("".join(map(str, row[0][::-1]))))[::-1] for row in base], dtype=torch.long, device=device)
        for _ in range(steps):
            canonical = (state * powers).sum(dim=-1) < modulus_value
            wrong = canonical & ~(state == target).all(dim=-1)
            for index in wrong.nonzero(as_tuple=False).flatten().tolist():
                row = (tuple(state[index].tolist()), tuple(modulus[index].tolist()), tuple(target[index].tolist()))
                if row not in seen:
                    seen.add(row); result.append(tuple(map(list, row)))
            state = teacher(state, modulus).argmax(dim=-1)
    del teacher
    return result


class SerialSubtractor(nn.Module):
    """A learned six-step recurrence over aligned operand/modulus digits."""

    def __init__(self, width: int = 128) -> None:
        super().__init__()
        self.digit = nn.Embedding(10, width)
        self.place = nn.Embedding(WIDTH, width)
        self.pair = nn.Linear(2 * width, width)
        self.cell = nn.GRUCell(width, width)
        self.head = nn.Linear(width, 10)

    def forward(self, u: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(u, n)[0])

    def encode(self, u: torch.Tensor, n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = torch.zeros(u.shape[0], self.cell.hidden_size, device=u.device)
        states = []
        for position in range(WIDTH):
            place = self.place.weight[position]
            x = torch.tanh(self.pair(torch.cat((self.digit(u[:, position]) + place, self.digit(n[:, position]) + place), dim=-1)))
            state = self.cell(x, state)
            states.append(state)
        return torch.stack(states, dim=1), state


def tensors(batch, device: str):
    return tuple(torch.tensor([row[i] for row in batch], dtype=torch.long, device=device) for i in range(3))


@torch.no_grad()
def evaluate(model: SerialSubtractor, dataset, device: str) -> dict:
    u, n, target = tensors(dataset, device)
    pred = model(u, n).argmax(dim=-1)
    return {
        "exact": float((pred == target).all(dim=-1).float().mean()),
        "token": float((pred == target).float().mean()),
        "per_lsd_position": [float((pred[:, i] == target[:, i]).float().mean()) for i in range(WIDTH)],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--train-moduli", type=int, default=48)
    ap.add_argument("--test-moduli", type=int, default=16)
    ap.add_argument("--per-modulus", type=int, default=128)
    ap.add_argument("--max-train-q", type=int, default=1)
    ap.add_argument("--canonical-identity", action="store_true")
    ap.add_argument("--recovery-checkpoint")
    ap.add_argument("--recovery-max-q", type=int, default=10)
    ap.add_argument("--recovery-steps", type=int, default=24)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    all_moduli = semiprimes(args.seed, args.train_moduli + args.test_moduli)
    train_moduli, test_moduli = all_moduli[:args.train_moduli], all_moduli[args.train_moduli:]
    transitions = rows(train_moduli, args.seed, args.per_modulus, heldout=False, qs=range(1, args.max_train_q + 1))
    identity = canonical_rows(train_moduli, args.seed, args.per_modulus) if args.canonical_identity else []
    recovery = wrong_canonical_recovery_rows(args.recovery_checkpoint, train_moduli, args.seed, args.per_modulus, args.recovery_max_q, args.recovery_steps, args.device) if args.recovery_checkpoint else []
    train = transitions + identity + recovery
    random.Random(args.seed + 91).shuffle(train)
    seen = rows(train_moduli, args.seed, args.per_modulus, heldout=True)
    unseen = rows(test_moduli, args.seed, args.per_modulus, heldout=False)
    model = SerialSubtractor().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({
        "classification": "NEW REIMPLEMENTATION — NOT SUBMISSION-RELEVANT",
        "task": "learned serial subtraction", "train_moduli": train_moduli,
        "test_moduli": test_moduli, "steps": args.steps, "batch_size": args.batch_size,
        "per_modulus": args.per_modulus, "max_train_q": args.max_train_q,
        "canonical_identity": args.canonical_identity, "recovery_checkpoint": args.recovery_checkpoint,
        "recovery_max_q": args.recovery_max_q, "recovery_steps": args.recovery_steps,
        "transition_examples": len(transitions), "identity_examples": len(identity), "recovery_examples": len(recovery),
        "model": "LSD-to-MSD learned GRU digit cell",
    }, indent=2) + "\n")
    start = time.perf_counter()
    with (out / "metrics.jsonl").open("w") as log:
        for step in range(1, args.steps + 1):
            offset = ((step - 1) * args.batch_size) % len(train)
            batch = [train[(offset + i) % len(train)] for i in range(args.batch_size)]
            u, n, target = tensors(batch, args.device)
            loss = F.cross_entropy(model(u, n).reshape(-1, 10), target.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step % 200 == 0 or step == args.steps:
                model.eval()
                record = {"step": step, "loss": float(loss.detach()), "steps_per_sec": step / (time.perf_counter() - start), "seen": evaluate(model, seen, args.device), "unseen_n": evaluate(model, unseen, args.device)}
                log.write(json.dumps(record) + "\n"); log.flush(); print(json.dumps(record), flush=True)
                model.train()
    model.eval()
    report = {"seen": evaluate(model, seen, args.device), "unseen_n": evaluate(model, unseen, args.device), "parameters": sum(p.numel() for p in model.parameters())}
    (out / "eval_report.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save(model.state_dict(), out / "final.pt")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

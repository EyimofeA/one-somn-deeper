"""Teacher-depth test of a fully learned, tied decimal reduction transition.

This is deliberately a diagnostic, not a competition model. Arithmetic occurs
only while constructing labelled training traces. ``ReducerCell.forward`` is a
neural mapping from decimal state/N digits to next-state digits; it contains no
modulo, comparison, quotient, subtraction, or data-dependent loop.
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


N_VALUE = 1349
STATE_WIDTH = 8
N_WIDTH = 4
DEPTHS = (0, 1, 2, 5, 10, 50, 100)


def digits(value: int, width: int) -> list[int]:
    return [int(c) for c in f"{value:0{width}d}"]


class ReducerCell(nn.Module):
    """One fully learned state transition; weights are reused at every depth."""

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.digit = nn.Embedding(10, d_model)
        self.place = nn.Embedding(N_WIDTH + STATE_WIDTH, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=4 * d_model,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.body = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 10)

    def forward(self, n_digits: torch.Tensor, state_digits: torch.Tensor) -> torch.Tensor:
        tokens = torch.cat((n_digits, state_digits), dim=1)
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.digit(tokens) + self.place(positions)[None, :, :]
        return self.head(self.norm(self.body(hidden))[:, -STATE_WIDTH:])


def make_rows(seed: int, count: int, used_remainders: set[int]) -> list[tuple[list[int], list[int], list[int]]]:
    rng = random.Random(seed)
    available = [r for r in range(N_VALUE) if r not in used_remainders]
    if count > len(available):
        raise ValueError("not enough disjoint remainders")
    remainders = rng.sample(available, count)
    used_remainders.update(remainders)
    rows = []
    n = digits(N_VALUE, N_WIDTH)
    for remainder in remainders:
        for depth in DEPTHS:
            current = remainder + depth * N_VALUE
            # Label construction only: the cell must learn this transition.
            next_state = current if depth == 0 else current - N_VALUE
            rows.append((n, digits(current, STATE_WIDTH), digits(next_state, STATE_WIDTH)))
    rng.shuffle(rows)
    return rows


def batch_tensors(rows, device: str):
    n = torch.tensor([row[0] for row in rows], dtype=torch.long, device=device)
    state = torch.tensor([row[1] for row in rows], dtype=torch.long, device=device)
    target = torch.tensor([row[2] for row in rows], dtype=torch.long, device=device)
    return n, state, target


@torch.no_grad()
def terminal_exact(model: ReducerCell, remainders: list[int], depth: int, device: str) -> float:
    n = torch.tensor([digits(N_VALUE, N_WIDTH)] * len(remainders), dtype=torch.long, device=device)
    state = torch.tensor([digits(r + depth * N_VALUE, STATE_WIDTH) for r in remainders], dtype=torch.long, device=device)
    for _ in range(depth):
        state = model(n, state).argmax(dim=-1)
    target = torch.tensor([digits(r, STATE_WIDTH) for r in remainders], dtype=torch.long, device=device)
    return float((state == target).all(dim=-1).float().mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    used: set[int] = set()
    train_rows = make_rows(args.seed, 800, used)
    test_remainders = random.Random(args.seed + 1).sample([r for r in range(N_VALUE) if r not in used], 256)
    model = ReducerCell().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({
        "classification": "NEW DIAGNOSTIC — NOT SUBMISSION-RELEVANT",
        "n": N_VALUE, "depths": DEPTHS, "seed": args.seed, "steps": args.steps,
        "batch_size": args.batch_size, "train_remainders": 800,
        "heldout_remainders": 256, "model": "tied learned decimal transition",
    }, indent=2) + "\n")
    metrics_path = out / "metrics.jsonl"
    start = time.perf_counter()
    with metrics_path.open("w") as metrics:
        for step in range(1, args.steps + 1):
            offset = ((step - 1) * args.batch_size) % len(train_rows)
            rows = [train_rows[(offset + i) % len(train_rows)] for i in range(args.batch_size)]
            n, state, target = batch_tensors(rows, args.device)
            logits = model(n, state)
            loss = F.cross_entropy(logits.reshape(-1, 10), target.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step % 200 == 0 or step == args.steps:
                model.eval()
                exact = {str(depth): terminal_exact(model, test_remainders, depth, args.device) for depth in DEPTHS}
                record = {
                    "step": step, "loss": float(loss.detach()), "steps_per_sec": step / (time.perf_counter() - start),
                    "terminal_exact": exact,
                }
                metrics.write(json.dumps(record) + "\n")
                metrics.flush()
                print(json.dumps(record), flush=True)
                model.train()
    model.eval()
    final = {str(depth): terminal_exact(model, test_remainders, depth, args.device) for depth in DEPTHS}
    report = {
        "terminal_exact": final,
        "parameters": sum(p.numel() for p in model.parameters()),
        "steps_per_sec": args.steps / (time.perf_counter() - start),
        "note": "Teacher depth is evaluator-supplied; this is not legal submission inference.",
    }
    (out / "eval_report.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save(model.state_dict(), out / "final.pt")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

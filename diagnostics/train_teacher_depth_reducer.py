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

    def __init__(self, d_model: int = 128, with_stop_head: bool = False) -> None:
        super().__init__()
        self.with_stop_head = with_stop_head
        self.digit = nn.Embedding(10, d_model)
        self.place = nn.Embedding(N_WIDTH + STATE_WIDTH, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=4 * d_model,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.body = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 10)
        self.stop_head = nn.Linear(d_model, 1) if with_stop_head else None

    def _encode(self, n_digits: torch.Tensor, state_digits: torch.Tensor) -> torch.Tensor:
        tokens = torch.cat((n_digits, state_digits), dim=1)
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.digit(tokens) + self.place(positions)[None, :, :]
        return self.norm(self.body(hidden))

    def forward(self, n_digits: torch.Tensor, state_digits: torch.Tensor) -> torch.Tensor:
        return self.head(self._encode(n_digits, state_digits)[:, -STATE_WIDTH:])

    def stop_logits(self, n_digits: torch.Tensor, state_digits: torch.Tensor) -> torch.Tensor:
        if self.stop_head is None:
            raise RuntimeError("stop head was not enabled")
        return self.stop_head(self._encode(n_digits, state_digits)[:, -STATE_WIDTH:].mean(dim=1)).squeeze(-1)


def make_rows(
    seed: int, count: int, used_remainders: set[int], depths: tuple[int, ...] = DEPTHS
) -> list[tuple[list[int], list[int], list[int], bool]]:
    rng = random.Random(seed)
    available = [r for r in range(N_VALUE) if r not in used_remainders]
    if count > len(available):
        raise ValueError("not enough disjoint remainders")
    remainders = rng.sample(available, count)
    used_remainders.update(remainders)
    rows = []
    n = digits(N_VALUE, N_WIDTH)
    for remainder in remainders:
        for depth in depths:
            current = remainder + depth * N_VALUE
            # Label construction only: the cell must learn this transition.
            next_state = current if depth == 0 else current - N_VALUE
            rows.append((n, digits(current, STATE_WIDTH), digits(next_state, STATE_WIDTH), depth == 0))
    rng.shuffle(rows)
    return rows


def batch_tensors(rows, device: str):
    n = torch.tensor([row[0] for row in rows], dtype=torch.long, device=device)
    state = torch.tensor([row[1] for row in rows], dtype=torch.long, device=device)
    target = torch.tensor([row[2] for row in rows], dtype=torch.long, device=device)
    stop = torch.tensor([row[3] for row in rows], dtype=torch.float32, device=device)
    return n, state, target, stop


@torch.no_grad()
def terminal_exact(model: ReducerCell, remainders: list[int], depth: int, device: str) -> float:
    n = torch.tensor([digits(N_VALUE, N_WIDTH)] * len(remainders), dtype=torch.long, device=device)
    state = torch.tensor([digits(r + depth * N_VALUE, STATE_WIDTH) for r in remainders], dtype=torch.long, device=device)
    for _ in range(depth):
        state = model(n, state).argmax(dim=-1)
    target = torch.tensor([digits(r, STATE_WIDTH) for r in remainders], dtype=torch.long, device=device)
    return float((state == target).all(dim=-1).float().mean())


@torch.no_grad()
def autonomous_halt_report(model: ReducerCell, remainders: list[int], device: str, max_steps: int = 120) -> dict:
    """Bounded diagnostic loop; q is used only to score, never to control it."""
    reports = {}
    for depth in DEPTHS:
        n = torch.tensor([digits(N_VALUE, N_WIDTH)] * len(remainders), dtype=torch.long, device=device)
        state = torch.tensor([digits(r + depth * N_VALUE, STATE_WIDTH) for r in remainders], dtype=torch.long, device=device)
        target = torch.tensor([digits(r, STATE_WIDTH) for r in remainders], dtype=torch.long, device=device)
        active = torch.ones(len(remainders), dtype=torch.bool, device=device)
        stopped_at = torch.full((len(remainders),), -1, dtype=torch.long, device=device)
        for iteration in range(max_steps + 1):
            stop = model.stop_logits(n, state).sigmoid() >= 0.5
            newly_stopped = active & stop
            stopped_at[newly_stopped] = iteration
            active = active & ~stop
            if not active.any() or iteration == max_steps:
                break
            next_state = model(n, state).argmax(dim=-1)
            state = torch.where(active[:, None], next_state, state)
        exact = (state == target).all(dim=-1)
        correct_depth = stopped_at == depth
        reports[str(depth)] = {
            "remainder_exact": float(exact.float().mean()),
            "halting_accuracy": float(correct_depth.float().mean()),
            "mean_iterations": float(torch.where(
                stopped_at < 0, torch.full_like(stopped_at, max_steps), stopped_at
            ).float().mean()),
            "stopped_early": float((stopped_at.ge(0) & (stopped_at < depth)).float().mean()),
            "stopped_late": float((stopped_at > depth).float().mean()),
            "failed_to_stop": float((stopped_at < 0).float().mean()),
            "wrong_remainder_after_correct_depth": float((correct_depth & ~exact).float().mean()),
        }
    return reports


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-train-depth", type=int, default=None)
    ap.add_argument("--learn-stop", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    used: set[int] = set()
    train_depths = DEPTHS if args.max_train_depth is None else tuple(range(args.max_train_depth + 1))
    train_rows = make_rows(args.seed, 800, used, train_depths)
    test_remainders = random.Random(args.seed + 1).sample([r for r in range(N_VALUE) if r not in used], 256)
    model = ReducerCell(with_stop_head=args.learn_stop).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({
        "classification": "NEW DIAGNOSTIC — NOT SUBMISSION-RELEVANT",
        "n": N_VALUE, "depths": train_depths, "seed": args.seed, "steps": args.steps,
        "batch_size": args.batch_size, "train_remainders": 800,
        "heldout_remainders": 256, "model": "tied learned decimal transition",
        "learn_stop": args.learn_stop,
    }, indent=2) + "\n")
    metrics_path = out / "metrics.jsonl"
    start = time.perf_counter()
    with metrics_path.open("w") as metrics:
        for step in range(1, args.steps + 1):
            offset = ((step - 1) * args.batch_size) % len(train_rows)
            rows = [train_rows[(offset + i) % len(train_rows)] for i in range(args.batch_size)]
            n, state, target, stop_target = batch_tensors(rows, args.device)
            logits = model(n, state)
            transition_loss = F.cross_entropy(logits.reshape(-1, 10), target.reshape(-1))
            stop_loss = torch.zeros((), device=args.device)
            if args.learn_stop:
                pos_weight = torch.tensor(float(args.max_train_depth + 1), device=args.device)
                stop_loss = F.binary_cross_entropy_with_logits(
                    model.stop_logits(n, state), stop_target, pos_weight=pos_weight
                )
            loss = transition_loss + stop_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step % 200 == 0 or step == args.steps:
                model.eval()
                exact = {str(depth): terminal_exact(model, test_remainders, depth, args.device) for depth in DEPTHS}
                record = {
                    "step": step, "loss": float(loss.detach()), "transition_loss": float(transition_loss.detach()),
                    "stop_loss": float(stop_loss.detach()), "steps_per_sec": step / (time.perf_counter() - start),
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
    if args.learn_stop:
        report["autonomous_halt"] = autonomous_halt_report(model, test_remainders, args.device)
    (out / "eval_report.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save(model.state_dict(), out / "final.pt")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

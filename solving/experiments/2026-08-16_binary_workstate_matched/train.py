"""Matched binary work-state diagnostic for reduction versus fused T=1.

Research only: Python constructs synthetic inputs and final modular targets.
The model receives no quotient, comparison, subtraction, carry, square label,
or execution trace.  The two arms differ only in the source tape: exact x^2
bits for ``exact_square`` versus zero-padded x bits for ``fused_x``.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim._muon import _zeropower_via_newtonschulz

OPERAND_BITS = 11
WORKSPACE_BITS = 22
LANES = 4


def is_prime(value: int) -> bool:
    if value < 2:
        return False
    for divisor in range(2, int(value**0.5) + 1):
        if value % divisor == 0:
            return False
    return True


def split_values(seed: int) -> tuple[list[int], list[int]]:
    values = list(range(1 << OPERAND_BITS))
    random.Random(seed + 50_000).shuffle(values)
    return values[:1600], values[1600:]


def split_fixed_values(
    modulus: int, seed: int
) -> tuple[list[int], list[int], list[int]]:
    values = list(range(modulus))
    random.Random(seed + 50_000).shuffle(values)
    train_end = int(0.70 * len(values))
    validation_end = int(0.85 * len(values))
    return values[:train_end], values[train_end:validation_end], values[validation_end:]


def split_moduli(seed: int) -> tuple[list[int], list[int]]:
    primes = [value for value in range(11, 256) if is_prime(value)]
    values = sorted(
        {
            left * right
            for index, left in enumerate(primes)
            for right in primes[index + 1 :]
            if 10 <= (left * right).bit_length() <= OPERAND_BITS
        }
    )
    random.Random(seed + 60_000).shuffle(values)
    return values[:90], values[90:120]


def make_rows(
    moduli: list[int], x_values: list[int], count: int, seed: int
) -> list[tuple[int, int, int]]:
    rng = random.Random(seed)
    valid_by_modulus = {n: [x for x in x_values if x < n] for n in moduli}
    rows = []
    for _ in range(count):
        n = rng.choice(moduli)
        x = rng.choice(valid_by_modulus[n])
        rows.append((n, x, (x * x) % n))
    return rows


def make_fixed_rows(modulus: int, values: list[int]) -> list[tuple[int, int, int]]:
    return [(modulus, x, (x * x) % modulus) for x in values]


def bit_tensor(values: torch.Tensor, width: int) -> torch.Tensor:
    shifts = torch.arange(width, device=values.device)
    return ((values[:, None] >> shifts) & 1).long()


def tensor_batch(
    rows: list[tuple[int, int, int]], mode: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = torch.tensor([row[0] for row in rows], dtype=torch.long, device=device)
    x = torch.tensor([row[1] for row in rows], dtype=torch.long, device=device)
    target = torch.tensor([row[2] for row in rows], dtype=torch.long, device=device)
    source = x * x if mode == "exact_square" else x
    return (
        bit_tensor(source, WORKSPACE_BITS),
        bit_tensor(n, WORKSPACE_BITS),
        bit_tensor(target, OPERAND_BITS).float(),
    )


class Cell(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.update = nn.Conv2d(channels, channels, 3, padding=1)
        self.reset = nn.Conv2d(channels, channels, 3, padding=1)
        self.candidate = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(
        self, state: torch.Tensor, dropout_mask: torch.Tensor | None
    ) -> torch.Tensor:
        update = torch.sigmoid(self.update(state))
        reset = torch.sigmoid(self.reset(state))
        candidate = torch.tanh(self.candidate(reset * state))
        if dropout_mask is not None:
            candidate = candidate * dropout_mask
        return (1.0 - update) * state + update * candidate


class BinaryWorkState(nn.Module):
    def __init__(self, channels: int, updates: int, dropout: float) -> None:
        super().__init__()
        self.channels = channels
        self.updates = updates
        self.dropout = dropout
        self.bit_embedding = nn.Embedding(2, channels)
        self.source_role = nn.Parameter(torch.randn(channels) * 0.02)
        self.modulus_role = nn.Parameter(torch.randn(channels) * 0.02)
        self.work_role = nn.Parameter(torch.randn(channels) * 0.02)
        self.boundaries = nn.Parameter(torch.randn(2, channels) * 0.02)
        self.cell = Cell(channels)
        self.readout = nn.Conv1d(channels, 1, 1)

    def forward(self, source_bits: torch.Tensor, modulus_bits: torch.Tensor) -> torch.Tensor:
        batch = source_bits.shape[0]
        source = self.bit_embedding(source_bits).transpose(1, 2)
        modulus = self.bit_embedding(modulus_bits).transpose(1, 2)
        source = source + self.source_role[None, :, None]
        modulus = modulus + self.modulus_role[None, :, None]
        state = source.new_zeros(batch, self.channels, LANES, WORKSPACE_BITS)
        state[:, :, 2] = source + self.work_role[None, :, None]
        dropout_mask = None
        if self.training and self.dropout:
            keep = 1.0 - self.dropout
            dropout_mask = torch.empty(
                batch, self.channels, 1, 1, device=state.device
            ).bernoulli_(keep) / keep
        for _ in range(self.updates):
            visible = state.clone()
            visible[:, :, 0] = source
            visible[:, :, 1] = modulus
            visible[:, :, :, 0] += self.boundaries[0][None, :, None]
            visible[:, :, :, -1] += self.boundaries[1][None, :, None]
            state = self.cell(visible, dropout_mask)
        return self.readout(state[:, :, 2, :OPERAND_BITS]).squeeze(1)


class ConvMuon(Optimizer):
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95) -> None:
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=1e-5))

    @torch.no_grad()
    def step(self, closure=None):
        del closure
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.reshape(parameter.shape[0], -1)
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(gradient)
                buffer = state["momentum_buffer"]
                buffer.lerp_(gradient, 1.0 - group["momentum"])
                update = gradient.lerp(buffer, group["momentum"])
                update = _zeropower_via_newtonschulz(
                    update, (3.4445, -4.775, 2.0315), 5, 1e-7
                )
                ratio = max(1.0, gradient.shape[0] / gradient.shape[1]) ** 0.5
                parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                parameter.add_(update.reshape_as(parameter), alpha=-group["lr"] * ratio)


def make_optimizers(
    model: nn.Module, optimizer_name: str, learning_rate: float
) -> list[Optimizer]:
    if optimizer_name == "adamw":
        return [
            torch.optim.AdamW(
                model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=1e-5
            )
        ]
    matrix = [parameter for parameter in model.parameters() if parameter.ndim >= 2]
    scalar = [parameter for parameter in model.parameters() if parameter.ndim < 2]
    return [
        ConvMuon(matrix),
        torch.optim.AdamW(scalar, lr=3e-4, betas=(0.9, 0.95), weight_decay=1e-5),
    ]


@torch.no_grad()
def evaluate(
    model: nn.Module,
    rows: list[tuple[int, int, int]],
    mode: str,
    device: torch.device,
) -> dict:
    model.eval()
    exact = bit_correct = bit_total = 0
    positions = torch.zeros(OPERAND_BITS, dtype=torch.long)
    for start in range(0, len(rows), 512):
        source, modulus, target = tensor_batch(rows[start : start + 512], mode, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(source, modulus).gt(0)
        matches = prediction == target.bool()
        exact += int(matches.all(-1).sum())
        bit_correct += int(matches.sum())
        bit_total += matches.numel()
        positions += matches.sum(0).cpu()
    return {
        "exact": exact / len(rows),
        "bit_accuracy": bit_correct / bit_total,
        "bit_accuracy_lsb_first": (positions / len(rows)).tolist(),
        "examples": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("exact_square", "fused_x"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--updates", type=int, default=44)
    parser.add_argument("--dropout", type=float, default=0.09)
    parser.add_argument("--seed", type=int, default=74)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--optimizer", choices=("muon", "adamw"), default="muon")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "warmup_cosine", "warmup_inverse_sqrt"),
        default="constant",
    )
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--final-learning-rate", type=float, default=1e-4)
    parser.add_argument("--fixed-modulus", type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")

    if args.fixed_modulus is None:
        train_x, heldout_x = split_values(args.seed)
        train_n, unseen_n = split_moduli(args.seed)
        train = make_rows(train_n, train_x, 100_000, args.seed + 1)
        validation = make_rows(train_n, heldout_x, 5_000, args.seed + 2)
        evaluation_sets = {
            "train": train,
            "validation_unseen_x_seen_n": validation,
            "audit_seen_x_unseen_n": make_rows(
                unseen_n, train_x, 5_000, args.seed + 3
            ),
            "audit_unseen_x_unseen_n": make_rows(
                unseen_n, heldout_x, 5_000, args.seed + 4
            ),
        }
    else:
        if not 2 <= args.fixed_modulus < (1 << OPERAND_BITS):
            raise ValueError("fixed modulus must fit in OPERAND_BITS")
        train_x, validation_x, audit_x = split_fixed_values(
            args.fixed_modulus, args.seed
        )
        train = make_fixed_rows(args.fixed_modulus, train_x)
        validation = make_fixed_rows(args.fixed_modulus, validation_x)
        evaluation_sets = {
            "train": train,
            "validation_unseen_x_same_n": validation,
            "audit_unseen_x_same_n": make_fixed_rows(args.fixed_modulus, audit_x),
        }

    raw_model = BinaryWorkState(args.channels, args.updates, args.dropout).to(device)
    model = torch.compile(raw_model) if args.compile else raw_model
    optimizers = make_optimizers(model, args.optimizer, args.learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 5)
    best_validation = -1.0
    best_step = 0
    best_state = None
    curve = []
    started = time.perf_counter()

    for step in range(1, args.steps + 1):
        if args.optimizer == "adamw" and args.lr_schedule != "constant":
            if step <= args.warmup_steps:
                learning_rate = args.learning_rate * step / args.warmup_steps
            elif args.lr_schedule == "warmup_cosine":
                progress = (step - args.warmup_steps) / (args.steps - args.warmup_steps)
                learning_rate = args.final_learning_rate + (
                    args.learning_rate - args.final_learning_rate
                ) * 0.5 * (1.0 + math.cos(math.pi * progress))
            else:
                learning_rate = args.learning_rate * math.sqrt(args.warmup_steps / step)
            optimizers[0].param_groups[0]["lr"] = learning_rate
        indices = torch.randint(len(train), (args.batch_size,), generator=generator).tolist()
        sample = [train[index] for index in indices]
        source, modulus, target = tensor_batch(sample, args.mode, device)
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(source, modulus)
            loss = F.binary_cross_entropy_with_logits(logits, target)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for optimizer in optimizers:
            optimizer.step()
        if args.optimizer == "muon":
            progress = min(1.0, max(0.0, (step - 1000) / 4000))
            optimizers[0].param_groups[0]["lr"] = 0.002 + 0.018 * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )

        if step == 1 or step % args.eval_every == 0:
            train_probe = evaluate(model, train[:5000], args.mode, device)
            validation_metrics = evaluate(model, validation, args.mode, device)
            record = {
                "type": "progress",
                "mode": args.mode,
                "step": step,
                "examples": step * args.batch_size,
                "seconds": time.perf_counter() - started,
                "loss": float(loss.detach()),
                "learning_rate": optimizers[0].param_groups[0]["lr"],
                "train_exact": train_probe["exact"],
                "validation_exact": validation_metrics["exact"],
            }
            curve.append(record)
            print(json.dumps(record), flush=True)
            if validation_metrics["exact"] > best_validation:
                best_validation = validation_metrics["exact"]
                best_step = step
                best_state = copy.deepcopy(model.state_dict())

    assert best_state is not None
    model.load_state_dict(best_state)
    report = {
        "mode": args.mode,
        "fixed_modulus": args.fixed_modulus,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "lr_schedule": args.lr_schedule,
        "seed": args.seed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "channels": args.channels,
        "updates": args.updates,
        "dropout": args.dropout,
        "steps": args.steps,
        "examples": args.steps * args.batch_size,
        "best_step": best_step,
        "elapsed_seconds": time.perf_counter() - started,
        "split": {name: len(rows) for name, rows in evaluation_sets.items()},
        "selected": {
            name: evaluate(model, rows, args.mode, device)
            for name, rows in evaluation_sets.items()
        },
        "curve": curve,
    }
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(state, args.out / "model_best.pt")
    (args.out / "eval_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / "source.py").write_text(Path(__file__).read_text())
    print(json.dumps({"type": "final", **report}), flush=True)


if __name__ == "__main__":
    main()

"""H13: learn x^2 mod N by scanning x one bit at a time.

The model receives one prompt-visible bit of x every four recurrent updates,
plus immutable N.  It is never given square, quotient, carry, comparison,
subtraction, or intermediate residue targets.  Training loss is applied only
to the final x^2 mod N bits.  Prefix readouts are diagnostics opened after
validation selection; they never contribute gradients or checkpoint choice.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim._muon import _zeropower_via_newtonschulz


OPERAND_BITS = 11
LANES = 4
MICRO_UPDATES = 4


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


def rows_tensor(rows: list[tuple[int, int, int]], device: torch.device) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.long, device=device)


def bit_tensor(values: torch.Tensor, width: int) -> torch.Tensor:
    shifts = torch.arange(width, device=values.device)
    return ((values[:, None] >> shifts) & 1).long()


def tensor_batch(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n, x, target = rows.unbind(1)
    return (
        bit_tensor(x, OPERAND_BITS),
        bit_tensor(n, OPERAND_BITS),
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


class PrefixResidueState(nn.Module):
    """One generic tied cell receives x in MSB-to-LSB order."""

    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.bit_embedding = nn.Embedding(2, channels)
        self.input_role = nn.Parameter(torch.randn(channels) * 0.02)
        self.modulus_role = nn.Parameter(torch.randn(channels) * 0.02)
        self.work_role = nn.Parameter(torch.randn(channels) * 0.02)
        self.scratch_role = nn.Parameter(torch.randn(channels) * 0.02)
        self.phase = nn.Parameter(torch.randn(MICRO_UPDATES, channels) * 0.02)
        self.boundaries = nn.Parameter(torch.randn(2, channels) * 0.02)
        self.cell = Cell(channels)
        self.readout = nn.Conv1d(channels, 1, 1)

    def decode(self, state: torch.Tensor) -> torch.Tensor:
        return self.readout(state[:, :, 2]).squeeze(1)

    def forward(
        self,
        x_bits: torch.Tensor,
        modulus_bits: torch.Tensor,
        return_prefixes: bool = False,
    ) -> torch.Tensor:
        batch = x_bits.shape[0]
        modulus = self.bit_embedding(modulus_bits).transpose(1, 2)
        modulus = modulus + self.modulus_role[None, :, None]
        state = modulus.new_zeros(batch, self.channels, LANES, OPERAND_BITS)
        state[:, :, 2] = self.work_role[None, :, None]
        state[:, :, 3] = self.scratch_role[None, :, None]
        dropout_mask = None
        if self.training and self.dropout:
            keep = 1.0 - self.dropout
            dropout_mask = torch.empty(
                batch, self.channels, 1, 1, device=state.device
            ).bernoulli_(keep) / keep

        prefix_logits = []
        for bit_index in range(OPERAND_BITS - 1, -1, -1):
            incoming = self.bit_embedding(x_bits[:, bit_index])
            incoming = incoming + self.input_role[None, :]
            incoming = incoming[:, :, None].expand(-1, -1, OPERAND_BITS)
            for phase_index in range(MICRO_UPDATES):
                visible = state.clone()
                visible[:, :, 0] = incoming + self.phase[phase_index][None, :, None]
                visible[:, :, 1] = modulus
                visible[:, :, :, 0] += self.boundaries[0][None, :, None]
                visible[:, :, :, -1] += self.boundaries[1][None, :, None]
                state = self.cell(visible, dropout_mask)
            if return_prefixes:
                prefix_logits.append(self.decode(state))

        if return_prefixes:
            return torch.stack(prefix_logits, dim=1)
        return self.decode(state)


class ConvMuon(Optimizer):
    def __init__(
        self, params, lr: float = 0.006, momentum: float = 0.95,
        weight_decay: float = 0.1,
    ) -> None:
        super().__init__(
            params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        )

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
    model: nn.Module, muon_learning_rate: float, muon_weight_decay: float
) -> list[Optimizer]:
    matrix = [parameter for parameter in model.parameters() if parameter.ndim >= 2]
    scalar = [parameter for parameter in model.parameters() if parameter.ndim < 2]
    return [
        ConvMuon(matrix, lr=muon_learning_rate, weight_decay=muon_weight_decay),
        torch.optim.AdamW(
            scalar, lr=3e-4, betas=(0.9, 0.95), weight_decay=1e-5
        ),
    ]


@torch.no_grad()
def evaluate(
    model: nn.Module, rows: torch.Tensor, batch_size: int
) -> dict:
    model.eval()
    exact = bit_correct = bit_total = 0
    positions = torch.zeros(OPERAND_BITS, dtype=torch.long)
    for start in range(0, len(rows), batch_size):
        x, modulus, target = tensor_batch(rows[start : start + batch_size])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(x, modulus).gt(0)
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


@torch.no_grad()
def evaluate_prefixes(
    model: PrefixResidueState, rows: torch.Tensor, batch_size: int
) -> list[dict]:
    """Diagnostic only: compare every hidden-prefix readout after selection."""
    model.eval()
    exact = torch.zeros(OPERAND_BITS, dtype=torch.long)
    bit_correct = torch.zeros(OPERAND_BITS, dtype=torch.long)
    bit_total = torch.zeros(OPERAND_BITS, dtype=torch.long)
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        n, x, _ = batch.unbind(1)
        x_bits = bit_tensor(x, OPERAND_BITS)
        n_bits = bit_tensor(n, OPERAND_BITS)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predictions = model(x_bits, n_bits, return_prefixes=True).gt(0)
        for step in range(1, OPERAND_BITS + 1):
            prefix = x >> (OPERAND_BITS - step)
            target = bit_tensor(torch.remainder(prefix * prefix, n), OPERAND_BITS).bool()
            matches = predictions[:, step - 1] == target
            exact[step - 1] += matches.all(-1).sum().cpu()
            bit_correct[step - 1] += matches.sum().cpu()
            bit_total[step - 1] += matches.numel()
    return [
        {
            "prefix_bits": step + 1,
            "exact": int(exact[step]) / len(rows),
            "bit_accuracy": int(bit_correct[step]) / int(bit_total[step]),
        }
        for step in range(OPERAND_BITS)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--channels", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.09)
    parser.add_argument("--seed", type=int, default=74)
    parser.add_argument("--muon-learning-rate", type=float, default=0.006)
    parser.add_argument("--muon-weight-decay", type=float, default=0.1)
    parser.add_argument("--muon-warmup-steps", type=int, default=250)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")

    train_x, heldout_x = split_values(args.seed)
    train_n, unseen_n = split_moduli(args.seed)
    rows = {
        "train": rows_tensor(
            make_rows(train_n, train_x, 100_000, args.seed + 1), device
        ),
        "validation_unseen_x_seen_n": rows_tensor(
            make_rows(train_n, heldout_x, 5_000, args.seed + 2), device
        ),
        "audit_seen_x_unseen_n": rows_tensor(
            make_rows(unseen_n, train_x, 5_000, args.seed + 3), device
        ),
        "audit_unseen_x_unseen_n": rows_tensor(
            make_rows(unseen_n, heldout_x, 5_000, args.seed + 4), device
        ),
    }

    raw_model = PrefixResidueState(args.channels, args.dropout).to(device)
    model = torch.compile(raw_model) if args.compile else raw_model
    optimizers = make_optimizers(
        raw_model, args.muon_learning_rate, args.muon_weight_decay
    )
    generator = torch.Generator(device=device).manual_seed(args.seed + 5)
    best_validation = -1.0
    best_step = 0
    best_state = None
    curve = []
    started = time.perf_counter()

    for step in range(1, args.steps + 1):
        optimizers[0].param_groups[0]["lr"] = args.muon_learning_rate * min(
            1.0, step / args.muon_warmup_steps
        )
        indices = torch.randint(
            len(rows["train"]), (args.batch_size,), generator=generator, device=device
        )
        x, modulus, target = tensor_batch(rows["train"][indices])
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x, modulus)
            loss = F.binary_cross_entropy_with_logits(logits, target)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        for optimizer in optimizers:
            optimizer.step()

        if step == 1 or step % args.eval_every == 0:
            train_metrics = evaluate(model, rows["train"][:5_000], args.batch_size)
            validation_metrics = evaluate(
                model, rows["validation_unseen_x_seen_n"], args.batch_size
            )
            record = {
                "type": "progress",
                "step": step,
                "examples": step * args.batch_size,
                "seconds": time.perf_counter() - started,
                "loss": float(loss.detach()),
                "learning_rate": optimizers[0].param_groups[0]["lr"],
                "train_exact": train_metrics["exact"],
                "validation_exact": validation_metrics["exact"],
            }
            curve.append(record)
            print(json.dumps(record), flush=True)
            if validation_metrics["exact"] > best_validation:
                best_validation = validation_metrics["exact"]
                best_step = step
                best_state = copy.deepcopy(raw_model.state_dict())

    if best_state is None:
        raise RuntimeError("checkpoint selection failed")
    raw_model.load_state_dict(best_state)
    selected = {
        name: evaluate(model, split_rows, args.batch_size)
        for name, split_rows in rows.items()
    }
    prefix_diagnostics = {
        name: evaluate_prefixes(raw_model, split_rows, args.batch_size)
        for name, split_rows in rows.items()
        if name != "train"
    }
    report = {
        "architecture": "H13 bit-serial prefix-of-x residue state",
        "loss": "final 11-bit residue BCE only",
        "intermediate_supervision": False,
        "seed": args.seed,
        "parameters": sum(parameter.numel() for parameter in raw_model.parameters()),
        "channels": args.channels,
        "micro_updates_per_bit": MICRO_UPDATES,
        "total_updates": OPERAND_BITS * MICRO_UPDATES,
        "dropout": args.dropout,
        "steps": args.steps,
        "examples": args.steps * args.batch_size,
        "muon_learning_rate": args.muon_learning_rate,
        "muon_weight_decay": args.muon_weight_decay,
        "muon_warmup_steps": args.muon_warmup_steps,
        "best_step": best_step,
        "elapsed_seconds": time.perf_counter() - started,
        "split": {name: len(split_rows) for name, split_rows in rows.items()},
        "selected": selected,
        "prefix_diagnostics_not_used_for_selection": prefix_diagnostics,
        "curve": curve,
    }
    state = {key: value.detach().cpu() for key, value in raw_model.state_dict().items()}
    torch.save(state, args.out / "model_best.pt")
    (args.out / "eval_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / "source.py").write_text(Path(__file__).read_text())
    print(json.dumps({"type": "final", **report}), flush=True)


if __name__ == "__main__":
    main()

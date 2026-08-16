"""H13 follow-up: prompt-visible significant-bit curriculum.

The processor, width, optimizer, dropout, and final-only residue loss are
unchanged. Each example consumes only the significant binary digits of x.
Training progressively admits longer x values. Every target is the example's
ordinary final x^2 mod N label; there are no prefix or intermediate targets.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def load_base(path: Path):
    spec = importlib.util.spec_from_file_location("h13_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base(Path(__file__).with_name("train.py"))


class PrefixCurriculumState(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.bit_embedding = nn.Embedding(2, channels)
        self.input_role = nn.Parameter(torch.randn(channels) * 0.02)
        self.modulus_role = nn.Parameter(torch.randn(channels) * 0.02)
        self.work_role = nn.Parameter(torch.randn(channels) * 0.02)
        self.scratch_role = nn.Parameter(torch.randn(channels) * 0.02)
        self.phase = nn.Parameter(torch.randn(BASE.MICRO_UPDATES, channels) * 0.02)
        self.boundaries = nn.Parameter(torch.randn(2, channels) * 0.02)
        self.cell = BASE.Cell(channels)
        self.readout = nn.Conv1d(channels, 1, 1)

    def forward(
        self, x_bits: torch.Tensor, modulus_bits: torch.Tensor,
        active_bits: torch.Tensor,
    ) -> torch.Tensor:
        batch = x_bits.shape[0]
        modulus = self.bit_embedding(modulus_bits).transpose(1, 2)
        modulus = modulus + self.modulus_role[None, :, None]
        state = modulus.new_zeros(
            batch, self.channels, BASE.LANES, BASE.OPERAND_BITS
        )
        state[:, :, 2] = self.work_role[None, :, None]
        state[:, :, 3] = self.scratch_role[None, :, None]
        dropout_mask = None
        if self.training and self.dropout:
            keep = 1.0 - self.dropout
            dropout_mask = torch.empty(
                batch, self.channels, 1, 1, device=state.device
            ).bernoulli_(keep) / keep

        for bit_index in range(BASE.OPERAND_BITS - 1, -1, -1):
            incoming = self.bit_embedding(x_bits[:, bit_index])
            incoming = incoming + self.input_role[None, :]
            incoming = incoming[:, :, None].expand(-1, -1, BASE.OPERAND_BITS)
            active = (active_bits > bit_index)[:, None, None, None]
            for phase_index in range(BASE.MICRO_UPDATES):
                visible = state.clone()
                visible[:, :, 0] = (
                    incoming + self.phase[phase_index][None, :, None]
                )
                visible[:, :, 1] = modulus
                visible[:, :, :, 0] += self.boundaries[0][None, :, None]
                visible[:, :, :, -1] += self.boundaries[1][None, :, None]
                candidate = self.cell(visible, dropout_mask)
                state = torch.where(active, candidate, state)
        return self.readout(state[:, :, 2]).squeeze(1)


def active_width(x: torch.Tensor) -> torch.Tensor:
    return torch.where(
        x > 0,
        torch.floor(torch.log2(x.float())).long() + 1,
        torch.ones_like(x),
    )


def tensor_batch(rows: torch.Tensor):
    n, x, target = rows.unbind(1)
    return (
        BASE.bit_tensor(x, BASE.OPERAND_BITS),
        BASE.bit_tensor(n, BASE.OPERAND_BITS),
        active_width(x),
        BASE.bit_tensor(target, BASE.OPERAND_BITS).float(),
    )


@torch.no_grad()
def evaluate(model, rows: torch.Tensor, batch_size: int) -> dict:
    model.eval()
    exact = bit_correct = bit_total = 0
    positions = torch.zeros(BASE.OPERAND_BITS, dtype=torch.long)
    length_exact = torch.zeros(BASE.OPERAND_BITS, dtype=torch.long)
    length_total = torch.zeros(BASE.OPERAND_BITS, dtype=torch.long)
    for start in range(0, len(rows), batch_size):
        x, modulus, widths, target = tensor_batch(rows[start : start + batch_size])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(x, modulus, widths).gt(0)
        matches = prediction == target.bool()
        row_exact = matches.all(-1)
        exact += int(row_exact.sum())
        bit_correct += int(matches.sum())
        bit_total += matches.numel()
        positions += matches.sum(0).cpu()
        for width in range(1, BASE.OPERAND_BITS + 1):
            mask = widths == width
            length_exact[width - 1] += row_exact[mask].sum().cpu()
            length_total[width - 1] += mask.sum().cpu()
    by_length = []
    for index in range(BASE.OPERAND_BITS):
        total = int(length_total[index])
        by_length.append(
            {
                "bits": index + 1,
                "exact": int(length_exact[index]) / total if total else None,
                "examples": total,
            }
        )
    return {
        "exact": exact / len(rows),
        "bit_accuracy": bit_correct / bit_total,
        "bit_accuracy_lsb_first": (positions / len(rows)).tolist(),
        "exact_by_x_bit_length": by_length,
        "examples": len(rows),
    }


def curriculum_cap(step: int) -> int:
    boundaries = (
        (500, 4), (1_000, 5), (1_500, 6), (2_500, 7),
        (3_500, 8), (4_500, 9), (5_500, 10),
    )
    for last_step, cap in boundaries:
        if step <= last_step:
            return cap
    return 11


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
    args.out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    train_x, heldout_x = BASE.split_values(args.seed)
    train_n, unseen_n = BASE.split_moduli(args.seed)
    train_by_cap = {}
    for cap in range(4, BASE.OPERAND_BITS + 1):
        eligible = [x for x in train_x if x < (1 << cap)]
        seed = args.seed + 1 if cap == BASE.OPERAND_BITS else args.seed + 100 + cap
        train_by_cap[cap] = BASE.rows_tensor(
            BASE.make_rows(train_n, eligible, 100_000, seed), device
        )
    rows = {
        "train": train_by_cap[BASE.OPERAND_BITS],
        "validation_unseen_x_seen_n": BASE.rows_tensor(
            BASE.make_rows(train_n, heldout_x, 5_000, args.seed + 2), device
        ),
        "audit_seen_x_unseen_n": BASE.rows_tensor(
            BASE.make_rows(unseen_n, train_x, 5_000, args.seed + 3), device
        ),
        "audit_unseen_x_unseen_n": BASE.rows_tensor(
            BASE.make_rows(unseen_n, heldout_x, 5_000, args.seed + 4), device
        ),
    }

    raw_model = PrefixCurriculumState(args.channels, args.dropout).to(device)
    model = torch.compile(raw_model) if args.compile else raw_model
    optimizers = BASE.make_optimizers(
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
        cap = curriculum_cap(step)
        train_rows = train_by_cap[cap]
        indices = torch.randint(
            len(train_rows), (args.batch_size,), generator=generator, device=device
        )
        x, modulus, widths, target = tensor_batch(train_rows[indices])
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x, modulus, widths)
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
                "type": "progress", "step": step, "curriculum_cap": cap,
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
    report = {
        "architecture": "H13 significant-bit state with length curriculum",
        "loss": "provided final 11-bit residue BCE only",
        "intermediate_supervision": False,
        "curriculum": [
            {"through_step": 500, "max_x_bits": 4},
            {"through_step": 1_000, "max_x_bits": 5},
            {"through_step": 1_500, "max_x_bits": 6},
            {"through_step": 2_500, "max_x_bits": 7},
            {"through_step": 3_500, "max_x_bits": 8},
            {"through_step": 4_500, "max_x_bits": 9},
            {"through_step": 5_500, "max_x_bits": 10},
            {"through_step": 10_000, "max_x_bits": 11},
        ],
        "seed": args.seed,
        "parameters": sum(parameter.numel() for parameter in raw_model.parameters()),
        "channels": args.channels,
        "total_updates_available": BASE.OPERAND_BITS * BASE.MICRO_UPDATES,
        "dropout": args.dropout,
        "steps": args.steps,
        "examples": args.steps * args.batch_size,
        "muon_learning_rate": args.muon_learning_rate,
        "muon_weight_decay": args.muon_weight_decay,
        "muon_warmup_steps": args.muon_warmup_steps,
        "best_step": best_step,
        "elapsed_seconds": time.perf_counter() - started,
        "selected": selected,
        "curve": curve,
    }
    state = {key: value.detach().cpu() for key, value in raw_model.state_dict().items()}
    torch.save(state, args.out / "model_best.pt")
    (args.out / "eval_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / "source.py").write_text(Path(__file__).read_text())
    (args.out / "base_source.py").write_text(Path(__file__).with_name("train.py").read_text())
    print(json.dumps({"type": "final", **report}), flush=True)


if __name__ == "__main__":
    main()

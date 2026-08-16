"""Frozen linear probes for H13's hidden state after every input bit.

Research diagnostic only. Prefix residue/value targets train probe heads but
never update the processor and were not used by the H13 training run.
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


def load_source(path: Path):
    spec = importlib.util.spec_from_file_location("h13_source", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class StateExtractor(nn.Module):
    def __init__(self, model, operand_bits: int, lanes: int, micro_updates: int):
        super().__init__()
        self.model = model
        self.operand_bits = operand_bits
        self.lanes = lanes
        self.micro_updates = micro_updates

    def forward(self, x_bits: torch.Tensor, modulus_bits: torch.Tensor) -> torch.Tensor:
        model = self.model
        batch = x_bits.shape[0]
        modulus = model.bit_embedding(modulus_bits).transpose(1, 2)
        modulus = modulus + model.modulus_role[None, :, None]
        state = modulus.new_zeros(
            batch, model.channels, self.lanes, self.operand_bits
        )
        state[:, :, 2] = model.work_role[None, :, None]
        state[:, :, 3] = model.scratch_role[None, :, None]
        features = []
        for bit_index in range(self.operand_bits - 1, -1, -1):
            incoming = model.bit_embedding(x_bits[:, bit_index])
            incoming = incoming + model.input_role[None, :]
            incoming = incoming[:, :, None].expand(-1, -1, self.operand_bits)
            for phase_index in range(self.micro_updates):
                visible = state.clone()
                visible[:, :, 0] = (
                    incoming + model.phase[phase_index][None, :, None]
                )
                visible[:, :, 1] = modulus
                visible[:, :, :, 0] += model.boundaries[0][None, :, None]
                visible[:, :, :, -1] += model.boundaries[1][None, :, None]
                state = model.cell(visible, None)
            features.append(state[:, :, 2:4].flatten(1))
        return torch.stack(features, dim=1)


class StepHeads(nn.Module):
    def __init__(self, steps: int, feature_dim: int, bits: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(steps, feature_dim, bits) * 0.01)
        self.bias = nn.Parameter(torch.zeros(steps, bits))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bsf,sfo->bso", features, self.weight) + self.bias


@torch.inference_mode()
def extract(module, extractor, rows, batch_size: int) -> torch.Tensor:
    output = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        _, x, _ = batch.unbind(1)
        x_bits = module.bit_tensor(x, module.OPERAND_BITS)
        n_bits = module.bit_tensor(batch[:, 0], module.OPERAND_BITS)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output.append(extractor(x_bits, n_bits))
    return torch.cat(output)


def targets(module, rows):
    n, x, _ = rows.unbind(1)
    residues = []
    values = []
    for step in range(1, module.OPERAND_BITS + 1):
        prefix = x >> (module.OPERAND_BITS - step)
        residue = torch.remainder(prefix * prefix, n)
        residues.append(module.bit_tensor(residue, module.OPERAND_BITS))
        values.append(module.bit_tensor(prefix, module.OPERAND_BITS))
    return torch.stack(residues, dim=1).float(), torch.stack(values, dim=1).float()


@torch.inference_mode()
def evaluate(heads, features, target, batch_size: int):
    heads.eval()
    steps = target.shape[1]
    exact = torch.zeros(steps, dtype=torch.long)
    bit_correct = torch.zeros(steps, dtype=torch.long)
    bit_total = torch.zeros(steps, dtype=torch.long)
    for start in range(0, len(features), batch_size):
        stop = min(start + batch_size, len(features))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = heads(features[start:stop]).gt(0)
        matches = prediction == target[start:stop].bool()
        exact += matches.all(-1).sum(0).cpu()
        bit_correct += matches.sum((0, 2)).cpu()
        bit_total += torch.tensor(
            [matches.shape[0] * matches.shape[2]] * steps, dtype=torch.long
        )
    return [
        {
            "prefix_bits": step + 1,
            "exact": int(exact[step]) / len(features),
            "bit_accuracy": int(bit_correct[step]) / int(bit_total[step]),
        }
        for step in range(steps)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--channels", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.09)
    parser.add_argument("--train-rows", type=int, default=20_000)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=74)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    module = load_source(args.source_file)
    model = module.PrefixResidueState(args.channels, args.dropout).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    extractor = StateExtractor(
        model, module.OPERAND_BITS, module.LANES, module.MICRO_UPDATES
    ).to(device)

    train_x, heldout_x = module.split_values(args.seed)
    train_n, unseen_n = module.split_moduli(args.seed)
    row_sets = {
        "train": module.rows_tensor(
            module.make_rows(train_n, train_x, args.train_rows, args.seed + 1), device
        ),
        "validation_unseen_x_seen_n": module.rows_tensor(
            module.make_rows(train_n, heldout_x, 5_000, args.seed + 2), device
        ),
        "audit_unseen_x_unseen_n": module.rows_tensor(
            module.make_rows(unseen_n, heldout_x, 5_000, args.seed + 4), device
        ),
    }
    started = time.perf_counter()
    features = {
        name: extract(module, extractor, rows, args.batch_size)
        for name, rows in row_sets.items()
    }
    target_sets = {name: targets(module, rows) for name, rows in row_sets.items()}
    feature_seconds = time.perf_counter() - started
    del extractor, model
    torch.cuda.empty_cache()

    feature_dim = features["train"].shape[-1]
    heads = {
        "prefix_residue": StepHeads(module.OPERAND_BITS, feature_dim, module.OPERAND_BITS).to(device),
        "prefix_value": StepHeads(module.OPERAND_BITS, feature_dim, module.OPERAND_BITS).to(device),
    }
    parameters = [parameter for head in heads.values() for parameter in head.parameters()]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=1e-5
    )
    generator = torch.Generator(device=device).manual_seed(args.seed + 70_000)
    best_keys = {
        name: [(-1.0, -1.0)] * module.OPERAND_BITS for name in heads
    }
    best_steps = {name: [0] * module.OPERAND_BITS for name in heads}
    best_slices = {name: [None] * module.OPERAND_BITS for name in heads}
    curve = []
    train_started = time.perf_counter()

    for step in range(1, args.steps + 1):
        indices = torch.randint(
            len(features["train"]), (args.batch_size,), generator=generator,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            residue_logits = heads["prefix_residue"](features["train"][indices])
            value_logits = heads["prefix_value"](features["train"][indices])
            loss = F.binary_cross_entropy_with_logits(
                residue_logits, target_sets["train"][0][indices]
            ) + F.binary_cross_entropy_with_logits(
                value_logits, target_sets["train"][1][indices]
            )
        loss.backward()
        optimizer.step()

        if step == 1 or step % args.eval_every == 0:
            metrics = {
                "prefix_residue": evaluate(
                    heads["prefix_residue"], features["validation_unseen_x_seen_n"],
                    target_sets["validation_unseen_x_seen_n"][0], args.batch_size,
                ),
                "prefix_value": evaluate(
                    heads["prefix_value"], features["validation_unseen_x_seen_n"],
                    target_sets["validation_unseen_x_seen_n"][1], args.batch_size,
                ),
            }
            point = {
                "step": step, "loss": float(loss.detach()),
                "seconds": time.perf_counter() - train_started,
                "validation": metrics,
            }
            curve.append(point)
            print(json.dumps({"type": "progress", **point}), flush=True)
            for name, head in heads.items():
                for prefix_index, metric in enumerate(metrics[name]):
                    key = (metric["exact"], metric["bit_accuracy"])
                    if key > best_keys[name][prefix_index]:
                        best_keys[name][prefix_index] = key
                        best_steps[name][prefix_index] = step
                        best_slices[name][prefix_index] = (
                            head.weight[prefix_index].detach().clone(),
                            head.bias[prefix_index].detach().clone(),
                        )

    for name, head in heads.items():
        for prefix_index, state in enumerate(best_slices[name]):
            if state is None:
                raise RuntimeError("probe selection failed")
            head.weight.data[prefix_index].copy_(state[0])
            head.bias.data[prefix_index].copy_(state[1])

    selected = {}
    for split_name in row_sets:
        selected[split_name] = {
            "prefix_residue": evaluate(
                heads["prefix_residue"], features[split_name],
                target_sets[split_name][0], args.batch_size,
            ),
            "prefix_value": evaluate(
                heads["prefix_value"], features[split_name],
                target_sets[split_name][1], args.batch_size,
            ),
        }
    report = {
        "checkpoint": str(args.checkpoint),
        "source_file": str(args.source_file),
        "processor_frozen": True,
        "features": "work and scratch lanes after every input bit",
        "feature_dim": feature_dim,
        "train_rows": args.train_rows,
        "probe_steps": args.steps,
        "probe_learning_rate": args.learning_rate,
        "feature_seconds": feature_seconds,
        "probe_seconds": time.perf_counter() - train_started,
        "selection": "each head and prefix independently by validation exact then bit accuracy",
        "best_steps": best_steps,
        "selected": selected,
        "curve": curve,
    }
    torch.save({name: head.state_dict() for name, head in heads.items()}, args.out / "probe_best.pt")
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"type": "final", **report}), flush=True)


if __name__ == "__main__":
    main()

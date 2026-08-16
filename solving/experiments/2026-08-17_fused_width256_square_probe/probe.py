"""Frozen linear probes for the best fused binary work-state checkpoint.

Research only. The trained processor is frozen. Probe targets include the
literal square and therefore cannot be used in a competition submission.
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
    spec = importlib.util.spec_from_file_location("fused_source", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load source from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class WorkTapeExtractor(nn.Module):
    def __init__(self, model: nn.Module, lanes: int, workspace_bits: int) -> None:
        super().__init__()
        self.model = model
        self.lanes = lanes
        self.workspace_bits = workspace_bits

    def forward(
        self, source_bits: torch.Tensor, modulus_bits: torch.Tensor
    ) -> torch.Tensor:
        model = self.model
        batch = source_bits.shape[0]
        source = model.bit_embedding(source_bits).transpose(1, 2)
        modulus = model.bit_embedding(modulus_bits).transpose(1, 2)
        source = source + model.source_role[None, :, None]
        modulus = modulus + model.modulus_role[None, :, None]
        state = source.new_zeros(
            batch, model.channels, self.lanes, self.workspace_bits
        )
        state[:, :, 2] = source + model.work_role[None, :, None]
        for _ in range(model.updates):
            visible = state.clone()
            visible[:, :, 0] = source
            visible[:, :, 1] = modulus
            visible[:, :, :, 0] += model.boundaries[0][None, :, None]
            visible[:, :, :, -1] += model.boundaries[1][None, :, None]
            state = model.cell(visible, None)
        return state[:, :, 2]


class Probes(nn.Module):
    def __init__(self, channels: int, workspace_bits: int, operand_bits: int) -> None:
        super().__init__()
        features = channels * workspace_bits
        self.local_square = nn.Conv1d(channels, 1, 1)
        self.global_square = nn.Linear(features, workspace_bits)
        self.global_x = nn.Linear(features, operand_bits)

    def forward(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        flat = state.flatten(1)
        return {
            "local_square": self.local_square(state).squeeze(1),
            "global_square": self.global_square(flat),
            "global_x": self.global_x(flat),
        }


def strip_compile_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "_orig_mod."
    return {
        key[len(prefix) :] if key.startswith(prefix) else key: value
        for key, value in state.items()
    }


def make_targets(module, rows, device):
    x = torch.tensor([row[1] for row in rows], dtype=torch.long, device=device)
    square = x * x
    return (
        module.bit_tensor(square, module.WORKSPACE_BITS).float(),
        module.bit_tensor(x, module.OPERAND_BITS).float(),
    )


@torch.inference_mode()
def extract_features(module, extractor, rows, device, batch_size):
    output = torch.empty(
        len(rows),
        extractor.model.channels,
        module.WORKSPACE_BITS,
        dtype=torch.bfloat16,
        device=device,
    )
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        x = torch.tensor(
            [row[1] for row in batch_rows], dtype=torch.long, device=device
        )
        n = torch.tensor(
            [row[0] for row in batch_rows], dtype=torch.long, device=device
        )
        source = module.bit_tensor(x, module.WORKSPACE_BITS)
        modulus = module.bit_tensor(n, module.WORKSPACE_BITS)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output[start : start + len(batch_rows)] = extractor(source, modulus)
    return output


@torch.inference_mode()
def evaluate(probes, features, square, x, batch_size):
    counts = {
        name: {"exact": 0, "bits": 0, "total_bits": 0}
        for name in ("local_square", "global_square", "global_x")
    }
    probes.eval()
    for start in range(0, len(features), batch_size):
        stop = min(start + batch_size, len(features))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = probes(features[start:stop])
        targets = {
            "local_square": square[start:stop],
            "global_square": square[start:stop],
            "global_x": x[start:stop],
        }
        for name, prediction in logits.items():
            matches = prediction.gt(0) == targets[name].bool()
            counts[name]["exact"] += int(matches.all(-1).sum())
            counts[name]["bits"] += int(matches.sum())
            counts[name]["total_bits"] += matches.numel()
    return {
        name: {
            "exact": values["exact"] / len(features),
            "bit_accuracy": values["bits"] / values["total_bits"],
            "examples": len(features),
        }
        for name, values in counts.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--channels", type=int, default=256)
    parser.add_argument("--updates", type=int, default=44)
    parser.add_argument("--dropout", type=float, default=0.09)
    parser.add_argument("--seed", type=int, default=74)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--extract-batch-size", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.003)
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
    module = load_source(args.source_file)

    model = module.BinaryWorkState(args.channels, args.updates, args.dropout).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(strip_compile_prefix(checkpoint))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    extractor = WorkTapeExtractor(model, module.LANES, module.WORKSPACE_BITS).to(
        device
    )
    if args.compile:
        extractor = torch.compile(extractor)

    train_x, heldout_x = module.split_values(args.seed)
    train_n, unseen_n = module.split_moduli(args.seed)
    train_rows = module.make_rows(train_n, train_x, 100_000, args.seed + 1)
    validation_rows = module.make_rows(
        train_n, heldout_x, 5_000, args.seed + 2
    )

    started = time.perf_counter()
    train_features = extract_features(
        module, extractor, train_rows, device, args.extract_batch_size
    )
    validation_features = extract_features(
        module, extractor, validation_rows, device, args.extract_batch_size
    )
    train_square, train_source = make_targets(module, train_rows, device)
    validation_square, validation_source = make_targets(
        module, validation_rows, device
    )
    feature_seconds = time.perf_counter() - started
    del extractor, model
    torch.cuda.empty_cache()

    probes = Probes(args.channels, module.WORKSPACE_BITS, module.OPERAND_BITS).to(
        device
    )
    optimizer = torch.optim.AdamW(
        probes.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=1e-5
    )
    generator = torch.Generator(device=device).manual_seed(args.seed + 70_000)
    probe_names = ("local_square", "global_square", "global_x")
    best_keys = {name: (-1.0, -1.0) for name in probe_names}
    best_steps = {name: 0 for name in probe_names}
    best_states = {name: None for name in probe_names}
    curve = []
    train_started = time.perf_counter()

    for step in range(1, args.steps + 1):
        indices = torch.randint(
            len(train_features),
            (args.batch_size,),
            generator=generator,
            device=device,
        )
        probes.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = probes(train_features[indices])
            loss = (
                F.binary_cross_entropy_with_logits(
                    logits["local_square"], train_square[indices]
                )
                + F.binary_cross_entropy_with_logits(
                    logits["global_square"], train_square[indices]
                )
                + F.binary_cross_entropy_with_logits(
                    logits["global_x"], train_source[indices]
                )
            )
        loss.backward()
        optimizer.step()

        if step == 1 or step % args.eval_every == 0:
            metrics = evaluate(
                probes,
                validation_features,
                validation_square,
                validation_source,
                args.batch_size,
            )
            point = {
                "step": step,
                "loss": float(loss.detach()),
                "seconds": time.perf_counter() - train_started,
                "validation": metrics,
            }
            curve.append(point)
            print(json.dumps({"type": "progress", **point}), flush=True)
            for name in probe_names:
                key = (
                    metrics[name]["exact"],
                    metrics[name]["bit_accuracy"],
                )
                if key > best_keys[name]:
                    best_keys[name] = key
                    best_steps[name] = step
                    best_states[name] = copy.deepcopy(
                        getattr(probes, name).state_dict()
                    )

    if any(state is None for state in best_states.values()):
        raise RuntimeError("probe selection failed")
    for name in probe_names:
        getattr(probes, name).load_state_dict(best_states[name])
    selected = {
        "train": evaluate(
            probes,
            train_features,
            train_square,
            train_source,
            args.batch_size,
        ),
        "validation_unseen_x_seen_n": evaluate(
            probes,
            validation_features,
            validation_square,
            validation_source,
            args.batch_size,
        ),
    }

    audit_rows = {
        "audit_seen_x_unseen_n": module.make_rows(
            unseen_n, train_x, 5_000, args.seed + 3
        ),
        "audit_unseen_x_unseen_n": module.make_rows(
            unseen_n, heldout_x, 5_000, args.seed + 4
        ),
    }
    extractor_model = module.BinaryWorkState(
        args.channels, args.updates, args.dropout
    ).to(device)
    extractor_model.load_state_dict(strip_compile_prefix(checkpoint))
    extractor_model.eval()
    extractor = WorkTapeExtractor(
        extractor_model, module.LANES, module.WORKSPACE_BITS
    ).to(device)
    if args.compile:
        extractor = torch.compile(extractor)
    for name, rows in audit_rows.items():
        features = extract_features(
            module, extractor, rows, device, args.extract_batch_size
        )
        square, source = make_targets(module, rows, device)
        selected[name] = evaluate(
            probes, features, square, source, args.batch_size
        )

    report = {
        "checkpoint": str(args.checkpoint),
        "source_file": str(args.source_file),
        "seed": args.seed,
        "channels": args.channels,
        "updates": args.updates,
        "probe_steps": args.steps,
        "probe_learning_rate": args.learning_rate,
        "feature_seconds": feature_seconds,
        "train_seconds": time.perf_counter() - train_started,
        "best_steps": best_steps,
        "selection": "each head independently by its own validation exact, then bit accuracy",
        "selected": selected,
        "curve": curve,
    }
    torch.save(probes.state_dict(), args.out / "probe_best.pt")
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"type": "final", **report}), flush=True)


if __name__ == "__main__":
    main()

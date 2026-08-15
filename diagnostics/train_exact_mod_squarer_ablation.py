"""Diagnostic squarer ablations through an exact modulo oracle.

This file is deliberately outside submissions/: it computes privileged modular
arithmetic and square diagnostics, so none of its losses are competition-legal.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("binary_square_submission", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def records(path: Path) -> list[dict]:
    output = []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["time_steps"] == 1:
                output.append(row)
    return output


def residue_distribution(bit_logits, modulus, maximum_modulus, temperature=1.0):
    batch = bit_logits.shape[0]
    residues = torch.arange(maximum_modulus, device=bit_logits.device)
    valid = residues[None, :] < modulus[:, None]
    distribution = bit_logits.new_zeros(batch, maximum_modulus, dtype=torch.float32)
    distribution[:, 0] = 1.0
    probabilities = (bit_logits.float() / temperature).sigmoid()
    for bit_index in range(bit_logits.shape[1]):
        weight = torch.remainder(1 << bit_index, modulus)
        source = torch.remainder(residues[None, :] - weight[:, None], modulus[:, None])
        shifted = distribution.gather(1, source)
        probability = probabilities[:, bit_index : bit_index + 1]
        distribution = ((1.0 - probability) * distribution + probability * shifted) * valid
    return distribution, probabilities


def sample_paired(train, by_x, batch_size):
    chosen = []
    keys = list(by_x)
    while len(chosen) < batch_size:
        group = by_x[random.choice(keys)]
        if len(group) >= 2:
            chosen.extend(random.sample(group, 2))
        else:
            chosen.append(group[0])
    return chosen[:batch_size]


@torch.no_grad()
def evaluate(module, model, items):
    model.eval()
    square_correct = mod_correct = bit_correct = bit_total = 0
    for start in range(0, len(items), 512):
        chunk = items[start : start + 512]
        x = torch.tensor([item["x"] for item in chunk], device="cuda")
        n = torch.tensor([item["modulus"] for item in chunk], device="cuda")
        target = torch.tensor([item["result"] for item in chunk], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(module._bits(x).float())
        bits = logits >= 0
        shifts = torch.arange(module.SQUARE_BITS, device="cuda")
        truth = torch.bitwise_and(torch.bitwise_right_shift((x * x)[:, None], shifts), 1).bool()
        powers = torch.bitwise_left_shift(torch.ones(module.SQUARE_BITS, dtype=torch.long, device="cuda"), shifts)
        value = (bits.long() * powers).sum(1)
        square_correct += int((bits == truth).all(1).sum())
        mod_correct += int((torch.remainder(value, n) == target).sum())
        bit_correct += int((bits == truth).sum())
        bit_total += truth.numel()
    return {
        "examples": len(items),
        "square_exact": square_correct / len(items),
        "square_bit": bit_correct / bit_total,
        "mod_exact": mod_correct / len(items),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("baseline", "curriculum", "paired", "consistency", "digitize", "pretrained"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=74)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    module = load_module(args.submission)
    train = records(args.data_root / "train.jsonl")
    test = records(args.data_root / "test.jsonl")
    by_x = defaultdict(list)
    for row in train:
        by_x[row["x"]].append(row)
    paired = {x: rows for x, rows in by_x.items() if len({r["modulus"] for r in rows}) >= 2}
    informative = [row for row in train if row["x"] * row["x"] < row["modulus"]]
    maximum_modulus = max(row["modulus"] for row in train + test) + 1
    model = module.BinarySquarer().cuda()
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if any(key.startswith("squarer.") for key in state):
            state = {key.removeprefix("squarer."): value for key, value in state.items() if key.startswith("squarer.")}
        translated = {}
        for key, value in state.items():
            key = key.removeprefix("_orig_mod.")
            key = key.replace("embedding.", "bit_embedding.")
            key = key.replace("cells.0.", "cell.")
            key = key.replace("readout.", "head.")
            translated[key] = value
        state = translated
        model.load_state_dict(state)
    matrix = [parameter for parameter in model.parameters() if parameter.ndim >= 2]
    scalar = [parameter for parameter in model.parameters() if parameter.ndim < 2]
    muon = module.FlattenedMuon(matrix)
    adam = torch.optim.AdamW(scalar, lr=3e-4, betas=(0.9, 0.95), weight_decay=1e-5)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoints = {0, 100, 500, 1000, 3000, args.steps}
    metrics = (args.output / "metrics.jsonl").open("w")

    def report(step, loss):
        row = {"mode": args.mode, "step": step, "loss": loss, "train": evaluate(module, model, train), "test": evaluate(module, model, test)}
        print(json.dumps(row), flush=True)
        metrics.write(json.dumps(row) + "\n"); metrics.flush()
        torch.save({f"squarer.{key}": value.cpu() for key, value in model.state_dict().items()}, args.output / f"step_{step:06d}.pt")

    if args.mode == "pretrained":
        report(0, None)
    for step in range(1, 0 if args.mode == "pretrained" else args.steps + 1):
        if args.mode == "curriculum" and step <= 1000:
            pool = informative
            chosen = random.choices(pool, k=args.batch_size)
        elif args.mode == "paired":
            chosen = sample_paired(train, paired, args.batch_size)
        else:
            chosen = random.choices(train, k=args.batch_size)
        x = torch.tensor([r["x"] for r in chosen], device="cuda")
        n = torch.tensor([r["modulus"] for r in chosen], device="cuda")
        target = torch.tensor([r["result"] for r in chosen], device="cuda")
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(module._bits(x).float())
        temperature = max(0.35, 1.0 - 0.65 * step / args.steps) if args.mode == "digitize" else 1.0
        distribution, probabilities = residue_distribution(logits, n, maximum_modulus, temperature)
        loss = -distribution.gather(1, target[:, None]).squeeze(1).clamp_min(1e-12).log().mean()
        if args.mode == "digitize":
            loss = loss + 0.1 * (probabilities * (1.0 - probabilities)).mean()
        if args.mode == "consistency":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                second = model(module._bits(x).float())
            loss = loss + 0.2 * F.mse_loss(logits.sigmoid(), second.sigmoid())
        muon.zero_grad(set_to_none=True); adam.zero_grad(set_to_none=True)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        muon.step(); adam.step()
        progress = min(1.0, max(0.0, (step - 1000) / 4000))
        muon.param_groups[0]["lr"] = 0.002 + 0.018 * 0.5 * (1.0 + math.cos(math.pi * progress))
        if step in checkpoints:
            report(step, float(loss.detach()))
    metrics.close()


if __name__ == "__main__":
    main()

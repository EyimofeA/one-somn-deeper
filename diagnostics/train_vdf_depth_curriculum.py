"""Research-only final-label depth curriculum for a tied VDF transition."""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from benchmark import ModelSpec
from train_vdf_trace_ablation import batch, forward_trace, gathered


def load_source(path):
    spec = importlib.util.spec_from_file_location("vdf_card", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def digits(value): return [7 + int(digit) for digit in str(value)]


def apply(state, modulus, depth):
    for _ in range(depth): state = state * state % modulus
    return state


def prompt(modulus, state, depth): return [2, *digits(modulus), 3, *digits(state), 4, *digits(depth)]


def row(modulus, state, depth):
    width = len(str(modulus))
    return (prompt(modulus, state, depth), [7 + int(digit) for digit in f"{apply(state, modulus, depth):0{width}d}"], [], depth)


def dataset(modulus, depths, seed):
    values = list(range(modulus)); random.Random(seed).shuffle(values)
    boundary = int(.8 * len(values)); train_x, test_x = values[:boundary], values[boundary:]
    return [row(modulus, state, depth) for state in train_x for depth in depths], test_x


@torch.no_grad()
def evaluate(model, source, items, device):
    model.eval(); result = {}
    for start in range(0, len(items), 512):
        ids, mask, labels, positions, _, depths = batch(items[start:start + 512], device)
        logits, _ = forward_trace(model, source, ids, mask)
        correct = (gathered(logits, positions).argmax(-1) == labels).all(-1)
        for depth, value in zip(depths.tolist(), correct.tolist()): result.setdefault(depth, []).append(value)
    return {str(depth): sum(values) / len(values) for depth, values in sorted(result.items())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=180)
    args = parser.parse_args()
    torch.manual_seed(74); device = torch.device("cuda")
    source = load_source(args.submission)
    train, held_x = dataset(323, (1, 2, 4), 45)
    test = [row(323, state, depth) for state in held_x for depth in (1, 2, 4, 8, 16, 32, 64)]
    ood = [row(modulus, state, depth) for modulus in (437, 493, 527) for state in range(0, modulus, max(1, modulus // 48)) for depth in (1, 2, 4, 8, 16, 32, 64)]
    model = source.VDFModel(ModelSpec(17, max(len(item[0]) for item in train + test + ood), 500_000_000)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, betas=(.9, .95), weight_decay=.05)
    schedule = source.Schedule(optimizer, args.seconds)
    started, step, history = time.monotonic(), 0, []
    while time.monotonic() - started < args.seconds:
        fraction = (time.monotonic() - started) / args.seconds
        limit = 1 if fraction < 1 / 3 else 2 if fraction < 2 / 3 else 4
        eligible = [item for item in train if item[3] <= limit]
        offset = (step * 512) % len(eligible); items = (eligible + eligible)[offset:offset + 512]
        ids, mask, labels, positions, _, _ = batch(items, device)
        model.train(); logits, _ = forward_trace(model, source, ids, mask)
        loss = F.cross_entropy(gathered(logits, positions).reshape(-1, 17), labels.reshape(-1))
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); schedule.step(); step += 1
        if step == 1 or step % 250 == 0:
            history.append({"step": step, "seconds": round(time.monotonic() - started, 1), "phase_max_T": limit, "loss": float(loss), "train_exact": float((gathered(logits, positions).argmax(-1) == labels).all(-1).float().mean())})
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"classification": "RESEARCH ONLY — custom generated final-label curriculum; no intermediate labels", "train_N": 323, "phase_schedule": "T=1 -> T<=2 -> T<=4", "steps": step, "seconds": time.monotonic() - started, "train_curve": history, "held_out_x_final_exact_by_T": evaluate(model, source, test, device), "unseen_N_final_exact_by_T": evaluate(model, source, ood, device)}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__": main()

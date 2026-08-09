"""Public-E5 support ablation for the unchanged factored T=1 tape."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from train_t1_phase_square_reduce import PhaseSquareReduce, evaluate
from train_t1_representation import batch, encode


def load_rows(path: Path, *, require_t1: bool = True):
    output = []
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            if require_t1 and int(record["time_steps"]) != 1:
                continue
            modulus = int(record["modulus"])
            x = int(record["x"])
            target = int(record["result"])
            output.append(
                (
                    encode(modulus, "decimal"),
                    encode(x, "decimal"),
                    encode(target, "decimal"),
                    target,
                )
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train = load_rows(args.data_root / "train.jsonl")
    seen = load_rows(args.data_root / "depth_t_1.jsonl")
    unseen = load_rows(args.data_root / "depth_ood_n_t_1.jsonl")
    model = PhaseSquareReduce("factored").to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-3, betas=(0.9, 0.95), weight_decay=0.05
    )

    started = time.monotonic()
    step = 0
    curve = []
    while time.monotonic() - started < args.seconds:
        n, x, y, _ = batch(train, 512, step, device)
        model.train()
        logits = model(n, x)
        loss = F.cross_entropy(logits.reshape(-1, 10), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        step += 1
        if step == 1 or step % 100 == 0:
            curve.append(
                {
                    "step": step,
                    "seconds": round(time.monotonic() - started, 2),
                    "loss": float(loss.detach()),
                    "last_batch_exact": float(
                        (logits.argmax(-1) == y).all(-1).float().mean()
                    ),
                }
            )

    elapsed = time.monotonic() - started
    result = {
        "arm": "factored_e5_support",
        "seed": args.seed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "steps": step,
        "seconds": elapsed,
        "steps_per_second": step / max(elapsed, 1e-9),
        "train_rows": len(train),
        "curve": curve,
        "train": evaluate(model, train, device),
        "seen_N_t1": evaluate(model, seen, device),
        "ood_N_t1": evaluate(model, unseen, device),
    }
    report = {
        "classification": "RESEARCH ONLY — public-E5 support ablation",
        "one_variable": "data support only",
        "intermediate_supervision": False,
        "data_root": str(args.data_root),
        "result": result,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "eval_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

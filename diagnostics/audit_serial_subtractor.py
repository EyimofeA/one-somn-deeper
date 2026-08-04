"""Audit learned serial subtraction on unseen moduli without changing weights."""

from __future__ import annotations

import argparse
import json

import torch

from train_serial_subtractor import WIDTH, SerialSubtractor, digits, rows, semiprimes, tensors


def exact_and_digits(pred, target):
    return {
        "exact": float((pred == target).all(dim=-1).float().mean()),
        "per_lsd_position": [float((pred[:, i] == target[:, i]).float().mean()) for i in range(WIDTH)],
    }


def first_below(report, threshold):
    return next((int(q) for q, values in report.items() if values["rollout_fixed_depth_exact"] < threshold), None)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-moduli", type=int, default=48)
    ap.add_argument("--test-moduli", type=int, default=16)
    ap.add_argument("--per-modulus", type=int, default=128)
    ap.add_argument("--max-q", type=int, default=5)
    ap.add_argument("--modulus-split", choices=("seen", "unseen"), default="unseen")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    all_moduli = semiprimes(args.seed, args.train_moduli + args.test_moduli)
    moduli = all_moduli[:args.train_moduli] if args.modulus_split == "seen" else all_moduli[args.train_moduli:]
    base = rows(moduli, args.seed, args.per_modulus, heldout=False)
    model = SerialSubtractor().to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device, weights_only=True))
    model.eval()
    report = {}
    for q in range(args.max_q + 1):
        examples = []
        for _, n_lsd, target in base:
            n_value = int("".join(map(str, n_lsd[::-1])))
            r_value = int("".join(map(str, target[::-1])))
            if len(str(r_value + q * n_value)) > WIDTH:
                raise ValueError(f"unrepresentable q={q} state at width={WIDTH}")
            examples.append((digits(r_value + q * n_value)[::-1], n_lsd, target))
        state, n, target = tensors(examples, args.device)
        teacher_target = target if q == 0 else torch.tensor([digits(int("".join(map(str, row[0][::-1]))) - int("".join(map(str, row[1][::-1]))))[::-1] for row in examples], dtype=torch.long, device=args.device)
        teacher = model(state, n).argmax(dim=-1)
        for _ in range(q):
            state = model(state, n).argmax(dim=-1)
        teacher_metrics = exact_and_digits(teacher, teacher_target)
        rollout_metrics = exact_and_digits(state, target)
        report[str(q)] = {
            "teacher_one_step_exact": teacher_metrics["exact"],
            "teacher_per_lsd_position": teacher_metrics["per_lsd_position"],
            "rollout_fixed_depth_exact": rollout_metrics["exact"],
            "rollout_per_lsd_position": rollout_metrics["per_lsd_position"],
            "examples": len(examples), "width_errors": 0,
        }
    result = {
        "per_q": report,
        "first_q_below_100": first_below(report, 1.0),
        "first_q_below_95": first_below(report, 0.95),
        "first_zero_exact_q": next((int(q) for q, values in report.items() if values["rollout_fixed_depth_exact"] == 0.0), None),
        "representability": {"width": WIDTH, "examples": sum(x["examples"] for x in report.values()), "width_errors": 0},
        "true_remainder_fixed_point_exact": report["0"]["teacher_one_step_exact"],
        "modulus_split": args.modulus_split,
    }
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

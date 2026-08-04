"""Roll out the learned q=1 serial subtractor on unseen moduli."""

from __future__ import annotations

import argparse
import json

import torch

from train_serial_subtractor import SerialSubtractor, digits, rows, semiprimes, tensors


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
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    all_moduli = semiprimes(args.seed, args.train_moduli + args.test_moduli)
    test_moduli = all_moduli[args.train_moduli:]
    base = rows(test_moduli, args.seed, args.per_modulus, heldout=False)
    model = SerialSubtractor().to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.eval()
    report = {}
    for q in range(1, args.max_q + 1):
        examples = []
        for _, n_lsd, target in base:
            n_value = int("".join(map(str, n_lsd[::-1])))
            r_value = int("".join(map(str, target[::-1])))
            examples.append((digits(r_value + q * n_value)[::-1], n_lsd, target))
        state, n, target = tensors(examples, args.device)
        teacher_target = torch.tensor([digits(int("".join(map(str, row[0][::-1]))) - int("".join(map(str, row[1][::-1]))))[::-1] for row in examples], dtype=torch.long, device=args.device)
        teacher = model(state, n).argmax(dim=-1)
        for _ in range(q):
            state = model(state, n).argmax(dim=-1)
        report[str(q)] = {"teacher_one_step_exact": float((teacher == teacher_target).all(dim=-1).float().mean()), "rollout_exact": float((state == target).all(dim=-1).float().mean())}
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

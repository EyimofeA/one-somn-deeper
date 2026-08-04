"""Train a frozen-subtractor canonicality head without supplying quotient depth."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from train_serial_subtractor import WIDTH, SerialSubtractor, digits, semiprimes


def examples(moduli, seed, per_modulus, qs, heldout):
    """True qN+r trace states; width failures are explicit, never truncated."""
    result = []
    for index, modulus in enumerate(moduli):
        rng = random.Random(seed + 10_000 * index)
        train = set(rng.sample(list(range(modulus)), per_modulus))
        pool = [r for r in range(modulus) if r not in train] if heldout else list(train)
        for remainder in rng.sample(pool, per_modulus):
            for quotient in qs:
                value = quotient * modulus + remainder
                if len(str(value)) > WIDTH:
                    raise ValueError(f"unrepresentable {value} at width={WIDTH}")
                result.append((digits(value)[::-1], digits(modulus)[::-1], int(quotient == 0), digits(remainder)[::-1], quotient))
    random.Random(seed + 77 + int(heldout)).shuffle(result)
    return result


class StopHead(nn.Module):
    """Only a learned readout of the frozen serial GRU's final state."""

    def __init__(self, subtractor):
        super().__init__()
        self.subtractor = subtractor
        self.head = nn.Linear(subtractor.cell.hidden_size, 1)

    def forward(self, state, modulus):
        return self.head(self.subtractor.encode(state, modulus)[1]).squeeze(-1)


def tensors(batch, device):
    return tuple(torch.tensor([row[i] for row in batch], dtype=torch.long, device=device) for i in (0, 1, 3))


@torch.no_grad()
def evaluate(model, dataset, device):
    report = {}
    total_fp = total_noncanonical = total_fn = total_canonical = 0
    for quotient in range(11):
        batch = [row for row in dataset if row[4] == quotient]
        state, modulus, target = tensors(batch, device)
        powers = torch.tensor([10**i for i in range(WIDTH)], device=device)
        modulus_value = (modulus * powers).sum(dim=-1)
        active = torch.ones(len(batch), dtype=torch.bool, device=device)
        stopped = torch.full((len(batch),), -1, dtype=torch.long, device=device)
        fp = noncanonical = fn = canonical = 0
        for step in range(17):  # up to 16 learned subtractor applications
            stop = model(state, modulus).sigmoid() >= 0.5
            # This is evaluation-only ground truth for the *generated* state.
            # It does not enter the model or choose an inference action.
            canonical_here = active & ((state * powers).sum(dim=-1) < modulus_value)
            noncanonical_here = active & ~canonical_here
            fp += int((stop & noncanonical_here).sum())
            noncanonical += int(noncanonical_here.sum())
            fn += int((~stop & canonical_here).sum())
            canonical += int(canonical_here.sum())
            newly_stopped = active & stop
            stopped[newly_stopped] = step
            active &= ~stop
            if not active.any() or step == 16:
                break
            next_state = model.subtractor(state, modulus).argmax(dim=-1)
            state = torch.where(active[:, None], next_state, state)
        exact = (state == target).all(dim=-1)
        stop_correct = stopped == quotient
        report[str(quotient)] = {
            "remainder_exact": float(exact.float().mean()),
            "halting_correct": float(stop_correct.float().mean()),
            "exact_stop_step": float(stop_correct.float().mean()),
            "early_stops": float((stopped.ge(0) & (stopped < quotient)).float().mean()),
            "late_stops": float((stopped > quotient).float().mean()),
            "non_stops": float((stopped < 0).float().mean()),
            "avg_executed_steps": float(torch.where(stopped < 0, torch.full_like(stopped, 16), stopped).float().mean()),
            "per_lsd_position": [float((state[:, i] == target[:, i]).float().mean()) for i in range(WIDTH)],
            "examples": len(batch),
            "arithmetic_errors": int((~exact).sum()),
            "width_errors": 0,
            "false_positive_stop_rate": fp / noncanonical if noncanonical else 0.0,
            "false_negative_continue_rate": fn / canonical if canonical else 0.0,
        }
        total_fp += fp; total_noncanonical += noncanonical; total_fn += fn; total_canonical += canonical
    return {
        "per_q": report,
        "false_positive_stop_rate_noncanonical_states": total_fp / total_noncanonical,
        "false_negative_continue_rate_canonical_states": total_fn / total_canonical,
        "representation_errors": 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    moduli = semiprimes(args.seed, 64)
    train_moduli, test_moduli = moduli[:48], moduli[48:]
    raw_train = examples(train_moduli, args.seed, 128, range(6), heldout=False)
    # q=0 is one sixth of a full q=0..5 trace set. Duplicate it five times so
    # stop and continue labels are exactly balanced without changing any state.
    train = raw_train + [row for row in raw_train if row[2]] * 4
    random.Random(args.seed + 313).shuffle(train)
    # Held-out moduli are never used by either frozen subtractor training or stop-head training.
    unseen = examples(test_moduli, args.seed, 128, range(11), heldout=False)
    subtractor = SerialSubtractor().to(args.device)
    subtractor.load_state_dict(torch.load(args.checkpoint, map_location=args.device, weights_only=True))
    subtractor.eval()
    for parameter in subtractor.parameters():
        parameter.requires_grad = False
    model = StopHead(subtractor).to(args.device)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=3e-4, weight_decay=0.01)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({
        "classification": "FROZEN-SUBTRACTOR LEARNED CANONICALITY SCREEN",
        "checkpoint": args.checkpoint, "seed": args.seed, "width": WIDTH,
        "train_q": [0, 1, 2, 3, 4, 5], "eval_q": list(range(11)),
        "train_moduli": train_moduli, "test_moduli": test_moduli,
        "train_examples": len(train), "stop_continue_balance": "exactly 1:1 via q=0 trace oversampling",
        "steps": args.steps, "batch_size": args.batch_size,
        "inference": "while learned_stop(state,N) is false: frozen_subtractor(state,N), capped at 16; no q/depth input",
    }, indent=2) + "\n")
    start = time.perf_counter()
    with (out / "metrics.jsonl").open("w") as log:
        for step in range(1, args.steps + 1):
            offset = ((step - 1) * args.batch_size) % len(train)
            batch = [train[(offset + i) % len(train)] for i in range(args.batch_size)]
            state, modulus, _ = tensors(batch, args.device)
            label = torch.tensor([row[2] for row in batch], dtype=torch.float32, device=args.device)
            loss = F.binary_cross_entropy_with_logits(model(state, modulus), label)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            if step % 200 == 0 or step == args.steps:
                record = {"step": step, "loss": float(loss.detach()), "steps_per_sec": step / (time.perf_counter() - start)}
                log.write(json.dumps(record) + "\n"); log.flush(); print(json.dumps(record), flush=True)
    model.eval()
    report = evaluate(model, unseen, args.device)
    (out / "eval_report.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save(model.head.state_dict(), out / "stop_head.pt")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

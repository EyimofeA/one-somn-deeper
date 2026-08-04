"""Separate teacher-forced transition error from self-fed rollout error."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from train_comparator_reducer import ComparatorReducer, SerialComparator
from train_serial_subtractor import SerialSubtractor, canonical_rows, digits, semiprimes


def inputs(rows, quotient, device):
    state, modulus, remainder, target = [], [], [], []
    for r_lsd, n_lsd, _ in rows:
        r = int("".join(map(str, r_lsd[::-1]))); n = int("".join(map(str, n_lsd[::-1])))
        state.append(digits(quotient * n + r)[::-1]); modulus.append(n_lsd); remainder.append(r_lsd)
        target.append(digits((quotient - 1) * n + r)[::-1])
    return tuple(torch.tensor(value, dtype=torch.long, device=device) for value in (state, modulus, remainder, target))


@torch.no_grad()
def evaluate(model, rows, quotients, max_steps, device):
    result = {}
    for quotient in quotients:
        state, modulus, remainder, expected_next = inputs(rows, quotient, device)
        teacher_target = expected_next.clone()
        raw_subtractor = model.subtractor(state, modulus).argmax(dim=-1)
        composed_log_probs, gate = model(state, modulus); composed = composed_log_probs.argmax(dim=-1)
        comparator = (gate >= .5) == (quotient > 0)

        active = torch.ones(len(rows), dtype=torch.bool, device=device)
        first_error = torch.full((len(rows),), -1, dtype=torch.long, device=device)  # 0 branch, 1 subtractor
        stopped = torch.full((len(rows),), -1, dtype=torch.long, device=device)
        for step in range(max_steps + 1):
            log_probs, current_gate = model(state, modulus)
            candidate = log_probs.argmax(dim=-1); continue_ = current_gate >= .5
            true_continue = step < quotient
            on_true_trace = active & (first_error < 0)
            branch_error = on_true_trace & (continue_ != true_continue)
            subtractor_error = on_true_trace & continue_ & true_continue & (candidate != expected_next).any(dim=-1)
            first_error[branch_error] = 0; first_error[subtractor_error] = 1
            newly_stopped = active & ~continue_; stopped[newly_stopped] = step
            active &= continue_
            if step == max_steps or not active.any():
                break
            state = torch.where(active[:, None], candidate, state)
            if step < quotient - 1:
                # Next true trace state is one reduction closer to r.
                expected_next = torch.tensor([digits((quotient - step - 2) * int("".join(map(str, row[1][::-1]))) + int("".join(map(str, row[0][::-1]))))[::-1] for row in rows], dtype=torch.long, device=device)
            elif step == quotient - 1:
                expected_next = remainder
        final_exact = (state == remainder).all(dim=-1)
        branch_count, subtraction_count = int((first_error == 0).sum()), int((first_error == 1).sum())
        mode = "none" if branch_count + subtraction_count == 0 else ("initial/later subtractor transition" if subtraction_count >= branch_count else "comparator branch")
        result[str(quotient)] = {
            "comparator_accuracy": float(comparator.float().mean()),
            "subtractor_next_state_exact": float((raw_subtractor == teacher_target).all(dim=-1).float().mean()),
            "composed_teacher_transition_exact": float((composed == teacher_target).all(dim=-1).float().mean()),
            "rollout_final_exact": float(final_exact.float().mean()),
            "first_failure_mode": mode,
            "first_branch_errors": branch_count,
            "first_subtractor_errors": subtraction_count,
            "examples": len(rows),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quotients", default="1,5,10,20,30,50,100", help="Comma-separated quotient buckets to audit.")
    parser.add_argument("--max-steps", type=int, help="Rollout cap; defaults to the largest requested quotient plus ten.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    comparator = SerialComparator().to(args.device); comparator.load_state_dict(checkpoint["comparator"])
    subtractor = SerialSubtractor().to(args.device); subtractor.load_state_dict(checkpoint["subtractor"])
    model = ComparatorReducer(comparator, subtractor).to(args.device).eval()
    rows = canonical_rows(semiprimes(args.seed, 64)[48:], args.seed, 128)
    quotients = tuple(int(value) for value in args.quotients.split(","))
    report = evaluate(model, rows, quotients, args.max_steps or max(quotients) + 10, args.device)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

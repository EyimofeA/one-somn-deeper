"""Diagnostic-only stability-gated halting for frozen learned serial models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from train_serial_stop_head import StopHead, examples, tensors
from train_serial_subtractor import WIDTH, SerialSubtractor


@torch.no_grad()
def evaluate(model, dataset, device, max_q, max_steps):
    report = {}
    powers = torch.tensor([10**i for i in range(WIDTH)], device=device)
    for quotient in range(max_q + 1):
        batch = [row for row in dataset if row[4] == quotient]
        state, modulus, target = tensors(batch, device)
        modulus_value = (modulus * powers).sum(dim=-1)
        active = torch.ones(len(batch), dtype=torch.bool, device=device)
        stopped = torch.full((len(batch),), -1, dtype=torch.long, device=device)
        updates_after_canonical = torch.zeros(len(batch), dtype=torch.long, device=device)
        ever_wrong_canonical = torch.zeros(len(batch), dtype=torch.bool, device=device)
        wrong_canonical_events = repairs_direct = stays_wrong = becomes_other_wrong = 0
        true_remainder_nonabsorbing = 0

        # This diagnostic's terminal rule is the only changed inference variable:
        # learned-canonical AND learned fixed point. No label enters this branch.
        for step in range(max_steps + 1):
            candidate = model.subtractor(state, modulus).argmax(dim=-1)
            stable = (candidate == state).all(dim=-1)
            learned_canonical = model(state, modulus).sigmoid() >= 0.5
            should_stop = active & learned_canonical & stable

            # Ground truth below is report-only; it never controls `should_stop`.
            actual_canonical = (state * powers).sum(dim=-1) < modulus_value
            correct = (state == target).all(dim=-1)
            wrong_canonical = active & actual_canonical & ~correct
            wrong_canonical_events += int(wrong_canonical.sum())
            ever_wrong_canonical |= wrong_canonical
            repairs_direct += int((wrong_canonical & (candidate == target).all(dim=-1)).sum())
            stays_wrong += int((wrong_canonical & stable).sum())
            becomes_other_wrong += int((wrong_canonical & ~(candidate == target).all(dim=-1) & ~stable).sum())
            true_remainder_nonabsorbing += int((active & correct & ~stable).sum())

            stopped[should_stop] = step
            continuing = active & ~should_stop
            if step == max_steps:
                break
            updates_after_canonical += (continuing & actual_canonical).long()
            state = torch.where(continuing[:, None], candidate, state)
            active = continuing

        exact = (state == target).all(dim=-1)
        stop_correct = stopped == quotient
        direct_remainder_candidate = model.subtractor(target, modulus).argmax(dim=-1)
        direct_remainder_stable = (direct_remainder_candidate == target).all(dim=-1)
        report[str(quotient)] = {
            "final_remainder_exact": float(exact.float().mean()),
            "exact_halt_step_accuracy": float(stop_correct.float().mean()),
            "early_stops": float((stopped.ge(0) & (stopped < quotient)).float().mean()),
            "late_stops": float((stopped > quotient).float().mean()),
            "non_stops": float((stopped < 0).float().mean()),
            "avg_executed_steps": float(torch.where(stopped < 0, torch.full_like(stopped, max_steps), stopped).float().mean()),
            "avg_additional_verification_steps": float(updates_after_canonical.float().mean()),
            "wrong_canonical_states_encountered": wrong_canonical_events,
            "wrong_canonical_examples_encountered": int(ever_wrong_canonical.sum()),
            "wrong_canonical_states_successfully_repaired": int((ever_wrong_canonical & exact).sum()),
            "wrong_canonical_direct_repairs": repairs_direct,
            "wrong_canonical_stays_unchanged": stays_wrong,
            "wrong_canonical_becomes_another_wrong_state": becomes_other_wrong,
            "true_remainder_states_not_absorbing": true_remainder_nonabsorbing,
            "true_remainder_stability": float(direct_remainder_stable.float().mean()),
            "true_remainder_nonabsorbing_direct": int((~direct_remainder_stable).sum()),
            "examples": len(batch),
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subtractor-checkpoint", required=True)
    parser.add_argument("--stop-head-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-q", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from train_serial_subtractor import semiprimes
    moduli = semiprimes(args.seed, 64)[48:]
    dataset = examples(moduli, args.seed, 128, range(args.max_q + 1), heldout=False)
    subtractor = SerialSubtractor().to(args.device)
    subtractor.load_state_dict(torch.load(args.subtractor_checkpoint, map_location=args.device, weights_only=True))
    subtractor.eval()
    for parameter in subtractor.parameters():
        parameter.requires_grad = False
    model = StopHead(subtractor).to(args.device)
    model.head.load_state_dict(torch.load(args.stop_head_checkpoint, map_location=args.device, weights_only=True))
    model.eval()
    report = {
        "classification": "DIAGNOSTIC ONLY — stability equality is not a submission mechanism",
        "max_steps": args.max_steps,
        "per_q": evaluate(model, dataset, args.device, args.max_q, args.max_steps),
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

"""Slice T=1 checkpoints by quotient depth without retraining or selection."""
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import torch


BASE_PATH = (
    Path(__file__).resolve().parents[1]
    / "2026-08-16_binary_workstate_matched"
    / "train.py"
)
_spec = importlib.util.spec_from_file_location("binary_matched_base", BASE_PATH)
assert _spec is not None and _spec.loader is not None
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)


def quotient_bucket(quotient: int) -> str:
    if quotient == 0:
        return "q=0"
    if quotient == 1:
        return "q=1"
    if quotient <= 3:
        return "q=2..3"
    if quotient <= 7:
        return "q=4..7"
    if quotient <= 15:
        return "q=8..15"
    if quotient <= 31:
        return "q=16..31"
    if quotient <= 63:
        return "q=32..63"
    return "q>=64"


def summarize(records: list[dict]) -> dict:
    exact = sum(record["exact"] for record in records)
    return {
        "examples": len(records),
        "exact": exact / len(records) if records else None,
        "predicted_residue_is_valid": (
            sum(record["prediction"] < record["n"] for record in records)
            / len(records)
            if records
            else None
        ),
        "identity_shortcut": (
            sum(record["prediction"] == record["x"] for record in records)
            / len(records)
            if records
            else None
        ),
        "zero_prediction": (
            sum(record["prediction"] == 0 for record in records) / len(records)
            if records
            else None
        ),
    }


@torch.no_grad()
def evaluate_rows(model, rows, device, mode: str) -> dict:
    model.eval()
    records = []
    shifts = torch.arange(base.OPERAND_BITS, device=device)
    for start in range(0, len(rows), 512):
        batch_rows = rows[start : start + 512]
        source, modulus, _ = base.tensor_batch(batch_rows, mode, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            bits = model(source, modulus).gt(0).long()
        predictions = (bits * (1 << shifts)).sum(-1).cpu().tolist()
        for (n, x, target), prediction in zip(batch_rows, predictions):
            records.append(
                {
                    "n": n,
                    "x": x,
                    "target": target,
                    "prediction": prediction,
                    "quotient": (x * x) // n,
                    "exact": prediction == target,
                }
            )

    by_quotient = defaultdict(list)
    by_centered_quotient = defaultdict(list)
    by_modulus_bits = defaultdict(list)
    for record in records:
        by_quotient[quotient_bucket(record["quotient"])].append(record)
        centered = min(record["x"], record["n"] - record["x"])
        by_centered_quotient[
            quotient_bucket((centered * centered) // record["n"])
        ].append(record)
        by_modulus_bits[str(record["n"].bit_length())].append(record)
    return {
        "overall": summarize(records),
        "by_quotient": {
            name: summarize(group) for name, group in by_quotient.items()
        },
        "by_centered_quotient": {
            name: summarize(group) for name, group in by_centered_quotient.items()
        },
        "by_modulus_bits": {
            name: summarize(group) for name, group in by_modulus_bits.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--updates", type=int, default=33)
    parser.add_argument("--dropout", type=float, default=0.09)
    parser.add_argument("--seed", type=int, default=74)
    parser.add_argument("--mode", choices=("fused_x", "exact_square"), default="fused_x")
    parser.add_argument(
        "--architecture",
        choices=("local", "learned_fast", "scratch_fast"),
        default="local",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    if args.architecture in ("learned_fast", "scratch_fast"):
        folder = (
            "2026-08-17_binary_learned_fast_message_exact_square_t1"
            if args.architecture == "learned_fast"
            else "2026-08-17_binary_scratch_fast_message_exact_square_t1"
        )
        model_path = (
            Path(__file__).resolve().parents[1]
            / folder
            / "train.py"
        )
        model_spec = importlib.util.spec_from_file_location(
            "learned_fast_message_model", model_path
        )
        assert model_spec is not None and model_spec.loader is not None
        model_module = importlib.util.module_from_spec(model_spec)
        model_spec.loader.exec_module(model_module)
        model_class = (
            model_module.LearnedMessageBinaryWorkState
            if args.architecture == "learned_fast"
            else model_module.ScratchMessageBinaryWorkState
        )
        model = model_class(args.channels, args.updates, args.dropout).to(device)
    else:
        model = base.BinaryWorkState(args.channels, args.updates, args.dropout).to(device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = {key.removeprefix("_orig_mod."): value for key, value in state.items()}
    model.load_state_dict(state)

    train_x, heldout_x = base.split_values(args.seed)
    train_n, unseen_n = base.split_moduli(args.seed)
    sets = {
        "validation_unseen_x_seen_n": base.make_rows(
            train_n, heldout_x, 5000, args.seed + 2
        ),
        "audit_seen_x_unseen_n": base.make_rows(
            unseen_n, train_x, 5000, args.seed + 3
        ),
        "audit_unseen_x_unseen_n": base.make_rows(
            unseen_n, heldout_x, 5000, args.seed + 4
        ),
    }
    report = {
        name: evaluate_rows(model, rows, device, args.mode)
        for name, rows in sets.items()
    }
    if hasattr(model, "fast_message_scales"):
        report["fast_message_scales"] = model.fast_message_scales.detach().cpu().tolist()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

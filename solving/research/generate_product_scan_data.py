"""Train random three-digit products; hold out complete operand pairs."""

from __future__ import annotations

import json
import random
from pathlib import Path


DIGIT_OFFSET = 7
SEED = 45


def digit_tokens(value: int, width: int) -> list[int]:
    return [DIGIT_OFFSET + int(digit) for digit in f"{value:0{width}d}"[::-1]]


def record(a: int, b: int, split: str, index: int) -> dict[str, object]:
    return {
        "input_ids": [2, *digit_tokens(a, 3), 3, *digit_tokens(b, 3), 4, 8],
        "labels": digit_tokens(a * b, 6),
        "instance_id": f"product_scan_{split}_{index:06d}",
        "split": split,
        "generator_family": "learned_pairwise_product_carry_scan",
        "label_exact": True,
        "label_method": "local_generator_decimal_product",
        "result": a * b,
        "configured_modulus_bits": None,
        "modulus": a,
        "modulus_bits": a.bit_length(),
        "time_steps": 1,
        "x": b,
        "seed": SEED,
    }


def main() -> None:
    output = Path("data/generated/product_scan_full_position_ood")
    rng = random.Random(SEED)
    test_set: set[tuple[int, int]] = set()
    while len(test_set) < 2_000:
        test_set.add((rng.randrange(100, 1_000), rng.randrange(100, 1_000)))
    train_set: set[tuple[int, int]] = set()
    while len(train_set) < 190_000:
        pair = (rng.randrange(1_000), rng.randrange(1_000))
        if pair not in test_set:
            train_set.add(pair)
    train_pairs = list(train_set)
    test_pairs = list(test_set)
    rng.shuffle(train_pairs)
    rng.shuffle(test_pairs)
    output.mkdir(parents=True, exist_ok=True)
    for split, pairs in (("train", train_pairs), ("test", test_pairs)):
        with (output / f"{split}.jsonl").open("w") as handle:
            for index, (a, b) in enumerate(pairs):
                handle.write(json.dumps(record(a, b, split, index)) + "\n")
    config = {
        "data_format": "separate_input_output",
        "dataset_kind": "squaring_mod",
        "max_seq_len": 10,
        "vocab_size": 17,
        "split_counts": {"train": len(train_pairs), "test": len(test_pairs)},
        "train_operands": "190,000 random pairs from 0..999, excluding test pairs",
        "test_operands": "a,b in 100..999 (both three-digit)",
        "label_format": "six LSD-first decimal digits, leading zeros retained",
    }
    (output / "dataset_config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()

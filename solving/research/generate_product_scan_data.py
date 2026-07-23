"""Train one-short-operand products; test products with both operands long."""

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
    output = Path("data/generated/product_scan_one_short_ood")
    rng = random.Random(SEED)
    # Every train product has at least one two-digit operand, but the other can
    # be three digits. Thus columns 0..3 receive nonzero learned pair features;
    # only the high-by-high interaction remains held out at test time.
    train_pairs = [
        (a, b)
        for a in range(1_000)
        for b in range(1_000)
        if min(a, b) < 100
    ]
    test_pairs = [(rng.randrange(100, 1_000), rng.randrange(100, 1_000)) for _ in range(2_000)]
    rng.shuffle(train_pairs)
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
        "train_operands": "a,b in 0..999 with min(a,b) < 100",
        "test_operands": "a,b in 100..999 (both three-digit)",
        "label_format": "six LSD-first decimal digits, leading zeros retained",
    }
    (output / "dataset_config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()

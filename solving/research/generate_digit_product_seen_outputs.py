"""Generate a held-pair digit-product split with no unseen product values."""

from __future__ import annotations

import json
import random
from pathlib import Path


DIGIT_OFFSET = 7
SEED = 45
REPEATS = 10
TEST_PAIRS = {
    (0, 0), (0, 2), (0, 3), (1, 4), (1, 6), (2, 0), (2, 6), (2, 8),
    (2, 9), (3, 0), (3, 3), (4, 1), (4, 6), (4, 9), (6, 1), (6, 2),
    (6, 4), (8, 2), (9, 2), (9, 4),
}


def digits(value: int) -> list[int]:
    return [DIGIT_OFFSET + int(digit) for digit in str(value)]


def record(a: int, b: int, split: str, index: int, repeat: int) -> dict[str, object]:
    return {
        "a": a,
        "b": b,
        "input_ids": [2, DIGIT_OFFSET + a, 3, DIGIT_OFFSET + b, 4, 8],
        "labels": digits(a * b),
        "instance_id": f"digit_product_seen_outputs_{split}_{index:03d}_{repeat:02d}",
        "split": split,
        "generator_family": "gate1_digit_product_seen_outputs",
        "label_exact": True,
        "label_method": "exact_single_digit_product",
        "result": a * b,
        "configured_modulus_bits": None,
        "modulus": a,
        "modulus_bits": a.bit_length(),
        "time_steps": 1,
        "x": b,
        "seed": SEED,
    }


def main() -> None:
    output = Path("data/generated/gate1_digit_product_seen_outputs")
    all_pairs = {(a, b) for a in range(10) for b in range(10)}
    train_pairs = all_pairs - TEST_PAIRS
    assert len(TEST_PAIRS) == 20 and len(train_pairs) == 80
    assert {a * b for a, b in TEST_PAIRS} <= {a * b for a, b in train_pairs}
    assert all((b, a) in TEST_PAIRS for a, b in TEST_PAIRS)
    rng = random.Random(SEED)
    output.mkdir(parents=True, exist_ok=True)
    for split, pairs in (("train", train_pairs), ("test", TEST_PAIRS)):
        ordered = sorted(pairs)
        rng.shuffle(ordered)
        rows = [
            record(a, b, split, index, repeat)
            for repeat in range(REPEATS)
            for index, (a, b) in enumerate(ordered)
        ]
        with (output / f"{split}.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    config = {
        "data_format": "separate_input_output",
        "dataset_kind": "squaring_mod",
        "max_seq_len": 6,
        "vocab_size": 17,
        "split_counts": {"train": 800, "test": 200},
        "split_group": "held_digit_pair_with_seen_product_values",
        "test_pairs": sorted(TEST_PAIRS),
    }
    (output / "dataset_config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()

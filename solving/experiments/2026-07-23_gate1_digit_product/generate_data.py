#!/usr/bin/env python3
"""Generate the Gate 1 held-out digit-product diagnostic."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


TOKEN_IDS = {
    "PAD": 0,
    "BOS": 1,
    "N": 2,
    "X": 3,
    "T": 4,
    "ANS": 5,
    "EOS": 6,
    "DIGIT_OFFSET": 7,
}
T = 1
SEED = 45
REPEATS_PER_PAIR = 10
TRAIN_UNIQUE_PAIRS = 80
TEST_UNIQUE_PAIRS = 20
TRAIN_SIZE = TRAIN_UNIQUE_PAIRS * REPEATS_PER_PAIR
TEST_SIZE = TEST_UNIQUE_PAIRS * REPEATS_PER_PAIR
DEFAULT_OUTPUT_DIR = Path("data/generated/gate1_digit_product")


def number_tokens(value: int) -> list[int]:
    if value < 0:
        raise ValueError("only non-negative integers can be tokenized")
    return [TOKEN_IDS["DIGIT_OFFSET"] + int(char) for char in str(value)]


def decode_digits(tokens: list[int]) -> int:
    if not tokens:
        raise ValueError("cannot decode an empty digit sequence")
    digits = [token - TOKEN_IDS["DIGIT_OFFSET"] for token in tokens]
    if any(not 0 <= digit <= 9 for digit in digits):
        raise ValueError("sequence contains a non-digit token")
    return int("".join(str(digit) for digit in digits))


def tokenize_prompt(a: int, b: int) -> list[int]:
    return [
        TOKEN_IDS["N"],
        *number_tokens(a),
        TOKEN_IDS["X"],
        *number_tokens(b),
        TOKEN_IDS["T"],
        *number_tokens(T),
    ]


def make_record(
    a: int,
    b: int,
    split: str,
    index: int,
    repetition: int,
) -> dict[str, Any]:
    result = a * b
    return {
        "a": a,
        "b": b,
        "configured_modulus_bits": None,
        "generator_family": "gate1_digit_product",
        "input_ids": tokenize_prompt(a, b),
        "instance_id": (
            f"gate1_digit_product_s{SEED}_{split}_{index:08d}_r{repetition:02d}"
        ),
        "label_exact": True,
        "label_method": "exact_single_digit_product",
        "labels": number_tokens(result),
        "modulus": a,
        "modulus_bits": a.bit_length(),
        "result": result,
        "seed": SEED,
        "split": split,
        "time_steps": T,
        "x": b,
    }


def build_records() -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    pairs = [(a, b) for a in range(10) for b in range(10)]
    # The split is symmetric: if (a, b) is held out, (b, a) is also held out.
    # Every digit remains visible in both operand roles during training.
    test_pairs = [(a, b) for a, b in pairs if (a + b) % 5 == 0]
    train_pairs = [(a, b) for a, b in pairs if (a + b) % 5 != 0]
    rng.shuffle(train_pairs)
    rng.shuffle(test_pairs)

    records: list[dict[str, Any]] = []
    for split, split_pairs in (
        ("train", train_pairs),
        ("test", test_pairs),
    ):
        for repetition in range(REPEATS_PER_PAIR):
            records.extend(
                make_record(a, b, split, index, repetition)
                for index, (a, b) in enumerate(split_pairs)
            )
    return records


def self_test(records: list[dict[str, Any]]) -> None:
    by_split = {
        split: [record for record in records if record["split"] == split]
        for split in ("train", "test")
    }
    assert len(by_split["train"]) == TRAIN_SIZE
    assert len(by_split["test"]) == TEST_SIZE

    train_pairs = {(record["a"], record["b"]) for record in by_split["train"]}
    test_pairs = {(record["a"], record["b"]) for record in by_split["test"]}
    all_pairs = {(a, b) for a in range(10) for b in range(10)}
    assert len(train_pairs) == TRAIN_UNIQUE_PAIRS
    assert len(test_pairs) == TEST_UNIQUE_PAIRS
    assert train_pairs.isdisjoint(test_pairs)
    assert train_pairs | test_pairs == all_pairs
    assert all((a + b) % 5 != 0 for a, b in train_pairs)
    assert all((a + b) % 5 == 0 for a, b in test_pairs)
    assert all((b, a) in train_pairs for a, b in train_pairs)
    assert all((b, a) in test_pairs for a, b in test_pairs)
    assert {a for a, _ in train_pairs} == set(range(10))
    assert {b for _, b in train_pairs} == set(range(10))

    for split, split_pairs in (("train", train_pairs), ("test", test_pairs)):
        counts = {
            pair: sum(
                (record["a"], record["b"]) == pair
                for record in by_split[split]
            )
            for pair in split_pairs
        }
        assert set(counts.values()) == {REPEATS_PER_PAIR}

    for record in records:
        input_ids = record["input_ids"]
        labels = record["labels"]
        assert set(record) == {
            "a",
            "b",
            "configured_modulus_bits",
            "generator_family",
            "input_ids",
            "instance_id",
            "label_exact",
            "label_method",
            "labels",
            "modulus",
            "modulus_bits",
            "result",
            "seed",
            "split",
            "time_steps",
            "x",
        }
        assert isinstance(input_ids, list) and all(
            isinstance(token, int) and 0 <= token < 17 for token in input_ids
        )
        assert isinstance(labels, list) and all(
            isinstance(token, int) and 7 <= token < 17 for token in labels
        )
        assert input_ids == tokenize_prompt(record["a"], record["b"])
        assert len(input_ids) == 6
        assert TOKEN_IDS["BOS"] not in input_ids
        assert TOKEN_IDS["ANS"] not in input_ids
        assert TOKEN_IDS["EOS"] not in input_ids
        assert input_ids[-2:] == [TOKEN_IDS["T"], TOKEN_IDS["DIGIT_OFFSET"] + T]
        assert decode_digits(labels) == record["a"] * record["b"]
        assert record["result"] == record["a"] * record["b"]
        assert 0 <= record["result"] <= 81
        assert len(labels) <= 2
        assert len(labels) <= len(input_ids)

    repeated = build_records()
    assert [
        (
            record["split"],
            record["a"],
            record["b"],
            record["instance_id"],
        )
        for record in repeated
    ] == [
        (
            record["split"],
            record["a"],
            record["b"],
            record["instance_id"],
        )
        for record in records
    ]


def dataset_config(records: list[dict[str, Any]]) -> dict[str, Any]:
    split_counts = {
        split: sum(record["split"] == split for record in records)
        for split in ("train", "test")
    }
    return {
        "data_format": "separate_input_output",
        "dataset_kind": "squaring_mod",
        "generator_config": {
            "diagnostic_target": "single_digit_product",
            "fixed_time_steps": T,
            "operand_range": [0, 9],
            "pair_repetitions": REPEATS_PER_PAIR,
            "seed": SEED,
            "separate_input_output": True,
            "split_rule": "test iff (a + b) mod 5 == 0; reverse pairs co-held-out",
            "test_examples": TEST_SIZE,
            "test_unique_pairs": TEST_UNIQUE_PAIRS,
            "train_examples": TRAIN_SIZE,
            "train_unique_pairs": TRAIN_UNIQUE_PAIRS,
        },
        "label_format": "tail_aligned_decimal_single_digit_product",
        "label_method": "exact_single_digit_product",
        "max_modulus_bits": 4,
        "max_seq_len": max(len(record["input_ids"]) for record in records),
        "max_time_steps": T,
        "num_examples": len(records),
        "split_counts": split_counts,
        "split_group": "ordered_digit_pair",
        "token_ids": TOKEN_IDS,
        "vocab_size": 17,
    }


def write_dataset(output_dir: Path, records: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        path = output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                if record["split"] == split:
                    handle.write(json.dumps(record, sort_keys=True))
                    handle.write("\n")
    with (output_dir / "dataset_config.json").open("w", encoding="utf-8") as handle:
        json.dump(dataset_config(records), handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--self-test-only",
        action="store_true",
        help="verify deterministic generation without writing files",
    )
    args = parser.parse_args()

    records = build_records()
    self_test(records)
    if not args.self_test_only:
        write_dataset(args.output_dir, records)
        print(f"wrote {len(records)} records to {args.output_dir}")
    print("self-test: PASS")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate the Gate 1 aligned independent-products diagnostic."""

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
TRAIN_X_COUNTS = {1: 8, 2: 72, 3: 220}
TEST_X_COUNTS = {1: 2, 2: 18, 3: 40}
OOD_X_COUNT = 100
MULTIPLIERS = tuple(range(10))
TRAIN_SIZE = sum(TRAIN_X_COUNTS.values()) * len(MULTIPLIERS)
TEST_SIZE = sum(TEST_X_COUNTS.values()) * len(MULTIPLIERS)
OOD_SIZE = OOD_X_COUNT * len(MULTIPLIERS)
DEFAULT_OUTPUT_DIR = Path("data/generated/gate1_aligned_products")


def number_tokens(value: int) -> list[int]:
    if value < 0:
        raise ValueError("only non-negative integers can be tokenized")
    return [TOKEN_IDS["DIGIT_OFFSET"] + int(char) for char in str(value)]


def digit_string(tokens: list[int]) -> str:
    if not tokens:
        raise ValueError("cannot decode an empty digit sequence")
    digits = [token - TOKEN_IDS["DIGIT_OFFSET"] for token in tokens]
    if any(not 0 <= digit <= 9 for digit in digits):
        raise ValueError("sequence contains a non-digit token")
    return "".join(str(digit) for digit in digits)


def tokenize_prompt(x: int, b: int) -> list[int]:
    return [
        TOKEN_IDS["N"],
        *number_tokens(x),
        TOKEN_IDS["X"],
        *number_tokens(b),
        TOKEN_IDS["T"],
        *number_tokens(T),
    ]


def aligned_product_string(x: int, b: int) -> str:
    return "".join(f"{int(digit) * b:02d}" for digit in str(x))


def aligned_product_tokens(x: int, b: int) -> list[int]:
    return [
        TOKEN_IDS["DIGIT_OFFSET"] + int(digit)
        for digit in aligned_product_string(x, b)
    ]


def make_record(x: int, b: int, split: str, index: int) -> dict[str, Any]:
    result = aligned_product_string(x, b)
    return {
        "b": b,
        "configured_modulus_bits": None,
        "generator_family": "gate1_aligned_products_no_carry",
        "input_ids": tokenize_prompt(x, b),
        "instance_id": (
            f"gate1_aligned_products_s{SEED}_{split}_{index:08d}"
        ),
        "label_exact": True,
        "label_method": "fixed_width_aligned_digit_products_no_carry",
        "labels": aligned_product_tokens(x, b),
        "modulus": x,
        "modulus_bits": x.bit_length(),
        "result": result,
        "seed": SEED,
        "split": split,
        "time_steps": T,
        "x": x,
    }


def select_x_splits() -> dict[str, list[int]]:
    rng = random.Random(SEED)
    train_x: list[int] = []
    test_x: list[int] = []
    for length in (1, 2, 3):
        start = 0 if length == 1 else 10 ** (length - 1)
        stop = 10**length
        candidates = list(range(start, stop))
        rng.shuffle(candidates)
        train_count = TRAIN_X_COUNTS[length]
        test_count = TEST_X_COUNTS[length]
        train_x.extend(candidates[:train_count])
        test_x.extend(candidates[train_count : train_count + test_count])

    ood_candidates = list(range(1000, 10000))
    rng.shuffle(ood_candidates)
    return {
        "train": train_x,
        "test": test_x,
        "ood": ood_candidates[:OOD_X_COUNT],
    }


def build_records() -> list[dict[str, Any]]:
    x_splits = select_x_splits()
    records: list[dict[str, Any]] = []
    for split in ("train", "test", "ood"):
        index = 0
        for x in x_splits[split]:
            for b in MULTIPLIERS:
                records.append(make_record(x, b, split, index))
                index += 1
    return records


def self_test(records: list[dict[str, Any]]) -> None:
    by_split = {
        split: [record for record in records if record["split"] == split]
        for split in ("train", "test", "ood")
    }
    assert len(by_split["train"]) == TRAIN_SIZE
    assert len(by_split["test"]) == TEST_SIZE
    assert len(by_split["ood"]) == OOD_SIZE

    x_sets = {
        split: {record["x"] for record in split_records}
        for split, split_records in by_split.items()
    }
    assert len(x_sets["train"]) == sum(TRAIN_X_COUNTS.values())
    assert len(x_sets["test"]) == sum(TEST_X_COUNTS.values())
    assert len(x_sets["ood"]) == OOD_X_COUNT
    assert x_sets["train"].isdisjoint(x_sets["test"])
    assert x_sets["train"].isdisjoint(x_sets["ood"])
    assert x_sets["test"].isdisjoint(x_sets["ood"])

    for length in (1, 2, 3):
        assert sum(len(str(x)) == length for x in x_sets["train"]) == (
            TRAIN_X_COUNTS[length]
        )
        assert sum(len(str(x)) == length for x in x_sets["test"]) == (
            TEST_X_COUNTS[length]
        )
    assert all(len(str(x)) == 4 for x in x_sets["ood"])

    for split, xs in x_sets.items():
        for x in xs:
            assert {
                record["b"]
                for record in by_split[split]
                if record["x"] == x
            } == set(MULTIPLIERS)

    covered_pairs = {
        (int(digit), record["b"])
        for record in by_split["train"]
        for digit in str(record["x"])
    }
    assert covered_pairs == {
        (digit, b) for digit in range(10) for b in MULTIPLIERS
    }

    for record in records:
        input_ids = record["input_ids"]
        labels = record["labels"]
        assert set(record) == {
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
        assert input_ids == tokenize_prompt(record["x"], record["b"])
        assert TOKEN_IDS["BOS"] not in input_ids
        assert TOKEN_IDS["ANS"] not in input_ids
        assert TOKEN_IDS["EOS"] not in input_ids
        assert input_ids[-2:] == [
            TOKEN_IDS["T"],
            TOKEN_IDS["DIGIT_OFFSET"] + T,
        ]
        expected = aligned_product_string(record["x"], record["b"])
        assert digit_string(labels) == expected
        assert record["result"] == expected
        assert len(labels) == 2 * len(str(record["x"]))
        assert len(labels) <= len(input_ids)
        assert all(0 <= token < 17 for token in input_ids)
        assert all(7 <= token < 17 for token in labels)

    assert aligned_product_string(372, 4) == "122808"
    repeated = build_records()
    assert [
        (
            record["split"],
            record["x"],
            record["b"],
            record["labels"],
        )
        for record in repeated
    ] == [
        (
            record["split"],
            record["x"],
            record["b"],
            record["labels"],
        )
        for record in records
    ]


def dataset_config(records: list[dict[str, Any]]) -> dict[str, Any]:
    split_counts = {
        split: sum(record["split"] == split for record in records)
        for split in ("train", "test", "ood")
    }
    return {
        "data_format": "separate_input_output",
        "dataset_kind": "squaring_mod",
        "generator_config": {
            "diagnostic_target": "aligned_digit_products_no_carry",
            "fixed_time_steps": T,
            "multipliers": list(MULTIPLIERS),
            "ood_length": 4,
            "ood_x_count": OOD_X_COUNT,
            "seed": SEED,
            "separate_input_output": True,
            "split_group": "x_sequence",
            "test_x_counts_by_length": TEST_X_COUNTS,
            "train_lengths": [1, 2, 3],
            "train_x_counts_by_length": TRAIN_X_COUNTS,
        },
        "label_format": "two_digits_per_input_digit_same_order",
        "label_method": "fixed_width_aligned_digit_products_no_carry",
        "max_modulus_bits": 14,
        "max_seq_len": max(len(record["input_ids"]) for record in records),
        "max_time_steps": T,
        "num_examples": len(records),
        "split_counts": split_counts,
        "split_group": "x_sequence",
        "token_ids": TOKEN_IDS,
        "vocab_size": 17,
    }


def write_dataset(output_dir: Path, records: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test", "ood"):
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

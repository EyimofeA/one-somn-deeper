#!/usr/bin/env python3
"""Generate the Gate 1 exact-square diagnostic in squaring_mod JSONL format."""

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
N = 10403
T = 1
SEED = 45
TRAIN_SIZE = 800
TEST_SIZE = 199
OOD_SIZE = 1000
DEFAULT_OUTPUT_DIR = Path("data/generated/gate1_square_n10403_t1")


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


def tokenize_prompt(x: int) -> list[int]:
    return [
        TOKEN_IDS["N"],
        *number_tokens(N),
        TOKEN_IDS["X"],
        *number_tokens(x),
        TOKEN_IDS["T"],
        *number_tokens(T),
    ]


def make_record(x: int, split: str, index: int) -> dict[str, Any]:
    result = x * x
    return {
        "configured_modulus_bits": None,
        "generator_family": "gate1_square_x_no_mod",
        "input_ids": tokenize_prompt(x),
        "instance_id": f"gate1_square_n{N}_t{T}_s{SEED}_{split}_{index:08d}",
        "label_exact": True,
        "label_method": "exact_square_x_no_mod",
        "labels": number_tokens(result),
        "modulus": N,
        "modulus_bits": N.bit_length(),
        "result": result,
        "seed": SEED,
        "split": split,
        "time_steps": T,
        "x": x,
    }


def build_records() -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    id_x = list(range(1, 1000))
    rng.shuffle(id_x)
    train_x = id_x[:TRAIN_SIZE]
    test_x = id_x[TRAIN_SIZE:]
    ood_x = rng.sample(range(1000, 10000), OOD_SIZE)

    records: list[dict[str, Any]] = []
    for split, values in (
        ("train", train_x),
        ("test", test_x),
        ("ood", ood_x),
    ):
        records.extend(
            make_record(x, split, index) for index, x in enumerate(values)
        )
    return records


def self_test(records: list[dict[str, Any]]) -> None:
    by_split = {
        split: [record for record in records if record["split"] == split]
        for split in ("train", "test", "ood")
    }
    assert len(by_split["train"]) == TRAIN_SIZE
    assert len(by_split["test"]) == TEST_SIZE
    assert len(by_split["ood"]) == OOD_SIZE

    train_x = {record["x"] for record in by_split["train"]}
    test_x = {record["x"] for record in by_split["test"]}
    ood_x = {record["x"] for record in by_split["ood"]}
    assert train_x.isdisjoint(test_x)
    assert train_x | test_x == set(range(1, 1000))
    assert len(ood_x) == OOD_SIZE
    assert all(1000 <= x <= 9999 and len(str(x)) == 4 for x in ood_x)

    for record in records:
        input_ids = record["input_ids"]
        labels = record["labels"]
        assert set(record) == {
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
        assert input_ids == tokenize_prompt(record["x"])
        assert TOKEN_IDS["BOS"] not in input_ids
        assert TOKEN_IDS["ANS"] not in input_ids
        assert TOKEN_IDS["EOS"] not in input_ids
        assert input_ids[-2:] == [TOKEN_IDS["T"], TOKEN_IDS["DIGIT_OFFSET"] + T]
        assert decode_digits(labels) == record["x"] * record["x"]
        assert record["result"] == record["x"] * record["x"]
        assert len(labels) <= len(input_ids)
        if record["split"] in ("train", "test"):
            assert len(labels) <= 6
        else:
            assert len(labels) <= 8

    repeated = build_records()
    assert [
        (record["split"], record["x"]) for record in repeated
    ] == [
        (record["split"], record["x"]) for record in records
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
            "diagnostic_target": "square_x_no_mod",
            "fixed_modulus": N,
            "fixed_time_steps": T,
            "id_x_range": [1, 999],
            "ood_x_range": [1000, 9999],
            "ood_examples": OOD_SIZE,
            "seed": SEED,
            "separate_input_output": True,
            "test_examples": TEST_SIZE,
            "train_examples": TRAIN_SIZE,
        },
        "label_format": "tail_aligned_decimal_square_x_no_mod",
        "label_method": "exact_square_x_no_mod",
        "max_modulus_bits": N.bit_length(),
        "max_seq_len": max(len(record["input_ids"]) for record in records),
        "max_time_steps": T,
        "num_examples": len(records),
        "split_counts": split_counts,
        "split_group": "x",
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

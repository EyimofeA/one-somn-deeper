#!/usr/bin/env python3
"""Generate the bounded decimal carry-normalization diagnostic."""

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
SEED = 45
PAIR_PRODUCT_MAX = 81
COUNTS_BY_C = {
    1: {"train": 64, "test": 16},
    2: {"train": 736, "test": 184},
    3: {"train": 1440, "test": 360},
    4: {"train": 1440, "test": 360},
    5: {"train": 1440, "test": 360},
    6: {"train": 1440, "test": 360},
    7: {"train": 1440, "test": 360},
}
TRAIN_SIZE = sum(counts["train"] for counts in COUNTS_BY_C.values())
TEST_SIZE = sum(counts["test"] for counts in COUNTS_BY_C.values())
DEFAULT_OUTPUT_DIR = Path("data/generated/gate1_carry_normalize")


def digit_tokens(text: str) -> list[int]:
    if not text or any(character not in "0123456789" for character in text):
        raise ValueError("expected a non-empty decimal digit string")
    return [TOKEN_IDS["DIGIT_OFFSET"] + int(character) for character in text]


def digit_string(tokens: list[int]) -> str:
    digits = [token - TOKEN_IDS["DIGIT_OFFSET"] for token in tokens]
    if not digits or any(not 0 <= digit <= 9 for digit in digits):
        raise ValueError("sequence contains a non-digit token")
    return "".join(str(digit) for digit in digits)


def contributor_counts(column_count: int) -> list[int]:
    left_digits = (column_count + 1) // 2
    right_digits = column_count + 1 - left_digits
    return [
        min(
            column + 1,
            left_digits,
            right_digits,
            column_count - column,
        )
        for column in range(column_count)
    ]


def column_bounds(column_count: int) -> list[int]:
    return [
        contributors * PAIR_PRODUCT_MAX
        for contributors in contributor_counts(column_count)
    ]


def normalize_columns(totals: tuple[int, ...]) -> list[int]:
    output_digits: list[int] = []
    carry = 0
    for total in totals:
        normalized = total + carry
        output_digits.append(normalized % 10)
        carry = normalized // 10
    if carry >= 100:
        raise ValueError("two carry digits are insufficient")
    output_digits.extend([carry % 10, carry // 10])
    return output_digits


def tokenize_totals(totals: tuple[int, ...]) -> list[int]:
    blocks = "".join(f"{total:03d}" for total in totals)
    return [TOKEN_IDS["N"], *digit_tokens(blocks)]


def sample_total_sequences(
    column_count: int,
    count: int,
    rng: random.Random,
) -> list[tuple[int, ...]]:
    bounds = column_bounds(column_count)
    sequences: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    while len(sequences) < count:
        totals = tuple(rng.randint(0, bound) for bound in bounds)
        if totals not in seen:
            seen.add(totals)
            sequences.append(totals)
    return sequences


def make_record(
    totals: tuple[int, ...],
    split: str,
    index: int,
) -> dict[str, Any]:
    output_digits = normalize_columns(totals)
    column_count = len(totals)
    return {
        "column_count": column_count,
        "column_totals": list(totals),
        "configured_modulus_bits": None,
        "contributor_counts": contributor_counts(column_count),
        "generator_family": "gate1_bounded_carry_normalization",
        "input_ids": tokenize_totals(totals),
        "instance_id": (
            f"gate1_carry_normalize_s{SEED}_{split}_{index:08d}"
        ),
        "label_exact": True,
        "label_method": "base10_lsd_first_carry_normalization",
        "labels": digit_tokens("".join(str(digit) for digit in output_digits)),
        "modulus": 0,
        "modulus_bits": 0,
        "result": "".join(str(digit) for digit in output_digits),
        "seed": SEED,
        "split": split,
        "time_steps": 0,
        "x": 0,
    }


def build_records() -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    by_split: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "test": [],
    }
    split_indices = {"train": 0, "test": 0}
    for column_count, counts in COUNTS_BY_C.items():
        total_count = counts["train"] + counts["test"]
        sequences = sample_total_sequences(column_count, total_count, rng)
        offset = 0
        for split in ("train", "test"):
            for totals in sequences[offset : offset + counts[split]]:
                by_split[split].append(
                    make_record(totals, split, split_indices[split])
                )
                split_indices[split] += 1
            offset += counts[split]
    return [*by_split["train"], *by_split["test"]]


def self_test(records: list[dict[str, Any]]) -> None:
    by_split = {
        split: [record for record in records if record["split"] == split]
        for split in ("train", "test")
    }
    assert len(by_split["train"]) == TRAIN_SIZE == 8000
    assert len(by_split["test"]) == TEST_SIZE == 2000

    for column_count, counts in COUNTS_BY_C.items():
        for split in ("train", "test"):
            assert sum(
                record["column_count"] == column_count
                for record in by_split[split]
            ) == counts[split]

    prompts = [tuple(record["input_ids"]) for record in records]
    assert len(prompts) == len(set(prompts))
    train_prompts = {tuple(record["input_ids"]) for record in by_split["train"]}
    test_prompts = {tuple(record["input_ids"]) for record in by_split["test"]}
    assert train_prompts.isdisjoint(test_prompts)

    for record in records:
        column_count = record["column_count"]
        totals = tuple(record["column_totals"])
        bounds = column_bounds(column_count)
        assert record["contributor_counts"] == contributor_counts(column_count)
        assert len(totals) == column_count
        assert all(0 <= total <= bound for total, bound in zip(totals, bounds))
        assert all(bound <= 324 for bound in bounds)

        input_ids = record["input_ids"]
        assert input_ids[0] == TOKEN_IDS["N"]
        encoded_blocks = digit_string(input_ids[1:])
        assert len(encoded_blocks) == 3 * column_count
        decoded_totals = tuple(
            int(encoded_blocks[index : index + 3])
            for index in range(0, len(encoded_blocks), 3)
        )
        assert decoded_totals == totals

        output_digits = normalize_columns(totals)
        assert [int(digit) for digit in digit_string(record["labels"])] == (
            output_digits
        )
        assert record["result"] == "".join(
            str(digit) for digit in output_digits
        )
        assert len(record["labels"]) == column_count + 2
        assert len(record["labels"]) <= len(input_ids)
        assert sum(
            total * (10**column)
            for column, total in enumerate(totals)
        ) == sum(
            digit * (10**column)
            for column, digit in enumerate(output_digits)
        )

    repeated = build_records()
    assert [
        (
            record["split"],
            record["column_totals"],
            record["labels"],
        )
        for record in repeated
    ] == [
        (
            record["split"],
            record["column_totals"],
            record["labels"],
        )
        for record in records
    ]


def dataset_config(records: list[dict[str, Any]]) -> dict[str, Any]:
    split_counts = {
        split: sum(record["split"] == split for record in records)
        for split in ("train", "test")
    }
    bounds_by_c = {
        str(column_count): column_bounds(column_count)
        for column_count in COUNTS_BY_C
    }
    return {
        "data_format": "separate_input_output",
        "dataset_kind": "squaring_mod",
        "generator_config": {
            "column_bounds_by_c": bounds_by_c,
            "column_counts": list(COUNTS_BY_C),
            "counts_by_c": COUNTS_BY_C,
            "diagnostic_target": "bounded_carry_normalization",
            "input_block_width": 3,
            "input_column_order": "lsd_first",
            "output_digit_order": "lsd_first",
            "pair_product_max": PAIR_PRODUCT_MAX,
            "seed": SEED,
            "separate_input_output": True,
            "terminal_carry_digits": 2,
        },
        "label_format": "fixed_c_plus_2_lsd_first_digits",
        "label_method": "base10_lsd_first_carry_normalization",
        "max_modulus_bits": 0,
        "max_seq_len": max(len(record["input_ids"]) for record in records),
        "max_time_steps": 0,
        "num_examples": len(records),
        "split_counts": split_counts,
        "split_group": "precarry_total_sequence",
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

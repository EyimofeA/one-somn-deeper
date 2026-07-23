"""Local held-T diagnostic for a learned soft-digit squaring recurrence."""

from __future__ import annotations

import json
import random
from pathlib import Path


DIGIT_OFFSET = 7


def digits(value: int, width: int = 4) -> list[int]:
    return [DIGIT_OFFSET + int(digit) for digit in f"{value:0{width}d}"[::-1]]


def record(base: int, steps: int, index: int, split: str) -> dict[str, object]:
    square_digits = digits((base ** (2**steps)) % 10**7, width=7)
    labels = square_digits[:4]
    if split == "train":
        labels = [*square_digits[3:7], *square_digits[:4]]
    return {
        "input_ids": [2, *digits(base), 4, DIGIT_OFFSET + steps, 6, 6],
        "labels": labels,
        "instance_id": f"soft_recurrence_{split}_{index:05d}",
        "split": split,
        "generator_family": "learned_soft_digit_squaring_recurrence",
        "label_exact": True,
        "label_method": "local_generator_integer_power",
        "result": base ** (2**steps),
        "configured_modulus_bits": None,
        "modulus": base,
        "modulus_bits": base.bit_length(),
        "time_steps": steps,
        "x": base,
    }


def main() -> None:
    output = Path("data/generated/one_step_four_digit_square")
    output.mkdir(parents=True, exist_ok=True)
    values = list(range(10_000))
    random.Random(45).shuffle(values)
    train = [(value, 1) for value in values[:8_000]]
    test = [(value, 1) for value in values[8_000:]]
    for split, examples in (("train", train), ("test", test)):
        with (output / f"{split}.jsonl").open("w") as handle:
            index = 0
            for base, steps in examples:
                handle.write(json.dumps(record(base, steps, index, split)) + "\n")
                index += 1
    config = {
        "data_format": "separate_input_output",
        "dataset_kind": "squaring_mod",
        "max_seq_len": 9,
        "vocab_size": 17,
        "train": "8,000 shuffled four-digit x values at T=1",
        "test": "2,000 disjoint four-digit x values at T=1",
        "label_format": "train: digits 3..6 then digits 0..3; test: digits 0..3",
    }
    (output / "dataset_config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()

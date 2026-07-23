"""Local held-T diagnostic for a learned soft-digit squaring recurrence."""

from __future__ import annotations

import json
from pathlib import Path


DIGIT_OFFSET = 7
REPEATS = 100


def digits(value: int, width: int = 4) -> list[int]:
    return [DIGIT_OFFSET + int(digit) for digit in f"{value:0{width}d}"[::-1]]


def record(base: int, steps: int, index: int, split: str) -> dict[str, object]:
    return {
        "input_ids": [2, *digits(base), 4, DIGIT_OFFSET + steps],
        "labels": digits((base ** (2**steps)) % 10**4),
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
    output = Path("data/generated/soft_digit_recurrence_held_t")
    output.mkdir(parents=True, exist_ok=True)
    train = [(base, steps) for base in range(10) for steps in (1, 2)]
    test = [(base, 3) for base in range(10)]
    for split, examples in (("train", train), ("test", test)):
        with (output / f"{split}.jsonl").open("w") as handle:
            index = 0
            for base, steps in examples:
                for _ in range(REPEATS):
                    handle.write(json.dumps(record(base, steps, index, split)) + "\n")
                    index += 1
    config = {
        "data_format": "separate_input_output",
        "dataset_kind": "squaring_mod",
        "max_seq_len": 7,
        "vocab_size": 17,
        "train": "bases 0..9 at T=1,2",
        "test": "bases 0..9 at held-out T=3",
        "label_format": "four LSD-first decimal digits",
    }
    (output / "dataset_config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()

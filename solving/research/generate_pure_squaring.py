"""Generate a pure-squaring diagnostic in the REAL competition token format.

Isolates squaring alone: labels are x^2 DIRECTLY (no modulus reduction at
all), so this is NOT the competition's actual label semantics (always x^2 mod
N) — it removes exactly one variable (modular reduction) while keeping the
real tokenization scheme identical (TOKEN_IDS, DIGIT_OFFSET, MSB-first digit
order — verified against real competition data and against
data/squaring_mod.py directly). N is present in the prompt for token-format
realism only; it plays no role in the label.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

DIGIT_OFFSET = 7
N_VALUE = 323          # present for format realism only, not used in the label
NUM_N_DIGITS = 3
NUM_X_DIGITS = 4
NUM_ANSWER_DIGITS = 8  # max x^2 for a 4-digit x is 99,980,001 (8 digits)


def digit_tokens(value: int, width: int) -> list[int]:
    return [DIGIT_OFFSET + int(d) for d in f"{value:0{width}d}"]


def record(x: int, index: int, split: str) -> dict[str, object]:
    y = x * x  # plain square, no mod
    input_ids = (
        [2, *digit_tokens(N_VALUE, NUM_N_DIGITS),
         3, *digit_tokens(x, NUM_X_DIGITS),
         4, DIGIT_OFFSET + 1]  # T=1, unused (T=1 is the only case tested here)
    )
    return {
        "input_ids": input_ids,
        "labels": digit_tokens(y, NUM_ANSWER_DIGITS),
        "instance_id": f"pure_square_{split}_{index:05d}",
        "split": split,
        "generator_family": "pure_squaring_no_mod",
        "label_exact": True,
        "label_method": "local_generator_plain_square",
        "result": y,
        "modulus": N_VALUE,
        "modulus_bits": N_VALUE.bit_length(),
        "time_steps": 1,
        "x": x,
    }


def main() -> None:
    output = Path("data/generated/pure_squaring_no_mod")
    output.mkdir(parents=True, exist_ok=True)
    values = list(range(10_000))
    random.Random(45).shuffle(values)
    n_train = 8_000
    train = values[:n_train]
    test = values[n_train:]
    for split, xs in (("train", train), ("test", test)):
        with (output / f"{split}.jsonl").open("w") as handle:
            for index, x in enumerate(xs):
                handle.write(json.dumps(record(x, index, split)) + "\n")
    config = {
        "data_format": "separate_input_output",
        "dataset_kind": "squaring_mod",
        "max_seq_len": 11,
        "vocab_size": 17,
        "train": "8,000 shuffled 4-digit x values, held-out x split",
        "test": "2,000 disjoint 4-digit x values",
        "label_format": "plain x^2, no modulus (8 LSD-first... "
        "actually MSB-first, matching real token order)",
    }
    (output / "dataset_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"wrote {len(train)} train / {len(test)} test to {output}")


if __name__ == "__main__":
    main()

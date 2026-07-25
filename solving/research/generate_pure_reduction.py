"""Generate a pure-reduction diagnostic: arbitrary P mod N, no squaring at all.

Isolates the reduction cell exactly the way pure_squaring_cell.py isolates
the multiply cell — P is a random 8-digit integer (NOT x^2 of anything),
N is fixed (=323, this project's recurring test modulus), label = P mod N,
directly and fully supervised (unlike the composed reduction cell, where
only the FINAL remainder was supervised through a chained squaring step).
This answers the question the composed test couldn't: can the reduction
mechanism learn division at all, decoupled from the joint-optimization
difficulty of chaining it after an unsupervised multiply stage.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

DIGIT_OFFSET = 7
N_VALUE = 323
NUM_N_DIGITS = 3
NUM_P_DIGITS = 8   # matches pure_squaring_cell's answer width, same P range


def digit_tokens(value: int, width: int) -> list[int]:
    return [DIGIT_OFFSET + int(d) for d in f"{value:0{width}d}"]


def record(p: int, index: int, split: str) -> dict[str, object]:
    y = p % N_VALUE
    input_ids = [
        2, *digit_tokens(N_VALUE, NUM_N_DIGITS),
        3, *digit_tokens(p, NUM_P_DIGITS),
        4, DIGIT_OFFSET + 1,  # T=1, unused placeholder for format parity
    ]
    return {
        "input_ids": input_ids,
        "labels": digit_tokens(y, NUM_N_DIGITS),
        "instance_id": f"pure_reduction_{split}_{index:05d}",
        "split": split,
        "generator_family": "pure_reduction_no_square",
        "label_exact": True,
        "label_method": "local_generator_plain_mod",
        "result": y,
        "modulus": N_VALUE,
        "modulus_bits": N_VALUE.bit_length(),
        "time_steps": 1,
        "x": p,
    }


def main() -> None:
    output = Path("data/generated/pure_reduction_no_square")
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(45)
    # sample P uniformly over the full 8-digit range (0 .. 99,999,999),
    # matching the value range pure_squaring_cell's products actually occupy
    values = rng.sample(range(100_000_000), 10_000)
    n_train = 8_000
    train = values[:n_train]
    test = values[n_train:]
    for split, ps in (("train", train), ("test", test)):
        with (output / f"{split}.jsonl").open("w") as handle:
            for index, p in enumerate(ps):
                handle.write(json.dumps(record(p, index, split)) + "\n")
    config = {
        "data_format": "separate_input_output",
        "dataset_kind": "squaring_mod",
        "max_seq_len": 15,
        "vocab_size": 17,
        "train": "8,000 random 8-digit P values, held-out P split",
        "test": "2,000 disjoint 8-digit P values",
        "label_format": "P mod 323, MSB-first, no squaring involved",
    }
    (output / "dataset_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"wrote {len(train)} train / {len(test)} to {output}")


if __name__ == "__main__":
    main()

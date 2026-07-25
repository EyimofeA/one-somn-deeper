"""Fixed-N=323 data for the learned reduction cell (solving/DESIGN_NEXT.md OPT 1).

Rung 1 of the plan: N held constant, held-out x only. If this doesn't train,
the reduction mechanism itself doesn't work — no point testing cross-N
transfer. Offline generator only: computing `% N` here to build ground-truth
labels is normal dataset construction, not a forward-pass oracle (the model
itself never sees or computes `%` — see learned_reduction_cell.py).
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

DIGIT_OFFSET = 7
N = 323
NUM_N_DIGITS = 3
NUM_X_DIGITS = 4


def digit_tokens(value: int, width: int) -> list[int]:
    return [DIGIT_OFFSET + int(d) for d in f"{value:0{width}d}"]


def record(x: int, split: str, index: int) -> dict[str, object]:
    y = (x * x) % N
    input_ids = (
        [2, *digit_tokens(N, NUM_N_DIGITS)]
        + [3, *digit_tokens(x, NUM_X_DIGITS)]
        + [4, DIGIT_OFFSET + 1]  # T=1 marker, single squaring step
    )
    return {
        "input_ids": input_ids,
        "labels": digit_tokens(y, NUM_N_DIGITS),
        "instance_id": f"reduction_cell_fixedn_{split}_{index:05d}",
        "split": split,
        "generator_family": "learned_reduction_cell_fixed_n",
        "label_exact": True,
        "label_method": "local_generator_integer_mod",
        "result": y,
        "modulus": N,
        "modulus_bits": N.bit_length(),
        "time_steps": 1,
        "x": x,
    }


def main() -> None:
    output = Path("data/generated/learned_reduction_cell_fixed_n323")
    output.mkdir(parents=True, exist_ok=True)

    units = [x for x in range(1, N) if math.gcd(x, N) == 1]
    random.Random(45).shuffle(units)
    n_train = int(0.8 * len(units))
    train_x, test_x = units[:n_train], units[n_train:]

    for split, xs in (("train", train_x), ("test", test_x)):
        with (output / f"{split}.jsonl").open("w") as handle:
            for index, x in enumerate(xs):
                handle.write(json.dumps(record(x, split, index)) + "\n")

    config = {
        "data_format": "separate_input_output",
        "dataset_kind": "squaring_mod",
        "max_seq_len": 11,
        "vocab_size": 17,
        "train": f"{len(train_x)} held-out-x values, N=323 fixed, T=1",
        "test": f"{len(test_x)} disjoint held-out-x values, N=323 fixed, T=1",
        "label_format": f"{NUM_N_DIGITS} MSB-first decimal digits of x^2 mod 323",
    }
    (output / "dataset_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"wrote {len(train_x)} train / {len(test_x)} test to {output}")


if __name__ == "__main__":
    main()

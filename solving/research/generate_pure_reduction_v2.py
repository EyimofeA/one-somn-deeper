"""Pure-reduction diagnostic v2: reciprocal (log-uniform) P sampling.

Same task as generate_pure_reduction.py (arbitrary P mod 323, no squaring),
one change to the DATA: P is sampled from the reciprocal/log-uniform
distribution derived in "Learning Modular Exponentiation with Transformers"
(arXiv 2506.23679, appendix A.1) instead of uniformly. Derivation: binning the
continuous reciprocal density f(x) ~ 1/x onto integers 0..M gives
P(n) ~ ln(n+2) - ln(n+1), i.e. CDF(n) = ln(n+2)/ln(M+2). Inverse-CDF sampling:
draw u ~ Uniform(0,1), n = floor((M+2)^u) - 2. Small values get dramatically
more weight than under uniform sampling over the same 8-digit range.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

DIGIT_OFFSET = 7
N_VALUE = 323
NUM_N_DIGITS = 3
NUM_P_DIGITS = 8
M = 10**NUM_P_DIGITS - 1  # 99,999,999


def sample_reciprocal(rng: random.Random) -> int:
    u = rng.random()
    n = math.floor((M + 2) ** u) - 2
    return max(0, min(M, n))


def digit_tokens(value: int, width: int) -> list[int]:
    return [DIGIT_OFFSET + int(d) for d in f"{value:0{width}d}"]


def record(p: int, index: int, split: str) -> dict[str, object]:
    y = p % N_VALUE
    input_ids = [
        2, *digit_tokens(N_VALUE, NUM_N_DIGITS),
        3, *digit_tokens(p, NUM_P_DIGITS),
        4, DIGIT_OFFSET + 1,
    ]
    return {
        "input_ids": input_ids,
        "labels": digit_tokens(y, NUM_N_DIGITS),
        "instance_id": f"pure_reduction_v2_{split}_{index:05d}",
        "split": split,
        "generator_family": "pure_reduction_reciprocal_sampled",
        "label_exact": True,
        "label_method": "local_generator_plain_mod",
        "result": y,
        "modulus": N_VALUE,
        "modulus_bits": N_VALUE.bit_length(),
        "time_steps": 1,
        "x": p,
    }


def main() -> None:
    output = Path("data/generated/pure_reduction_reciprocal")
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(45)
    seen: set[int] = set()
    values: list[int] = []
    while len(values) < 10_000:
        p = sample_reciprocal(rng)
        if p not in seen:
            seen.add(p)
            values.append(p)
    rng.shuffle(values)
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
        "train": "8,000 reciprocal-sampled 8-digit P values, held-out P split",
        "test": "2,000 disjoint reciprocal-sampled P values",
        "label_format": "P mod 323, MSB-first, no squaring involved",
        "sampling": "reciprocal/log-uniform per arXiv 2506.23679 appendix A.1",
    }
    (output / "dataset_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"wrote {len(train)} train / {len(test)} to {output}")
    print(f"sample values (should skew small): {sorted(values)[:10]} ... {sorted(values)[-3:]}")


if __name__ == "__main__":
    main()

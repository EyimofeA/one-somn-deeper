"""Trivial baselines for a mod-family split (Tasks B/C/D), to sanity-check
whether a model's exact-match on a split is actually beating no-op guessing.

Usage:
    python baselines.py data/generated/square_mod/hard.jsonl
    python baselines.py data/generated/square_mod/val_iid.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def digits_to_int(digits: list[int]) -> int:
    return int("".join(str(d) for d in digits))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--output-width", type=int, default=4, help="mod result width (4 for Tasks B/C/D)")
    args = ap.parse_args()

    rows = load_jsonl(args.path)
    n = len(rows)
    targets = [row["labels"][-args.output_width :] for row in rows]
    target_ints = [digits_to_int(t) for t in targets]

    always_zero = sum(1 for t in targets if all(d == 0 for d in t)) / n
    counts = Counter(tuple(t) for t in targets)
    most_common_tuple, most_common_count = counts.most_common(1)[0]
    most_common_frac = most_common_count / n

    # "target length" = number of significant digits (leading zeros stripped), 0 counted as length 1
    lengths = [max(1, len(str(v))) for v in target_ints]
    length_counts = Counter(lengths)
    modal_length, modal_length_count = length_counts.most_common(1)[0]

    frac_remainder_zero = sum(1 for v in target_ints if v == 0) / n

    print(f"split: {args.path}  (n={n})")
    print(f"always-predict-zero exact-match:     {always_zero:.4f}")
    print(f"most-common-target exact-match:      {most_common_frac:.4f}  (mode={most_common_tuple})")
    print(f"target-length distribution:          {dict(sorted(length_counts.items()))}")
    print(f"  -> guessing the modal length only:  {modal_length_count / n:.4f}  (modal length={modal_length})")
    print(f"frequency of remainder == 0:          {frac_remainder_zero:.4f}")


if __name__ == "__main__":
    main()

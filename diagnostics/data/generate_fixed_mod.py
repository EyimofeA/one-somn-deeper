"""Generate fixed-modulus Task-B data with disjoint u per modulus."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .generate import record, sample_u_stratified, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, action="append", default=None, help="repeat for each fixed modulus")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train", type=int, default=99_990)
    ap.add_argument("--val", type=int, default=9_990)
    ap.add_argument("--paired-u", action="store_true", help="use the same disjoint u values under every modulus")
    args = ap.parse_args()

    moduli = args.n or [1349]
    if args.train % len(moduli) or args.val % len(moduli):
        raise ValueError("train and val counts must divide evenly across moduli")
    rng = random.Random(args.seed)
    used = {n: set() for n in moduli}

    def sample(n: int, count: int) -> list[int]:
        values = set(sample_u_stratified(n, count, rng, exclude=used[n]))
        # Fixed N has finite tiny strata (notably only N values where u<N),
        # so the shared sampler cannot fill a large disjoint split by itself.
        while len(values) < count:
            u = rng.randint(0, (n - 1) ** 2)
            if u not in used[n]:
                values.add(u)
        values = list(values)
        rng.shuffle(values)
        used[n].update(values)
        return values

    out = Path(args.out)
    splits = {}
    if args.paired_u:
        paired_used: set[int] = set()

        def sample_paired(count: int) -> list[int]:
            values = set(sample_u_stratified(min(moduli), count, rng, exclude=paired_used))
            while len(values) < count:
                u = rng.randint(0, (min(moduli) - 1) ** 2)
                if u not in paired_used:
                    values.add(u)
            values = list(values)
            rng.shuffle(values)
            paired_used.update(values)
            return values

        for split, count in (("train", args.train), ("val_iid", args.val), ("heldout_u", args.val)):
            rows = [record("mod", False, split, i, n=n, u=u)
                    for i, u in enumerate(sample_paired(count // len(moduli))) for n in moduli]
            rng.shuffle(rows)
            for i, row in enumerate(rows):
                row["instance_id"] = f"mod_{split}_{i:06d}"
            splits[split] = rows
            write_jsonl(out / f"{split}.jsonl", rows)
    else:
        for split, count in (("train", args.train), ("val_iid", args.val), ("heldout_u", args.val)):
            rows = []
            for n in moduli:
                rows.extend(record("mod", False, split, len(rows), n=n, u=u) for u in sample(n, count // len(moduli)))
            rng.shuffle(rows)
            for i, row in enumerate(rows):
                row["instance_id"] = f"mod_{split}_{i:06d}"
            splits[split] = rows
            write_jsonl(out / f"{split}.jsonl", rows)
    (out / "manifest.json").write_text(json.dumps({
        "task": "mod", "moduli": moduli, "seed": args.seed,
        "u_ranges": {str(n): [0, (n - 1) ** 2] for n in moduli},
        "sampling": "paired u across moduli" if args.paired_u else "stratified sampler, then uniform unseen-u fill when fixed-N strata exhaust; splits have disjoint u per modulus",
        "counts": {name: len(rows) for name, rows in splits.items()},
    }, indent=2) + "\n")
    print(f"wrote fixed-N={moduli}: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))


if __name__ == "__main__":
    main()

"""Generate all diagnostic datasets (Tasks A/B/C/D x 6 splits) as jsonl.

Usage:
    python -m data.generate --task square --out data/generated/square --scale small
    python -m data.generate --task all --out data/generated --scale full

`--scale small` produces a CPU-smoke-test-sized dataset (hundreds of rows);
`--scale full` produces the spec's suggested sizes (100k train / 10k val /
10k per diagnostic split). Every split is deduplicated by (task-relevant)
input tuple within itself.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from . import splits as sp
from . import tokens as tok

SCALES = {
    "small": dict(n_train=400, n_val=100, n_diag=100, n_train_moduli=6, n_test_moduli=3, n_factor=3),
    # n_train_moduli/n_test_moduli/n_factor are capped by how many distinct
    # 10-11 bit semiprimes actually exist (130, with primes up to 11 bits) --
    # see data/splits.py's ValueError if these are raised past that ceiling.
    "full": dict(n_train=100_000, n_val=10_000, n_diag=10_000, n_train_moduli=90, n_test_moduli=30, n_factor=30),
}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def record(task: str, reverse_digits: bool, split: str, index: int, **fields) -> dict:
    if task == "square":
        x = fields["x"]
        input_ids, labels = tok.encode_square(x, reverse_digits)
        meta = dict(x=x, y=x * x, x_digits=len(str(x)), carry_chain=fields.get("carry_chain", 0))
    elif task == "mod":
        n, u = fields["n"], fields["u"]
        input_ids, labels = tok.encode_mod(n, u, reverse_digits)
        meta = dict(
            n=n, u=u, y=u % n, modulus_bits=n.bit_length(), quotient=u // n,
            dist_to_multiple=min(u % n, n - (u % n)),
        )
    elif task == "mod_extra":
        raise ValueError("unreachable")
    else:  # square_mod, square_mod_trace
        n, x = fields["n"], fields["x"]
        enc = tok.ENCODERS[task]
        input_ids, labels = enc(n, x, reverse_digits)
        y = (x * x) % n
        meta = dict(
            n=n, x=x, y=y, modulus_bits=n.bit_length(), x_digits=len(str(x)),
            quotient=(x * x) // n, dist_to_multiple=min(y, n - y),
        )
    return {
        "input_ids": input_ids,
        "labels": labels,
        "instance_id": f"{task}_{split}_{index:06d}",
        "split": split,
        "task": task,
        **meta,
    }


def carry_chain_length(x: int) -> int:
    """Length of the longest run of decimal digits >= 5 in x (proxy for carry propagation depth)."""
    s = str(x)
    best = cur = 0
    for c in s:
        if int(c) >= 5:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def gen_square_task(out_dir: Path, scale: dict, seed: int, reverse_digits: bool) -> None:
    rng = random.Random(seed)
    x_max = 10 ** tok.NUM_SQUARE_X_DIGITS - 1
    seen: set[int] = set()

    def sample_unique(n: int) -> list[int]:
        vals = []
        while len(vals) < n:
            x = rng.randint(1, x_max)
            if x not in seen:
                seen.add(x)
                vals.append(x)
        return vals

    train_x = sample_unique(scale["n_train"])
    val_x = sample_unique(scale["n_val"])
    # held-out x: same domain, disjoint from train/val by construction (global `seen` set)
    heldout_x = sample_unique(scale["n_diag"])
    # hard arithmetic: carry-chain-enriched x, still disjoint
    hard_x: list[int] = []
    attempts = 0
    while len(hard_x) < scale["n_diag"] and attempts < scale["n_diag"] * 50:
        attempts += 1
        x = rng.randint(1, x_max)
        if x in seen:
            continue
        if carry_chain_length(x) >= 2:
            seen.add(x)
            hard_x.append(x)
    while len(hard_x) < scale["n_diag"]:  # top up if the enrichment loop ran dry
        hard_x.extend(sample_unique(scale["n_diag"] - len(hard_x)))

    for split, xs in (("train", train_x), ("val_iid", val_x), ("heldout_x", heldout_x), ("hard", hard_x)):
        rows = [
            record("square", reverse_digits, split, i, x=x, carry_chain=carry_chain_length(x))
            for i, x in enumerate(xs)
        ]
        write_jsonl(out_dir / f"{split}.jsonl", rows)
    print(f"[square] wrote train={len(train_x)} val_iid={len(val_x)} heldout_x={len(heldout_x)} hard={len(hard_x)}")


def sample_u_stratified(n: int, count: int, rng: random.Random, exclude: set[int] | None = None) -> list[int]:
    """Cover u < N, u near multiples of N (below/at/above), and large quotients.
    `exclude` (if given) is a set of u values already claimed by another split
    for this same modulus -- e.g. train's u's, when sampling val_iid/heldout_u
    for the same n -- so the three splits never draw the same u by chance."""
    u_max = (n - 1) ** 2
    max_q = max(1, u_max // n)
    exclude = exclude or set()
    out: set[int] = set()
    buckets = max(1, count // 5)

    def add(u: int) -> None:
        u = max(0, min(u_max, u))
        if u not in exclude:
            out.add(u)

    # u < N
    for _ in range(buckets):
        add(rng.randint(0, n - 1))
    # near a multiple of N: pick a random quotient, offset by -2..2
    for _ in range(buckets):
        q = rng.randint(0, max_q)
        add(q * n + rng.randint(-2, 2))
    # large quotient
    for _ in range(buckets):
        q = rng.randint(max_q // 2, max_q)
        add(q * n + rng.randint(0, n - 1))
    # remainder near 0
    for _ in range(buckets):
        q = rng.randint(0, max_q)
        add(q * n + rng.randint(0, min(3, n - 1)))
    # remainder near N-1
    attempts = 0
    max_attempts = count * 200
    while len(out) < count and attempts < max_attempts:
        attempts += 1
        q = rng.randint(0, max_q)
        r = n - 1 - rng.randint(0, min(3, n - 1))
        add(q * n + r)
    return list(out)[:count]


def gen_mod_task(out_dir: Path, scale: dict, seed: int, reverse_digits: bool) -> None:
    rng = random.Random(seed + 1)
    pool = sp.ModulusPool(bit_lo=10, bit_hi=11, seed=seed)
    pool.build(scale["n_train_moduli"], scale["n_test_moduli"])
    train_n = [n for n, _, _ in pool.train_moduli]

    used_u_by_n: dict[int, set[int]] = {n: set() for n in train_n}

    def rows_for(split: str, moduli: list[int], per_modulus: int, stratified: bool, track_used: bool = False) -> list[dict]:
        out = []
        idx = 0
        for n in moduli:
            exclude = used_u_by_n.get(n) if track_used else None
            us = (
                sample_u_stratified(n, per_modulus, rng, exclude=exclude)
                if stratified
                else [rng.randint(0, (n - 1) ** 2) for _ in range(per_modulus)]
            )
            if track_used and n in used_u_by_n:
                used_u_by_n[n].update(us)
            for u in us:
                out.append(record("mod", reverse_digits, split, idx, n=n, u=u))
                idx += 1
        return out

    per_train = max(1, scale["n_train"] // max(1, len(train_n)))
    per_diag = max(1, scale["n_diag"] // max(1, len(train_n)))
    # train/val_iid/heldout_u share the same moduli (train_n) -- track claimed u's
    # per modulus across all three calls so none of them can draw the same u.
    train_rows = rows_for("train", train_n, per_train, stratified=True, track_used=True)
    val_rows = rows_for("val_iid", train_n, max(1, scale["n_val"] // max(1, len(train_n))), stratified=True, track_used=True)
    heldout_u_rows = rows_for("heldout_u", train_n, per_diag, stratified=True, track_used=True)

    test_n = [n for n, _, _ in pool.test_modulus_moduli]
    sp.assert_no_modulus_overlap(pool.train_moduli, pool.test_modulus_moduli)
    heldout_modulus_rows = rows_for(
        "heldout_modulus", test_n, max(1, scale["n_diag"] // max(1, len(test_n))), stratified=True
    )

    factor_count = max(2, scale["n_factor"] // 2)
    factor_train, factor_test = sp.held_out_factor_moduli(
        pool.factor_train_primes, pool.factor_test_primes, pool.bit_lo, pool.bit_hi, factor_count, rng
    )
    sp.assert_no_factor_overlap(factor_train, factor_test)
    heldout_factor_rows = rows_for(
        "heldout_factor", [n for n, _, _ in factor_test], max(1, scale["n_diag"] // max(1, len(factor_test))), True
    )

    len12 = sp.semiprimes_with_bits(sp.primes_in_bit_range(4, 12), 12, 12)
    len13 = sp.semiprimes_with_bits(sp.primes_in_bit_range(4, 13), 13, 13)
    rng.shuffle(len12)
    rng.shuffle(len13)
    n_len = max(2, scale["n_factor"] // 2)
    len12_n = [n for n, _, _ in len12[:n_len]]
    len13_n = [n for n, _, _ in len13[:n_len]]
    len12_rows = rows_for("length12", len12_n, max(1, scale["n_diag"] // max(1, len(len12_n))), True)
    len13_rows = rows_for("length13", len13_n, max(1, scale["n_diag"] // max(1, len(len13_n))), True)

    used_nu = {(r["n"], r["u"]) for r in train_rows + val_rows + heldout_u_rows}
    hard_rows = []
    idx = 0
    for n, p, q in pool.train_moduli:
        for x in sp.hard_x_candidates(n, p, q):
            u = x * x
            if (n, u) in used_nu:
                continue
            hard_rows.append(record("mod", reverse_digits, "hard", idx, n=n, u=u))
            idx += 1
    rng.shuffle(hard_rows)
    hard_rows = hard_rows[: scale["n_diag"]]

    for name, rows in (
        ("train", train_rows), ("val_iid", val_rows), ("heldout_u", heldout_u_rows),
        ("heldout_modulus", heldout_modulus_rows), ("heldout_factor", heldout_factor_rows),
        ("length12", len12_rows), ("length13", len13_rows), ("hard", hard_rows),
    ):
        write_jsonl(out_dir / f"{name}.jsonl", rows)
    print(f"[mod] wrote " + ", ".join(f"{n}={len(r)}" for n, r in (
        ("train", train_rows), ("val_iid", val_rows), ("heldout_u", heldout_u_rows),
        ("heldout_modulus", heldout_modulus_rows), ("heldout_factor", heldout_factor_rows),
        ("length12", len12_rows), ("length13", len13_rows), ("hard", hard_rows),
    )))


def gen_square_mod_task(out_dir: Path, scale: dict, seed: int, reverse_digits: bool, task: str) -> None:
    rng = random.Random(seed + 2)
    pool = sp.ModulusPool(bit_lo=10, bit_hi=11, seed=seed)
    pool.build(scale["n_train_moduli"], scale["n_test_moduli"])

    def rows_for(split: str, moduli: list[tuple[int, int, int]], per_modulus: int) -> list[dict]:
        out = []
        idx = 0
        for n, p, q in moduli:
            phi = (p - 1) * (q - 1)  # exact size of the coprime pool for a semiprime n=p*q
            xs = sp.sample_coprime_x(n, p, q, min(per_modulus, phi), rng)
            for x in xs:
                out.append(record(task, reverse_digits, split, idx, n=n, x=x))
                idx += 1
        return out

    per_train = max(1, scale["n_train"] // max(1, len(pool.train_moduli)))
    per_diag = max(1, scale["n_diag"] // max(1, len(pool.train_moduli)))

    all_train_x_rows = rows_for("_pool", pool.train_moduli, per_train + max(2, scale["n_val"] // max(1, len(pool.train_moduli))) + per_diag)
    # split coprime x's per modulus into train/val_iid/heldout_x disjoint chunks
    by_n: dict[int, list[dict]] = {}
    for r in all_train_x_rows:
        by_n.setdefault(r["n"], []).append(r)
    train_rows, val_rows, heldout_x_rows = [], [], []
    for n, rs in by_n.items():
        rng.shuffle(rs)
        n_val = max(1, len(rs) * 20 // 100)
        n_ho = max(1, len(rs) * 20 // 100)
        val_rows += rs[:n_val]
        heldout_x_rows += rs[n_val : n_val + n_ho]
        train_rows += rs[n_val + n_ho :]
    for i, r in enumerate(train_rows):
        r["instance_id"] = f"{task}_train_{i:06d}"
        r["split"] = "train"
    for i, r in enumerate(val_rows):
        r["instance_id"] = f"{task}_val_iid_{i:06d}"
        r["split"] = "val_iid"
    for i, r in enumerate(heldout_x_rows):
        r["instance_id"] = f"{task}_heldout_x_{i:06d}"
        r["split"] = "heldout_x"

    test_moduli = pool.test_modulus_moduli
    sp.assert_no_modulus_overlap(pool.train_moduli, test_moduli)
    heldout_modulus_rows = rows_for("heldout_modulus", test_moduli, per_diag)

    factor_count = max(2, scale["n_factor"] // 2)
    factor_train, factor_test = sp.held_out_factor_moduli(
        pool.factor_train_primes, pool.factor_test_primes, pool.bit_lo, pool.bit_hi, factor_count, rng
    )
    sp.assert_no_factor_overlap(factor_train, factor_test)
    heldout_factor_rows = rows_for("heldout_factor", factor_test, per_diag)

    len12 = sp.semiprimes_with_bits(sp.primes_in_bit_range(4, 12), 12, 12)
    len13 = sp.semiprimes_with_bits(sp.primes_in_bit_range(4, 13), 13, 13)
    rng.shuffle(len12)
    rng.shuffle(len13)
    n_len = max(2, scale["n_factor"] // 2)
    len12_rows = rows_for("length12", len12[:n_len], per_diag)
    len13_rows = rows_for("length13", len13[:n_len], per_diag)

    used_nx = {(r["n"], r["x"]) for r in train_rows + val_rows + heldout_x_rows}
    hard_rows = []
    idx = 0
    for n, p, q in pool.train_moduli:
        for x in sp.hard_x_candidates(n, p, q):
            if (n, x) in used_nx:
                continue
            hard_rows.append(record(task, reverse_digits, "hard", idx, n=n, x=x))
            idx += 1
    rng.shuffle(hard_rows)
    hard_rows = hard_rows[: scale["n_diag"]]

    splits = {
        "train": train_rows, "val_iid": val_rows, "heldout_x": heldout_x_rows,
        "heldout_modulus": heldout_modulus_rows, "heldout_factor": heldout_factor_rows,
        "length12": len12_rows, "length13": len13_rows, "hard": hard_rows,
    }
    for name, rows in splits.items():
        write_jsonl(out_dir / f"{name}.jsonl", rows)
    print(f"[{task}] wrote " + ", ".join(f"{n}={len(r)}" for n, r in splits.items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["square", "mod", "square_mod", "square_mod_trace", "all"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", choices=list(SCALES), default="small")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reverse-digits", action="store_true")
    args = ap.parse_args()

    scale = SCALES[args.scale]
    out = Path(args.out)
    tasks = ["square", "mod", "square_mod", "square_mod_trace"] if args.task == "all" else [args.task]
    for task in tasks:
        task_out = out / task if args.task == "all" else out
        if task == "square":
            gen_square_task(task_out, scale, args.seed, args.reverse_digits)
        elif task == "mod":
            gen_mod_task(task_out, scale, args.seed, args.reverse_digits)
        else:
            gen_square_mod_task(task_out, scale, args.seed, args.reverse_digits, task)


if __name__ == "__main__":
    main()

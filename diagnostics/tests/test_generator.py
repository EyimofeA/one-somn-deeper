import json
from pathlib import Path

from data.generate import SCALES, gen_mod_task, gen_square_mod_task, gen_square_task
from data.tokens import NUM_SQUARE_DIGITS


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def test_square_task_generation_has_no_duplicate_x_within_or_across_splits(tmp_path):
    gen_square_task(tmp_path, SCALES["small"], seed=0, reverse_digits=False)
    seen = set()
    for name in ("train", "val_iid", "heldout_x", "hard"):
        rows = _load(tmp_path / f"{name}.jsonl")
        assert rows
        xs = [r["x"] for r in rows]
        assert len(xs) == len(set(xs)), f"duplicate x within {name}"
        overlap = seen & set(xs)
        assert not overlap, f"{name} reuses x values from an earlier split: {list(overlap)[:5]}"
        seen |= set(xs)


def test_square_task_labels_match_python_square(tmp_path):
    gen_square_task(tmp_path, SCALES["small"], seed=0, reverse_digits=False)
    for r in _load(tmp_path / "train.jsonl"):
        out = r["labels"][-NUM_SQUARE_DIGITS:]
        assert int("".join(str(d) for d in out)) == r["x"] * r["x"]


def test_mod_task_zero_modulus_overlap_between_train_and_heldout_modulus(tmp_path):
    gen_mod_task(tmp_path, SCALES["small"], seed=0, reverse_digits=False)
    train_n = {r["n"] for r in _load(tmp_path / "train.jsonl")}
    heldout_n = {r["n"] for r in _load(tmp_path / "heldout_modulus.jsonl")}
    assert not (train_n & heldout_n)


def test_mod_task_labels_match_python_mod(tmp_path):
    gen_mod_task(tmp_path, SCALES["small"], seed=0, reverse_digits=False)
    for r in _load(tmp_path / "train.jsonl"):
        out = r["labels"][-4:]
        assert int("".join(str(d) for d in out)) == r["u"] % r["n"]


def test_square_mod_task_zero_modulus_overlap_and_zero_factor_overlap(tmp_path):
    gen_square_mod_task(tmp_path, SCALES["small"], seed=0, reverse_digits=False, task="square_mod")
    train_n = {r["n"] for r in _load(tmp_path / "train.jsonl")}
    heldout_modulus_n = {r["n"] for r in _load(tmp_path / "heldout_modulus.jsonl")}
    assert not (train_n & heldout_modulus_n)

    heldout_factor_n = {r["n"] for r in _load(tmp_path / "heldout_factor.jsonl")}
    # heldout_factor moduli must also be disjoint from every train modulus
    assert not (train_n & heldout_factor_n)


def test_square_mod_heldout_x_disjoint_from_train_per_modulus(tmp_path):
    gen_square_mod_task(tmp_path, SCALES["small"], seed=0, reverse_digits=False, task="square_mod")
    train_rows = _load(tmp_path / "train.jsonl")
    ho_rows = _load(tmp_path / "heldout_x.jsonl")
    train_pairs = {(r["n"], r["x"]) for r in train_rows}
    ho_pairs = {(r["n"], r["x"]) for r in ho_rows}
    assert not (train_pairs & ho_pairs)


def test_square_mod_labels_match_python(tmp_path):
    gen_square_mod_task(tmp_path, SCALES["small"], seed=0, reverse_digits=False, task="square_mod")
    for r in _load(tmp_path / "train.jsonl"):
        out = r["labels"][-4:]
        assert int("".join(str(d) for d in out)) == (r["x"] * r["x"]) % r["n"]


def test_mod_hard_split_has_no_overlap_with_train_val_or_heldout_u(tmp_path):
    gen_mod_task(tmp_path, SCALES["small"], seed=0, reverse_digits=False)
    used = set()
    for name in ("train", "val_iid", "heldout_u"):
        used |= {(r["n"], r["u"]) for r in _load(tmp_path / f"{name}.jsonl")}
    hard = {(r["n"], r["u"]) for r in _load(tmp_path / "hard.jsonl")}
    assert not (used & hard)


def test_square_mod_hard_split_has_no_overlap_with_train_val_or_heldout_x(tmp_path):
    gen_square_mod_task(tmp_path, SCALES["small"], seed=0, reverse_digits=False, task="square_mod")
    used = set()
    for name in ("train", "val_iid", "heldout_x"):
        used |= {(r["n"], r["x"]) for r in _load(tmp_path / f"{name}.jsonl")}
    hard = {(r["n"], r["x"]) for r in _load(tmp_path / "hard.jsonl")}
    assert not (used & hard)


def test_length_extrapolation_splits_are_strictly_wider_than_training_range(tmp_path):
    gen_mod_task(tmp_path, SCALES["small"], seed=0, reverse_digits=False)
    train_bits = {r["modulus_bits"] for r in _load(tmp_path / "train.jsonl")}
    len12_bits = {r["modulus_bits"] for r in _load(tmp_path / "length12.jsonl")}
    len13_bits = {r["modulus_bits"] for r in _load(tmp_path / "length13.jsonl")}
    assert train_bits <= {10, 11}
    assert len12_bits == {12}
    assert len13_bits == {13}

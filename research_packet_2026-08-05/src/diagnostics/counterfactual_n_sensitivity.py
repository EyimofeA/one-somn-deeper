"""Counterfactual Task-B evaluation: same held-out u under both moduli."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import yaml

from data.tokens import NUM_MOD_DIGITS, encode_mod
from train import build_model


@torch.no_grad()
def predict(model, rows: list[dict], device: str) -> torch.Tensor:
    out = []
    for start in range(0, len(rows), 256):
        ids = torch.tensor([r["input_ids"] for r in rows[start : start + 256]], device=device)
        logits = model(ids, torch.ones_like(ids, dtype=torch.bool))[:, -NUM_MOD_DIGITS:]
        out.append(logits.argmax(-1).cpu())
    return torch.cat(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", choices=("peak", "final"), default="final")
    ap.add_argument("--n", type=int, action="append", required=True)
    ap.add_argument("--count", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260730)
    args = ap.parse_args()
    if len(args.n) != 2:
        raise ValueError("exactly two moduli are required")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data_path = out.with_suffix(".jsonl")
    if data_path.exists():
        rows = [json.loads(line) for line in data_path.read_text().splitlines()]
    else:
        train_u = {json.loads(line)["u"] for line in Path(args.train).read_text().splitlines()}
        rng = random.Random(args.seed)
        values = []
        limit = (min(args.n) - 1) ** 2
        while len(values) < args.count:
            u = rng.randint(0, limit)
            if u not in train_u:
                train_u.add(u)
                values.append(u)
        rows = []
        for pair_id, u in enumerate(values):
            for n in args.n:
                ids, labels = encode_mod(n, u)
                rows.append({"pair_id": pair_id, "n": n, "u": u, "input_ids": ids, "target": labels[-NUM_MOD_DIGITS:]})
        data_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    run_dir = Path(args.run_dir)
    cfg = yaml.safe_load((run_dir / "config_used.yaml").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg, max_seq_len=len(rows[0]["input_ids"]), task="mod").to(device)
    model.load_state_dict(torch.load(run_dir / f"{args.checkpoint}.pt", map_location=device))
    model.eval()
    preds = predict(model, rows, device).tolist()

    pairs = {}
    for row, pred in zip(rows, preds):
        pairs.setdefault(row["pair_id"], {})[row["n"]] = (pred, row["target"])
    n0, n1 = args.n
    stats = {"n_pairs": len(pairs), "checkpoint": args.checkpoint, "run_dir": str(run_dir)}
    changed = true_changed = correct_alternate = unchanged_despite_target_change = 0
    acc = {n0: 0, n1: 0}
    confusion = {"ignores_n": 0, "responds_wrong_both": 0, "correct_modulus_specific": 0, "partial_or_same_target": 0}
    for pair in pairs.values():
        p0, t0 = pair[n0]
        p1, t1 = pair[n1]
        p0, p1, t0, t1 = tuple(p0), tuple(p1), tuple(t0), tuple(t1)
        pred_changed, target_changed = p0 != p1, t0 != t1
        changed += pred_changed
        true_changed += target_changed
        acc[n0] += p0 == t0
        acc[n1] += p1 == t1
        correct_alternate += pred_changed and p0 == t0 and p1 == t1
        unchanged_despite_target_change += (not pred_changed) and target_changed
        if p0 == t0 and p1 == t1:
            confusion["correct_modulus_specific"] += 1
        elif not pred_changed and target_changed:
            confusion["ignores_n"] += 1
        elif pred_changed and p0 != t0 and p1 != t1:
            confusion["responds_wrong_both"] += 1
        else:
            confusion["partial_or_same_target"] += 1
    total = len(pairs)
    stats.update({
        "prediction_changes_when_n_changes": changed / total,
        "true_target_changes_when_n_changes": true_changed / total,
        "prediction_changes_to_correct_alternate_remainder": correct_alternate / total,
        "prediction_unchanged_despite_different_target": unchanged_despite_target_change / total,
        "accuracy_by_modulus": {str(n0): acc[n0] / total, str(n1): acc[n1] / total},
        "confusion": {k: v / total for k, v in confusion.items()},
        "definition": "correct_alternate requires both predictions to match their respective modulus-specific targets",
    })
    out.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

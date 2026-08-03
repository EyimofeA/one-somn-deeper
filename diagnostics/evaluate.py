"""Stratified evaluation for a trained checkpoint.

Usage:
    python evaluate.py runs/square_transformer --data data/generated/square --splits val_iid heldout_x hard
    python evaluate.py runs/square_mod_transformer --data data/generated/square_mod --splits heldout_modulus \
        --compare-product path/to/square_metrics.json path/to/mod_metrics.json

Loads `<run_dir>/config_used.yaml` + `<run_dir>/peak.pt` (falls back to
final.pt), rebuilds the model, and reports token accuracy / exact-match,
overall and stratified by the fields the spec calls out (bit length, digit
length, carry-chain length, quotient bucket, remainder bucket, seen-vs-
unseen-modulus, recurrence depth for the recurrent model).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from data.dataset import DiagnosticDataset, ShuffledContextDataset
from data.tokens import OUTPUT_WIDTH
from models.recurrent_workspace import RecurrentWorkspaceModel
from models.transformer import StandardTransformer
from train import build_model, extract_targets_and_logits, forward_from_batch


def bucket_distance(d: int) -> str:
    if d <= 2:
        return "near_boundary"
    if d <= 10:
        return "close"
    return "far"


def bucket_quotient(q: int, max_q: int) -> str:
    if max_q <= 0:
        return "single"
    frac = q / max_q
    if frac < 0.1:
        return "small"
    if frac > 0.9:
        return "large"
    return "mid"


@torch.no_grad()
def run_split(model, ds: DiagnosticDataset, output_width: int, model_type: str, device: str, batch_size: int = 128):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    all_correct = []
    all_token_correct = []
    all_meta = []
    idx = 0
    for batch in loader:
        labels = batch["labels"].to(device)
        logits = forward_from_batch(model, batch, device)
        logits, targets = extract_targets_and_logits(logits, labels, output_width, model_type)
        preds = logits.argmax(dim=-1)
        row_correct = (preds == targets).all(dim=-1)
        token_correct = (preds == targets).float().mean(dim=-1)
        for i in range(row_correct.shape[0]):
            all_correct.append(bool(row_correct[i]))
            all_token_correct.append(float(token_correct[i]))
            all_meta.append(ds.meta(idx))
            idx += 1
    return all_correct, all_token_correct, all_meta


def stratify(correct: list[bool], meta: list[dict], task: str, train_moduli: set[int] | None) -> dict:
    out = {}

    def bucket_report(key_fn, name):
        buckets: dict[str, list[bool]] = {}
        for c, m in zip(correct, meta):
            k = key_fn(m)
            if k is None:
                continue
            buckets.setdefault(str(k), []).append(c)
        out[name] = {k: sum(v) / len(v) for k, v in buckets.items()}

    if "modulus_bits" in meta[0]:
        bucket_report(lambda m: m["modulus_bits"], "by_modulus_bit_length")
    if "x_digits" in meta[0]:
        bucket_report(lambda m: m["x_digits"], "by_x_digit_length")
    if "carry_chain" in meta[0]:
        bucket_report(lambda m: m["carry_chain"], "by_carry_chain_length")
    if "quotient" in meta[0]:
        max_q = max((m["quotient"] for m in meta), default=0)
        bucket_report(lambda m: bucket_quotient(m["quotient"], max_q), "by_quotient_size")
    if "dist_to_multiple" in meta[0]:
        bucket_report(lambda m: bucket_distance(m["dist_to_multiple"]), "by_remainder_bucket")
    if train_moduli is not None and "n" in meta[0]:
        bucket_report(lambda m: "seen" if m["n"] in train_moduli else "unseen", "by_seen_vs_unseen_modulus")
    return out


def recurrence_depth_sweep(model: RecurrentWorkspaceModel, ds, output_width, device, batch_size=128) -> dict:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    depths = list(range(1, model.num_loops + 1))
    acc_per_depth = {d: [] for d in depths}
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(device)
            targets = labels[:, -output_width:]
            for d in depths:
                kwargs = {"override_loops": d}
                if "init_input_ids" in batch:
                    kwargs["init_input_ids"] = batch["init_input_ids"].to(device)
                    kwargs["init_attention_mask"] = batch["init_attention_mask"].to(device)
                logits = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    **kwargs,
                )
                preds = logits.argmax(dim=-1)
                row_correct = (preds == targets).all(dim=-1)
                acc_per_depth[d].append(row_correct.float().mean().item())
    return {str(d): sum(v) / len(v) for d, v in acc_per_depth.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--data", required=True, help="directory containing <split>.jsonl files")
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--out", default=None, help="where to write the json report (default: <run_dir>/eval_report.json)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = yaml.safe_load((run_dir / "config_used.yaml").read_text())
    task = cfg["task"]
    output_width = OUTPUT_WIDTH[task]
    device = cfg.get("device", "cpu")
    model_type = cfg["model"]["type"]

    ckpt_path = run_dir / "peak.pt"
    ckpt_kind = "peak"
    if not ckpt_path.exists():
        ckpt_path = run_dir / "final.pt"
        ckpt_kind = "final"

    train_ds = DiagnosticDataset(cfg["data"]["train"])
    model = build_model(cfg, max_seq_len=train_ds.max_len, task=task).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"CHECKPOINT LOADED: {ckpt_path} (kind={ckpt_kind}, task={task}, model_type={model_type})")

    train_moduli = {m["n"] for m in (train_ds.meta(i) for i in range(len(train_ds))) if "n" in m}

    report = {
        "run_dir": str(run_dir), "checkpoint": str(ckpt_path), "checkpoint_kind": ckpt_kind,
        "task": task, "model_type": model_type, "splits": {},
    }
    data_dir = Path(args.data)
    for split in args.splits:
        path = data_dir / f"{split}.jsonl"
        ds = DiagnosticDataset(path)
        if cfg["model"].get("workspace_init_mode") == "shuffled_context":
            ds = ShuffledContextDataset(ds, seed=10_000 + cfg.get("seed", 0))
        correct, token_correct, meta = run_split(model, ds, output_width, model_type, device)
        split_report = {
            "n": len(correct),
            "exact_match": sum(correct) / len(correct),
            "token_accuracy": sum(token_correct) / len(token_correct),
        }
        split_report.update(stratify(correct, meta, task, train_moduli))
        if model_type == "recurrent_workspace":
            split_report["by_recurrence_depth"] = recurrence_depth_sweep(model, ds, output_width, device)
        report["splits"][split] = split_report
        print(f"{split}: exact_match={split_report['exact_match']:.4f} token_accuracy={split_report['token_accuracy']:.4f}")

    out_path = Path(args.out) if args.out else run_dir / "eval_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {out_path}")


def compare_product(acc_a: float, acc_b: float, acc_c: float) -> None:
    predicted = acc_a * acc_b
    print(f"Task A exact-match: {acc_a:.4f}")
    print(f"Task B exact-match: {acc_b:.4f}")
    print(f"Task A * Task B (predicted if composition adds no extra failure): {predicted:.4f}")
    print(f"Task C exact-match (measured): {acc_c:.4f}")
    gap = acc_c - predicted
    print(f"gap (measured - predicted): {gap:+.4f} — {'composition costs extra' if gap < 0 else 'composition is free or beneficial'}")


if __name__ == "__main__":
    main()

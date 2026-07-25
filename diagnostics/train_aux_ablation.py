"""Causal training ablation: does supervising carry / diagonal-sum / both help
Task A squaring, and does the effect survive annealing the auxiliary weight
to zero?

Same standard Transformer (d_model=128, n_layers=4, n_heads=4, d_ff=512),
same data (data/generated/square), same optimizer (AdamW lr=3e-4 wd=0.01,
cosine schedule w/ 5% warmup, grad_clip=1.0), same step budget (50,000,
batch_size=64), same eval cadence (every 1,000 steps, fixed 2,000-example
train subset, early-stop patience 10 evals) as the corrected 50k Task A run
in runs/square_transformer_50k -- the ONLY thing that varies across
conditions is which auxiliary loss (if any) is added on top of the digit
cross-entropy, and whether its weight is annealed to zero.

Conditions (--condition):
  baseline      no auxiliary heads/loss at all (identical param count to
                models/transformer.py's StandardTransformer)
  carry         + auxiliary head predicting normalized (carry_in, carry_out)
                at every output position, MSE loss, fixed weight 1.0
  diagonal      + auxiliary head predicting normalized raw diagonal sum at
                every output position, MSE loss, fixed weight 1.0
  both          both heads above, weight 1.0 each, both fixed for the whole run
  both_annealed both heads, weight linearly decayed 1.0 -> 0 over the first
                50% of steps, then held at 0 for the second half -- tests
                whether the auxiliary signal taught a reusable computation
                (performance should hold after the weight hits zero) or was
                only propping up performance while active (performance should
                regress once removed)

Usage:
    python train_aux_ablation.py --condition baseline --seed 0
    python train_aux_ablation.py --condition both_annealed --seed 2 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import hashlib
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import repro
from analysis_task_a import align_column_feature_to_output, carry_features
from data.dataset import load_jsonl
from data.tokens import IGNORE_INDEX, NUM_SQUARE_DIGITS, PAD
from models.transformer_aux import StandardTransformerAux

W = NUM_SQUARE_DIGITS  # 12
TRAIN_PATH = "data/generated/square/train.jsonl"
VAL_PATH = "data/generated/square/val_iid.jsonl"
TOTAL_STEPS = 50_000
BATCH_SIZE = 64
EVAL_EVERY = 1000
EVAL_TRAIN_SUBSET_SIZE = 2000
EARLY_STOP_PATIENCE = 10
LR = 3e-4
WEIGHT_DECAY = 0.01
WARMUP_FRAC = 0.05
GRAD_CLIP = 1.0
D_MODEL, N_LAYERS, N_HEADS, D_FF, DROPOUT = 128, 4, 4, 512, 0.0

CONDITIONS = {
    "baseline": dict(use_carry=False, use_diag=False, anneal=False, shuffle_carry=False),
    "carry": dict(use_carry=True, use_diag=False, anneal=False, shuffle_carry=False),
    "diagonal": dict(use_carry=False, use_diag=True, anneal=False, shuffle_carry=False),
    "both": dict(use_carry=True, use_diag=True, anneal=False, shuffle_carry=False),
    "both_annealed": dict(use_carry=True, use_diag=True, anneal=True, shuffle_carry=False),
    # A2 control: same backbone/head/weight/optimizer/budget/params/data as
    # "carry", but each training example's carry TARGET is replaced by
    # another example's (a fixed derangement, one per seed, applied only to
    # the training set -- validation is always the real digit-prediction
    # task and never touches carry targets at all).
    "carry_shuffled": dict(use_carry=True, use_diag=False, anneal=False, shuffle_carry=True),
}

LEARNING_CURVE_FIELDS = [
    "step", "train_loss_main", "train_aux_carry_mse", "train_aux_diag_mse",
    "train_token_accuracy", "train_exact_match", "val_token_accuracy", "val_exact_match",
    "lr", "grad_norm", "aux_carry_weight", "aux_diag_weight",
    "examples_per_sec", "steps_per_sec", "gpu_util_pct", "gpu_mem_used_mb",
]


def gpu_stats() -> tuple[float | None, float | None]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            timeout=5,
        ).decode().strip().splitlines()[0]
        util, mem = out.split(",")
        return float(util), float(mem)
    except Exception:
        return None, None


def lr_factor(step: int, total_steps: int, warmup_frac: float) -> float:
    warmup_steps = max(1, int(total_steps * warmup_frac))
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = min((step - warmup_steps) / max(1, total_steps - warmup_steps), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def aux_weight_schedule(step: int, total_steps: int, base_weight: float, anneal: bool) -> float:
    if not anneal:
        return base_weight
    anneal_end = total_steps // 2
    if step >= anneal_end:
        return 0.0
    return base_weight * (1.0 - step / anneal_end)


# ---------------------------------------------------------------------------
# aux target precomputation
# ---------------------------------------------------------------------------
def compute_aux_targets(rows: list[dict]) -> dict[str, np.ndarray]:
    n = len(rows)
    carry_in = np.zeros((n, W), dtype=np.float32)
    carry_out = np.zeros((n, W), dtype=np.float32)
    diag_sum = np.zeros((n, W), dtype=np.float32)
    for i, r in enumerate(rows):
        f = carry_features(int(r["x"]))
        n_sig = f["n_digits_x"]
        carry_in[i] = align_column_feature_to_output(f["carry_in_by_column"], n_sig)
        carry_out[i] = align_column_feature_to_output(f["carry_out_by_column"], n_sig)
        diag_sum[i] = align_column_feature_to_output(f["column_sum"], n_sig)
    return {"carry_in": carry_in, "carry_out": carry_out, "diag_sum": diag_sum}


def derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """A permutation of range(n) with zero fixed points (perm[i] != i for
    all i), fully resampled until valid -- cheap at this scale (P(no fixed
    points) -> 1/e, so ~e ~= 2.7 resamples expected)."""
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm


class SquareAuxDataset(Dataset):
    def __init__(self, rows: list[dict], aux: dict[str, np.ndarray], norm_stats: dict[str, tuple[float, float]]):
        self.rows = rows
        self.aux = aux
        self.norm_stats = norm_stats
        self.max_len = max(len(r["input_ids"]) for r in rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        ids = row["input_ids"]
        labels = row["labels"]
        pad_n = self.max_len - len(ids)
        input_ids = ids + [PAD] * pad_n
        label_ids = labels + [IGNORE_INDEX] * pad_n
        attention_mask = [1] * len(ids) + [0] * pad_n

        ci_mean, ci_std = self.norm_stats["carry_in"]
        co_mean, co_std = self.norm_stats["carry_out"]
        ds_mean, ds_std = self.norm_stats["diag_sum"]
        carry_target = np.stack([
            (self.aux["carry_in"][idx] - ci_mean) / ci_std,
            (self.aux["carry_out"][idx] - co_mean) / co_std,
        ], axis=-1)  # (W, 2)
        diag_target = (self.aux["diag_sum"][idx] - ds_mean) / ds_std  # (W,)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "carry_target": torch.tensor(carry_target, dtype=torch.float32),
            "diag_target": torch.tensor(diag_target, dtype=torch.float32),
        }


@torch.no_grad()
def evaluate_full(model, loader, device) -> dict:
    model.eval()
    row_correct = digit_correct = digit_total = 0
    carry_sq_err = carry_n = diag_sq_err = diag_n = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits, carry_pred, diag_pred = model(input_ids, attention_mask)
        logits = logits[:, -W:, :]
        targets = labels[:, -W:]
        preds = logits.argmax(dim=-1)
        row_correct += (preds == targets).all(dim=-1).sum().item()
        digit_correct += (preds == targets).sum().item()
        digit_total += targets.numel()
        if carry_pred is not None:
            ct = batch["carry_target"].to(device)
            carry_sq_err += ((carry_pred[:, -W:, :] - ct) ** 2).sum().item()
            carry_n += ct.numel()
        if diag_pred is not None:
            dt = batch["diag_target"].to(device)
            diag_sq_err += ((diag_pred[:, -W:, 0] - dt) ** 2).sum().item()
            diag_n += dt.numel()
    model.train()
    n_rows = len(loader.dataset)
    return {
        "exact_match": row_correct / n_rows if n_rows else 0.0,
        "token_accuracy": digit_correct / digit_total if digit_total else 0.0,
        "carry_aux_mse": carry_sq_err / carry_n if carry_n else None,
        "diag_aux_mse": diag_sq_err / diag_n if diag_n else None,
    }


@torch.no_grad()
def per_position_and_bucket_analysis(model, rows: list[dict], device: str, batch_size: int = 256) -> dict:
    """Per-position accuracy, accuracy by term-count bucket, accuracy by
    carry-in-magnitude bucket -- computed post-hoc from a fresh forward pass,
    reusing the same true schoolbook features as analysis_task_a.py."""
    model.eval()
    xs = np.array([r["x"] for r in rows])
    n = len(rows)
    all_preds = []
    for i in range(0, n, batch_size):
        chunk = rows[i : i + batch_size]
        ids = torch.tensor([r["input_ids"] for r in chunk], dtype=torch.long, device=device)
        mask = torch.ones_like(ids, dtype=torch.bool)
        logits, _, _ = model(ids, mask)
        preds = logits[:, -W:, :].argmax(dim=-1).cpu().numpy()
        all_preds.append(preds)
    preds = np.concatenate(all_preds, axis=0)
    targets = np.array([r["labels"][-W:] for r in rows])
    correct = preds == targets
    model.train()

    n_terms = np.zeros((n, W), dtype=int)
    carry_in = np.zeros((n, W), dtype=int)
    for i, x in enumerate(xs):
        f = carry_features(int(x))
        n_sig = f["n_digits_x"]
        n_terms[i] = align_column_feature_to_output(f["contributions"], n_sig)
        carry_in[i] = align_column_feature_to_output(f["carry_in_by_column"], n_sig)

    per_position_acc = {int(p): float(correct[:, p].mean()) for p in range(W)}

    carry_bucket_edges = [0, 1, 5, 10, 20, 1000]
    carry_bucket_labels = ["0", "1-4", "5-9", "10-19", "20+"]
    carry_flat, correct_flat = carry_in.ravel(), correct.ravel()
    acc_by_carry_bucket = {}
    for lo, hi, lab in zip(carry_bucket_edges[:-1], carry_bucket_edges[1:], carry_bucket_labels):
        mask = (carry_flat >= lo) & (carry_flat < hi)
        if mask.sum() > 0:
            acc_by_carry_bucket[lab] = {"n": int(mask.sum()), "accuracy": float(correct_flat[mask].mean())}

    terms_flat = n_terms.ravel()
    acc_by_terms = {}
    for t in np.unique(terms_flat):
        mask = terms_flat == t
        if mask.sum() > 20:
            acc_by_terms[int(t)] = {"n": int(mask.sum()), "accuracy": float(correct_flat[mask].mean())}

    return {
        "exact_match": float(correct.all(axis=1).mean()),
        "token_accuracy": float(correct.mean()),
        "per_position_accuracy": per_position_acc,
        "accuracy_by_carry_in_bucket": acc_by_carry_bucket,
        "accuracy_by_n_terms": acc_by_terms,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=list(CONDITIONS), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-root", default="runs/aux_ablation")
    ap.add_argument("--total-steps", type=int, default=None, help="override TOTAL_STEPS, for smoke tests only")
    ap.add_argument("--eval-every", type=int, default=None, help="override EVAL_EVERY, for smoke tests only")
    ap.add_argument("--d-model", type=int, default=None, help="override D_MODEL (A3 scale check)")
    ap.add_argument("--n-layers", type=int, default=None, help="override N_LAYERS (A3 scale check)")
    ap.add_argument("--n-heads", type=int, default=None, help="override N_HEADS (A3 scale check)")
    ap.add_argument("--d-ff", type=int, default=None, help="override D_FF (A3 scale check)")
    args = ap.parse_args()
    cond = CONDITIONS[args.condition]

    global TOTAL_STEPS, EVAL_EVERY, D_MODEL, N_LAYERS, N_HEADS, D_FF
    if args.total_steps is not None:
        TOTAL_STEPS = args.total_steps
    if args.eval_every is not None:
        EVAL_EVERY = args.eval_every
    if args.d_model is not None:
        D_MODEL = args.d_model
    if args.n_layers is not None:
        N_LAYERS = args.n_layers
    if args.n_heads is not None:
        N_HEADS = args.n_heads
    if args.d_ff is not None:
        D_FF = args.d_ff

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_rows = load_jsonl(TRAIN_PATH)
    val_rows = load_jsonl(VAL_PATH)
    print(f"loaded train={len(train_rows)} val={len(val_rows)}")

    train_aux = compute_aux_targets(train_rows)
    val_aux = compute_aux_targets(val_rows)

    norm_stats = {
        "carry_in": (float(train_aux["carry_in"].mean()), float(train_aux["carry_in"].std() + 1e-8)),
        "carry_out": (float(train_aux["carry_out"].mean()), float(train_aux["carry_out"].std() + 1e-8)),
        "diag_sum": (float(train_aux["diag_sum"].mean()), float(train_aux["diag_sum"].std() + 1e-8)),
    }
    print("normalization stats (mean, std), computed once from train, reused for all conditions/seeds:")
    print(json.dumps(norm_stats, indent=2))

    shuffle_perm = None
    if cond["shuffle_carry"]:
        rng = np.random.default_rng(args.seed)
        shuffle_perm = derangement(len(train_rows), rng)
        assert not np.any(shuffle_perm == np.arange(len(train_rows))), "derangement has a fixed point"
        train_aux = dict(train_aux)  # shallow copy so val_aux (computed above) is untouched
        train_aux["carry_in"] = train_aux["carry_in"][shuffle_perm]
        train_aux["carry_out"] = train_aux["carry_out"][shuffle_perm]
        print(f"A2 shuffled-carry-label control: derangement applied to {len(train_rows)} training rows' "
              f"carry_in/carry_out targets (fixed for this seed, val untouched, diag_sum untouched)")

    train_ds = SquareAuxDataset(train_rows, train_aux, norm_stats)
    val_ds = SquareAuxDataset(val_rows, val_aux, norm_stats)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    train_eval_loader = DataLoader(
        torch.utils.data.Subset(train_ds, range(EVAL_TRAIN_SUBSET_SIZE)), batch_size=BATCH_SIZE, shuffle=False
    )

    model = StandardTransformerAux(
        max_seq_len=train_ds.max_len, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS, d_ff=D_FF,
        dropout=DROPOUT, use_carry_aux=cond["use_carry"], use_diag_aux=cond["use_diag"],
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    n_params_baseline = sum(
        p.numel() for name, p in model.named_parameters() if "carry_head" not in name and "diag_head" not in name
    )
    print(f"condition={args.condition} seed={args.seed} total_params={n_params} backbone_params={n_params_baseline} "
          f"aux_head_params={n_params - n_params_baseline}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_factor(s, TOTAL_STEPS, WARMUP_FRAC))

    out_dir = Path(args.out_root) / f"{args.condition}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    curve_jsonl = (out_dir / "learning_curve.jsonl").open("w")
    curve_csv_f = (out_dir / "learning_curve.csv").open("w", newline="")
    csv_writer = csv.DictWriter(curve_csv_f, fieldnames=LEARNING_CURVE_FIELDS)
    csv_writer.writeheader()

    run_info = {
        "condition": args.condition, "seed": args.seed, "total_steps": TOTAL_STEPS, "batch_size": BATCH_SIZE,
        "lr": LR, "weight_decay": WEIGHT_DECAY, "warmup_frac": WARMUP_FRAC, "grad_clip": GRAD_CLIP,
        "d_model": D_MODEL, "n_layers": N_LAYERS, "n_heads": N_HEADS, "d_ff": D_FF, "dropout": DROPOUT,
        "use_carry_aux": cond["use_carry"], "use_diag_aux": cond["use_diag"], "anneal": cond["anneal"],
        "shuffle_carry": cond["shuffle_carry"],
        "shuffle_perm_sha256": (hashlib.sha256(shuffle_perm.tobytes()).hexdigest() if shuffle_perm is not None else None),
        "aux_base_weight": 1.0, "n_params_total": n_params, "n_params_backbone": n_params_baseline,
        "n_params_aux_heads": n_params - n_params_baseline, "norm_stats": norm_stats,
        "eval_train_subset_size": EVAL_TRAIN_SUBSET_SIZE, "early_stop_patience_evals": EARLY_STOP_PATIENCE,
        "train_example_count": len(train_rows), "device": args.device,
        "repro": repro.capture(),
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_info, indent=2))
    print("RUN CONFIG:", json.dumps(run_info, indent=2))

    run_wallclock_start = time.monotonic()
    step = 0
    best_acc = -1.0
    evals_since_improvement = 0
    last_eval_time = time.monotonic()
    last_eval_step = 0
    stopped_early = False
    train_iter = itertools.cycle(train_loader)

    for step in range(1, TOTAL_STEPS + 1):
        batch = next(train_iter)
        input_ids = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        labels = batch["labels"].to(args.device)

        logits, carry_pred, diag_pred = model(input_ids, attention_mask)
        logits_out = logits[:, -W:, :]
        targets = labels[:, -W:]
        main_loss = nn.functional.cross_entropy(logits_out.reshape(-1, 10), targets.reshape(-1))

        cw = aux_weight_schedule(step, TOTAL_STEPS, run_info["aux_base_weight"], cond["anneal"] and cond["use_carry"])
        dw = aux_weight_schedule(step, TOTAL_STEPS, run_info["aux_base_weight"], cond["anneal"] and cond["use_diag"])

        total_loss = main_loss
        carry_mse_val = 0.0
        diag_mse_val = 0.0
        if carry_pred is not None:
            ct = batch["carry_target"].to(args.device)
            carry_mse = nn.functional.mse_loss(carry_pred[:, -W:, :], ct)
            total_loss = total_loss + cw * carry_mse
            carry_mse_val = carry_mse.item()
        if diag_pred is not None:
            dt = batch["diag_target"].to(args.device)
            diag_mse = nn.functional.mse_loss(diag_pred[:, -W:, 0], dt)
            total_loss = total_loss + dw * diag_mse
            diag_mse_val = diag_mse.item()

        optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP).item()
        optimizer.step()
        scheduler.step()

        if step % EVAL_EVERY == 0:
            now = time.monotonic()
            elapsed = now - last_eval_time
            steps_done = step - last_eval_step
            steps_per_sec = steps_done / elapsed if elapsed > 0 else 0.0
            examples_per_sec = steps_per_sec * BATCH_SIZE
            last_eval_time, last_eval_step = now, step
            util_pct, mem_mb = gpu_stats()

            val_metrics = evaluate_full(model, val_loader, args.device)
            train_metrics = evaluate_full(model, train_eval_loader, args.device)
            lr = optimizer.param_groups[0]["lr"]

            row = {
                "step": step, "train_loss_main": main_loss.item(),
                "train_aux_carry_mse": carry_mse_val, "train_aux_diag_mse": diag_mse_val,
                "train_token_accuracy": train_metrics["token_accuracy"], "train_exact_match": train_metrics["exact_match"],
                "val_token_accuracy": val_metrics["token_accuracy"], "val_exact_match": val_metrics["exact_match"],
                "lr": lr, "grad_norm": grad_norm, "aux_carry_weight": cw, "aux_diag_weight": dw,
                "examples_per_sec": examples_per_sec, "steps_per_sec": steps_per_sec,
                "gpu_util_pct": util_pct, "gpu_mem_used_mb": mem_mb,
            }
            curve_jsonl.write(json.dumps(row) + "\n")
            curve_jsonl.flush()
            csv_writer.writerow(row)
            curve_csv_f.flush()

            if val_metrics["exact_match"] > best_acc:
                best_acc = val_metrics["exact_match"]
                evals_since_improvement = 0
                torch.save(model.state_dict(), out_dir / "peak.pt")
            else:
                evals_since_improvement += 1

            print(
                f"[{args.condition} seed={args.seed}] step={step}/{TOTAL_STEPS} main_loss={main_loss.item():.4f} "
                f"carry_mse={carry_mse_val:.4f} diag_mse={diag_mse_val:.4f} "
                f"train_exact={train_metrics['exact_match']:.4f} val_exact={val_metrics['exact_match']:.4f} "
                f"best={best_acc:.4f} steps/s={steps_per_sec:.1f} no_improve={evals_since_improvement}/{EARLY_STOP_PATIENCE}"
            )

            if evals_since_improvement >= EARLY_STOP_PATIENCE:
                print(f"EARLY STOP at step {step}")
                stopped_early = True
                break

    curve_jsonl.close()
    curve_csv_f.close()
    torch.save(model.state_dict(), out_dir / "final.pt")

    print("Running post-hoc per-position / term-count / carry-magnitude analysis on val set...")
    final_analysis = per_position_and_bucket_analysis(model, val_rows, args.device)
    final_val_metrics = evaluate_full(model, val_loader, args.device)
    report = {
        "run_config": run_info, "stopped_early": stopped_early, "best_val_exact_match": best_acc,
        "final_step": step, "final_val_metrics": final_val_metrics, "post_hoc_analysis": final_analysis,
        "wallclock_seconds": time.monotonic() - run_wallclock_start,
    }
    (out_dir / "eval_report.json").write_text(json.dumps(report, indent=2))
    print(f"done. condition={args.condition} seed={args.seed} best_val_exact={best_acc:.4f} "
          f"final_val_exact={final_val_metrics['exact_match']:.4f}. wrote {out_dir}/eval_report.json")


if __name__ == "__main__":
    main()

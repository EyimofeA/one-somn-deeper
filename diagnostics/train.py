"""Generic trainer: reads a yaml config, trains either baseline on one task.

Usage:
    python train.py configs/square.yaml
    python train.py configs/square_mod.yaml --override model.type=recurrent_workspace

Metrics land in <out_dir>/metrics.jsonl. Plot with:
    python plot_metrics.py runs/square_transformer
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import subprocess
import time
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Subset

from data.dataset import DiagnosticDataset
from data.tokens import OUTPUT_WIDTH
from models.recurrent_workspace import RecurrentWorkspaceModel
from models.transformer import StandardTransformer

LEARNING_CURVE_FIELDS = [
    "step", "train_loss", "train_token_accuracy", "train_exact_match",
    "val_token_accuracy", "val_exact_match", "lr", "grad_norm",
    "examples_per_sec", "steps_per_sec", "gpu_util_pct", "gpu_mem_used_mb",
]


def gpu_stats() -> tuple[float | None, float | None]:
    """(utilization %, memory used MiB) via nvidia-smi, or (None, None) off-GPU."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            timeout=5,
        ).decode().strip().splitlines()[0]
        util, mem = out.split(",")
        return float(util), float(mem)
    except Exception:
        return None, None


def set_nested(cfg: dict, dotted_key: str, value: str) -> None:
    keys = dotted_key.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    # best-effort type coercion so CLI overrides don't stay strings
    for cast in (int, float):
        try:
            value = cast(value)
            break
        except ValueError:
            continue
    if value in ("true", "false"):
        value = value == "true"
    d[keys[-1]] = value


def load_config(path: str, overrides: list[str]) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for o in overrides:
        key, _, value = o.partition("=")
        set_nested(cfg, key, value)
    return cfg


def build_model(cfg: dict, max_seq_len: int, task: str) -> nn.Module:
    m = cfg["model"]
    output_width = OUTPUT_WIDTH[task]
    if m["type"] == "transformer":
        return StandardTransformer(
            max_seq_len=max_seq_len,
            d_model=m.get("d_model", 128),
            n_layers=m.get("n_layers", 4),
            n_heads=m.get("n_heads", 4),
            d_ff=m.get("d_ff", 512),
            dropout=m.get("dropout", 0.0),
        )
    if m["type"] == "recurrent_workspace":
        workspace_size = m.get("workspace_size", max(8, output_width))
        return RecurrentWorkspaceModel(
            max_seq_len=max_seq_len,
            d_model=m.get("d_model", 128),
            n_heads=m.get("n_heads", 4),
            d_ff=m.get("d_ff", 512),
            context_layers=m.get("context_layers", 2),
            workspace_size=workspace_size,
            num_output_slots=m.get("num_output_slots", output_width),
            num_loops=m.get("num_loops", 8),
        )
    raise ValueError(f"unknown model.type {m['type']!r}")


def extract_targets_and_logits(logits: torch.Tensor, labels: torch.Tensor, output_width: int, model_type: str):
    targets = labels[:, -output_width:]
    if model_type == "transformer":
        logits = logits[:, -output_width:, :]
    # recurrent_workspace already returns exactly (batch, output_width, digit_vocab)
    return logits, targets


def lr_factor(step: int, total_steps: int, warmup_frac: float) -> float:
    warmup_steps = max(1, int(total_steps * warmup_frac))
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = min((step - warmup_steps) / max(1, total_steps - warmup_steps), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def param_l2(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.requires_grad:
            total += p.detach().float().pow(2).sum().item()
    return math.sqrt(total)


@torch.no_grad()
def grad_l2(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().float().pow(2).sum().item()
    return math.sqrt(total)


@torch.no_grad()
def evaluate_exact_match(model: nn.Module, loader: DataLoader, output_width: int, model_type: str, device: str) -> tuple[float, float]:
    """Returns (exact_match, token_accuracy) over the whole loader."""
    model.eval()
    row_correct_total = digit_correct_total = digit_total = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids, attention_mask)
        logits, targets = extract_targets_and_logits(logits, labels, output_width, model_type)
        preds = logits.argmax(dim=-1)
        row_correct_total += (preds == targets).all(dim=-1).sum().item()
        digit_correct_total += (preds == targets).sum().item()
        digit_total += targets.numel()
    model.train()
    n_rows = len(loader.dataset)
    return (row_correct_total / n_rows if n_rows else 0.0, digit_correct_total / digit_total if digit_total else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--override", action="append", default=[], dest="overrides")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)

    torch.manual_seed(cfg.get("seed", 0))
    task = cfg["task"]
    output_width = OUTPUT_WIDTH[task]
    device = cfg.get("device", "cpu")

    train_ds = DiagnosticDataset(cfg["data"]["train"])
    val_ds = DiagnosticDataset(cfg["data"]["val"])
    batch_size = cfg["optim"].get("batch_size", 64)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    eval_train_subset_size = cfg.get("eval_train_subset_size")
    train_eval_loader = None
    if eval_train_subset_size:
        n = min(eval_train_subset_size, len(train_ds))
        train_eval_loader = DataLoader(Subset(train_ds, range(n)), batch_size=batch_size, shuffle=False)

    model = build_model(cfg, max_seq_len=train_ds.max_len, task=task).to(device)
    model_type = cfg["model"]["type"]

    optim_cfg = cfg["optim"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=optim_cfg.get("lr", 3e-4), weight_decay=optim_cfg.get("weight_decay", 0.01)
    )
    steps_per_epoch = len(train_loader)
    fixed_total_steps = optim_cfg.get("total_steps")
    if fixed_total_steps:
        total_steps = int(fixed_total_steps)
        epochs = None  # step-driven, not epoch-driven
    else:
        epochs = optim_cfg.get("epochs", 5)
        total_steps = max(1, epochs * steps_per_epoch)
    warmup_frac = optim_cfg.get("warmup_frac", 0.05)
    grad_clip = optim_cfg.get("grad_clip", 1.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, total_steps, warmup_frac)
    )

    out_dir = Path(cfg.get("out_dir", "runs/default"))
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    curve_jsonl_path = out_dir / "learning_curve.jsonl"
    curve_csv_path = out_dir / "learning_curve.csv"
    log_every = cfg.get("log_every", 50)
    eval_every = cfg.get("eval_every", steps_per_epoch)
    early_stop_patience = cfg.get("early_stop_patience")  # in number of eval events, or None to disable

    run_info = {
        "type": "run_config",
        "task": task,
        "dataset_train_path": str(cfg["data"]["train"]),
        "dataset_val_path": str(cfg["data"]["val"]),
        "train_example_count": len(train_ds),
        "eval_train_subset_size": eval_train_subset_size,
        "batch_size": batch_size,
        "optimizer_steps": total_steps,
        "epochs": epochs,
        "learning_rate": optim_cfg.get("lr", 3e-4),
        "weight_decay": optim_cfg.get("weight_decay", 0.01),
        "dropout": cfg["model"].get("dropout", 0.0),
        "warmup_steps": max(1, int(total_steps * warmup_frac)),
        "early_stop_patience_evals": early_stop_patience,
        "model_class": type(model).__name__,
        "model_type": model_type,
        "device": device,
    }
    print("RUN CONFIG:", json.dumps(run_info, indent=2))

    step = 0
    best_acc = -1.0
    evals_since_improvement = 0
    prev_weight_norm = param_l2(model)
    last_eval_time = time.monotonic()
    last_eval_step = 0
    stopped_early = False

    with metrics_path.open("w") as metrics_f, curve_jsonl_path.open("w") as curve_f, \
         curve_csv_path.open("w", newline="") as curve_csv_f:
        metrics_f.write(json.dumps(run_info) + "\n")
        metrics_f.flush()
        csv_writer = csv.DictWriter(curve_csv_f, fieldnames=LEARNING_CURVE_FIELDS)
        csv_writer.writeheader()

        train_iter = itertools.cycle(train_loader)  # steps, not epochs, drive the loop
        for step in range(1, total_steps + 1):
            batch = next(train_iter)
            epoch = (step - 1) // steps_per_epoch
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            logits, targets = extract_targets_and_logits(logits, labels, output_width, model_type)
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            grad_norm_pre = grad_l2(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                train_exact = (preds == targets).all(dim=-1).float().mean().item()
                train_token_acc = (preds == targets).float().mean().item()

            if step % log_every == 0:
                weight_norm = param_l2(model)
                weight_update = abs(weight_norm - prev_weight_norm)
                prev_weight_norm = weight_norm
                lr = optimizer.param_groups[0]["lr"]
                rec = {
                    "type": "train",
                    "step": step,
                    "epoch": epoch,
                    "loss": loss.item(),
                    "exact_accuracy": train_exact,
                    "token_accuracy": train_token_acc,
                    "lr": lr,
                    "grad_norm": grad_norm_pre,
                    "weight_norm": weight_norm,
                    "weight_update": weight_update,
                }
                metrics_f.write(json.dumps(rec) + "\n")
                metrics_f.flush()

            if step % eval_every == 0:
                now = time.monotonic()
                elapsed = now - last_eval_time
                steps_done = step - last_eval_step
                steps_per_sec = steps_done / elapsed if elapsed > 0 else 0.0
                examples_per_sec = steps_per_sec * batch_size
                last_eval_time, last_eval_step = now, step
                util_pct, mem_mb = gpu_stats()

                val_exact, val_token_acc = evaluate_exact_match(model, val_loader, output_width, model_type, device)
                if train_eval_loader is not None:
                    fixed_train_exact, fixed_train_token = evaluate_exact_match(
                        model, train_eval_loader, output_width, model_type, device
                    )
                else:
                    fixed_train_exact, fixed_train_token = train_exact, train_token_acc

                rec = {
                    "type": "eval", "step": step, "epoch": epoch, "split": "val_iid",
                    "exact_accuracy": val_exact, "token_accuracy": val_token_acc,
                }
                metrics_f.write(json.dumps(rec) + "\n")
                metrics_f.flush()

                lr = optimizer.param_groups[0]["lr"]
                curve_row = {
                    "step": step,
                    "train_loss": loss.item(),
                    "train_token_accuracy": fixed_train_token,
                    "train_exact_match": fixed_train_exact,
                    "val_token_accuracy": val_token_acc,
                    "val_exact_match": val_exact,
                    "lr": lr,
                    "grad_norm": grad_norm_pre,
                    "examples_per_sec": examples_per_sec,
                    "steps_per_sec": steps_per_sec,
                    "gpu_util_pct": util_pct,
                    "gpu_mem_used_mb": mem_mb,
                }
                curve_f.write(json.dumps(curve_row) + "\n")
                curve_f.flush()
                csv_writer.writerow(curve_row)
                curve_csv_f.flush()

                if val_exact > best_acc:
                    best_acc = val_exact
                    evals_since_improvement = 0
                    torch.save(model.state_dict(), out_dir / "peak.pt")
                else:
                    evals_since_improvement += 1

                gpu_note = f" gpu_util={util_pct:.0f}% gpu_mem={mem_mb:.0f}MiB" if util_pct is not None else ""
                print(
                    f"step={step}/{total_steps} epoch={epoch} train_loss={loss.item():.4f} "
                    f"train_exact(fixed_subset)={fixed_train_exact:.4f} train_token={fixed_train_token:.4f} "
                    f"val_exact={val_exact:.4f} val_token={val_token_acc:.4f} best_val_exact={best_acc:.4f} "
                    f"steps/s={steps_per_sec:.2f} ex/s={examples_per_sec:.0f}{gpu_note} "
                    f"no_improve={evals_since_improvement}/{early_stop_patience or '-'}"
                )

                if early_stop_patience and evals_since_improvement >= early_stop_patience:
                    print(f"EARLY STOP: no val_exact improvement for {early_stop_patience} consecutive evals.")
                    stopped_early = True
                    break

    torch.save(model.state_dict(), out_dir / "final.pt")
    (out_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg))
    print(f"done. stopped_early={stopped_early}. best val_iid exact-match={best_acc:.4f}. checkpoints in {out_dir}")
    print(f"learning curve: {curve_jsonl_path} / {curve_csv_path}")
    print(f"plot: python plot_metrics.py {out_dir}")


if __name__ == "__main__":
    main()

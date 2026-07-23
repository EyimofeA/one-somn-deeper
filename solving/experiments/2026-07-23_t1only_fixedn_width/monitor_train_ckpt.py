"""Local-only diagnostic training loop with periodic held-out eval + weight/grad norms.

Copy of `scripts_local/monitor_train.py` with ONE addition: saves a checkpoint
whenever the "test" split's exact_accuracy hits a new peak, not just at the
final step. Motivation: every rung-1 run today showed a non-monotone peak
(width sweep, 1800s reruns) that decayed by the run's end — the peak model was
never inspectable because only the final (decayed) state got saved. This lets
us actually look at what the peak checkpoint predicts, not just its number.

NOT the evaluator-owned runner (benchmark/runner.py, left untouched). Local-only
diagnostic tool, zero quota cost — see RESEARCH_PROTOCOL.md section 7.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
import time

import torch

from benchmark.api import OptimizerSpec
from benchmark.manifest import load_manifest
from benchmark.runner import (
    _autocast,
    _compile_model,
    _configure_seed,
    _evaluate,
    _loss_and_accuracy,
    _make_model_spec,
    _next_batch,
    _resolve_batch_sizes,
    _resolve_device,
    _scoring_split_names,
)
from benchmark.validation import validate_model_state, validate_optimizer
from data import make_dataloaders


def _load_submission(path: str):
    spec = importlib.util.spec_from_file_location("submission_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SUBMISSION


def _weight_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        total += float(p.detach().float().pow(2).sum().item())
    return total**0.5


def _grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().float().pow(2).sum().item())
    return total**0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--submission-file", required=True)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-budget-seconds", type=float, default=10.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--peak-split", default="test")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    submission = _load_submission(args.submission_file)
    device = _resolve_device(manifest)
    model_spec = _make_model_spec(manifest)
    seed = manifest.runtime.seeds[0]
    _configure_seed(seed, device)

    batch_size, eval_batch_size = _resolve_batch_sizes(submission, manifest)
    dataloaders = make_dataloaders(
        replace(
            manifest.data,
            seed=seed,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
        ),
        device=device,
    )

    model = submission.build_model(model_spec)
    model_dtype = (
        torch.float32
        if manifest.runtime.amp
        else getattr(torch, manifest.runtime.dtype)
    )
    model = model.to(device=device, dtype=model_dtype)
    validate_model_state(model, manifest.model_state, device)

    bundle = submission.build_optimizer(
        model,
        OptimizerSpec(
            training_time_seconds=manifest.runtime.total_training_time_seconds,
            device_type=device.type,
        ),
    )
    validate_optimizer(bundle, model, device)
    train_model = _compile_model(model, manifest)

    optimizer = bundle.optimizer
    model.train()
    iterator = iter(dataloaders["train"])

    budget_seconds = manifest.runtime.total_training_time_seconds
    started_at = time.monotonic()
    deadline = started_at + budget_seconds
    max_steps = min(
        manifest.runtime.max_steps, submission.max_steps or manifest.runtime.max_steps
    )
    scoring_splits = _scoring_split_names(dataloaders)

    print(f"scoring splits: {scoring_splits}", flush=True)
    out_f = open(args.out, "w")

    def log(record: dict) -> None:
        out_f.write(json.dumps(record) + "\n")
        out_f.flush()
        print(json.dumps(record), flush=True)

    best_peak_acc = -1.0
    peak_ckpt_path = args.out.replace(".jsonl", "_peak.pt")

    step = 0
    while step < max_steps and time.monotonic() < deadline:
        step += 1
        batch, iterator = _next_batch(iterator, dataloaders["train"])
        optimizer.zero_grad(set_to_none=True)
        loss, accuracy, _, _ = _loss_and_accuracy(
            train_model,
            batch,
            manifest,
            device,
            training_loss=submission.training_loss,
        )
        loss.backward()
        if manifest.runtime.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), manifest.runtime.grad_clip)
        gnorm = _grad_norm(model)
        optimizer.step()
        if bundle.scheduler is not None:
            bundle.scheduler.step()

        if step == 1 or step % args.eval_every == 0:
            elapsed = time.monotonic() - started_at
            lr = optimizer.param_groups[0]["lr"]
            wnorm = _weight_norm(model)
            log(
                {
                    "type": "train",
                    "step": step,
                    "elapsed_seconds": elapsed,
                    "loss": float(loss.item()),
                    "exact_accuracy": accuracy,
                    "lr": lr,
                    "weight_norm": wnorm,
                    "grad_norm": gnorm,
                }
            )

            eval_deadline = time.monotonic() + args.eval_budget_seconds
            for split_name in scoring_splits:
                try:
                    metrics = _evaluate(
                        model,
                        dataloaders[split_name],
                        manifest,
                        device,
                        deadline=eval_deadline,
                        budget_seconds=args.eval_budget_seconds,
                    )
                except TimeoutError:
                    continue
                log(
                    {
                        "type": "eval",
                        "split": split_name,
                        "step": step,
                        "elapsed_seconds": elapsed,
                        "loss": metrics["loss"],
                        "exact_accuracy": metrics["exact_accuracy"],
                    }
                )
                if split_name == args.peak_split and metrics["exact_accuracy"] > best_peak_acc:
                    best_peak_acc = metrics["exact_accuracy"]
                    torch.save(
                        {
                            "step": step,
                            "elapsed_seconds": elapsed,
                            "exact_accuracy": metrics["exact_accuracy"],
                            "loss": metrics["loss"],
                            "state_dict": {
                                k: v.detach().cpu() for k, v in model.state_dict().items()
                            },
                        },
                        peak_ckpt_path,
                    )
                    print(
                        f"new peak {args.peak_split} acc {metrics['exact_accuracy']:.4f} "
                        f"at step {step} -> saved {peak_ckpt_path}",
                        flush=True,
                    )

    try:
        _ckpt = args.out.replace(".jsonl", "_final.pt")
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, _ckpt)
        print("saved final checkpoint:", _ckpt, flush=True)
    except Exception as _e:
        print("checkpoint save failed:", _e, flush=True)

    out_f.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()

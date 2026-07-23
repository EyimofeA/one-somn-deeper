"""Load a saved rung-1 checkpoint and print real (x, true y, predicted y) rows.

Local-only inspection tool — not part of any submission, not used at eval
time. Reuses prepare_batch (same collate/target_positions logic the real
harness uses) so decoding is faithful to how scoring actually works.
"""

from __future__ import annotations

import argparse
import importlib.util
from dataclasses import replace

import torch

from benchmark.batches import prepare_batch
from benchmark.manifest import load_manifest
from benchmark.runner import _autocast, _make_model_spec, _resolve_batch_sizes, _resolve_device
from data import make_dataloaders

DIGIT_OFFSET = 7


def _load_submission(path: str):
    spec = importlib.util.spec_from_file_location("submission_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SUBMISSION


def _digits_to_int(token_ids: list[int]) -> str:
    return "".join(str(t - DIGIT_OFFSET) for t in token_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--submission-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-rows", type=int, default=1000)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    submission = _load_submission(args.submission_file)
    device = _resolve_device(manifest)
    model_spec = _make_model_spec(manifest)

    batch_size, eval_batch_size = _resolve_batch_sizes(submission, manifest)
    dataloaders = make_dataloaders(
        replace(
            manifest.data,
            seed=manifest.runtime.seeds[0],
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
        ),
        device=device,
    )

    model = submission.build_model(model_spec)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    if isinstance(ckpt, dict) and "step" in ckpt:
        print(
            f"loaded checkpoint from step={ckpt['step']} "
            f"elapsed={ckpt['elapsed_seconds']:.1f}s "
            f"saved_acc={ckpt['exact_accuracy']:.4f}"
        )
    model = model.to(device=device)
    model.eval()

    n_correct = 0
    n_total = 0
    rows_printed = 0

    with torch.no_grad():
        for batch in dataloaders[args.split]:
            input_ids, targets, attention_mask, target_positions = prepare_batch(batch, device)
            with _autocast(manifest, device):
                logits, _ = model(input_ids, attention_mask=attention_mask)

            batch_size_actual = input_ids.shape[0]
            for row in range(batch_size_actual):
                row_targets = targets[row]
                row_positions = target_positions[row]
                mask = row_targets != -100
                if not mask.any():
                    continue
                valid_positions = row_positions[mask]
                true_tokens = row_targets[mask].tolist()
                row_logits = logits[row, valid_positions, :]
                pred_tokens = row_logits.argmax(dim=-1).tolist()
                correct = true_tokens == pred_tokens
                n_total += 1
                n_correct += int(correct)

                if rows_printed < args.max_rows:
                    # recover x from the prompt: field tokens are N=2, X=3, T=4;
                    # digits follow each marker until the next marker/pad.
                    ids = input_ids[row].tolist()
                    true_str = _digits_to_int(true_tokens)
                    pred_str = _digits_to_int(pred_tokens)
                    tag = "OK  " if correct else "WRONG"
                    print(
                        f"{tag} input_ids={ids} true={true_str:>4s} pred={pred_str:>4s}"
                    )
                    rows_printed += 1

    print(f"\n{args.split}: {n_correct}/{n_total} = {100*n_correct/max(1,n_total):.2f}% exact-match")


if __name__ == "__main__":
    main()

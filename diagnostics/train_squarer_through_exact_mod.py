from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
from pathlib import Path

import torch


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("binary_square_submission", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def records(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if json.loads(line)["time_steps"] == 1]


def exact_residue_distribution(bit_logits, modulus, maximum_modulus):
    """Exact distribution of a Bernoulli-bit integer modulo each row's N."""
    batch = bit_logits.shape[0]
    residues = torch.arange(maximum_modulus, device=bit_logits.device)
    valid = residues[None, :] < modulus[:, None]
    distribution = bit_logits.new_zeros(batch, maximum_modulus, dtype=torch.float32)
    distribution[:, 0] = 1.0
    probabilities = bit_logits.float().sigmoid()
    for bit_index in range(bit_logits.shape[1]):
        weight = torch.remainder(1 << bit_index, modulus)
        source = torch.remainder(residues[None, :] - weight[:, None], modulus[:, None])
        shifted = distribution.gather(1, source)
        probability = probabilities[:, bit_index : bit_index + 1]
        distribution = ((1.0 - probability) * distribution + probability * shifted) * valid
    return distribution


@torch.no_grad()
def evaluate(module, model, items, device):
    model.eval()
    square_correct = mod_correct = bit_correct = bit_total = 0
    for start in range(0, len(items), 512):
        chunk = items[start : start + 512]
        x = torch.tensor([item["x"] for item in chunk], device=device)
        n = torch.tensor([item["modulus"] for item in chunk], device=device)
        target_mod = torch.tensor([item["result"] for item in chunk], device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(module._bits(x).float())
        predicted_bits = logits >= 0
        true_square = x * x
        shifts = torch.arange(module.SQUARE_BITS, device=device)
        true_bits = torch.bitwise_and(torch.bitwise_right_shift(true_square[:, None], shifts), 1).bool()
        powers = torch.bitwise_left_shift(torch.ones(module.SQUARE_BITS, dtype=torch.long, device=device), shifts)
        predicted_value = (predicted_bits.long() * powers).sum(1)
        predicted_mod = torch.remainder(predicted_value, n)
        square_correct += int((predicted_bits == true_bits).all(1).sum())
        mod_correct += int((predicted_mod == target_mod).sum())
        bit_correct += int((predicted_bits == true_bits).sum())
        bit_total += true_bits.numel()
    return {
        "examples": len(items),
        "square_exact": square_correct / len(items),
        "square_bit": bit_correct / bit_total,
        "mod_exact": mod_correct / len(items),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=74)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    module = load_module(args.submission)
    train = records(args.data_root / "train.jsonl")
    test = records(args.data_root / "test.jsonl")
    maximum_modulus = max(item["modulus"] for item in train + test) + 1
    model = module.BinarySquarer().cuda()
    matrix = [parameter for parameter in model.parameters() if parameter.ndim >= 2]
    scalar = [parameter for parameter in model.parameters() if parameter.ndim < 2]
    muon = module.FlattenedMuon(matrix)
    adam = torch.optim.AdamW(scalar, lr=3e-4, betas=(0.9, 0.95), weight_decay=1e-5)
    checkpoints = {100, 500, 1000, 3000, args.steps}
    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "metrics.jsonl"
    with metrics_path.open("w") as metrics:
        for step in range(1, args.steps + 1):
            chosen = random.choices(train, k=args.batch_size)
            x = torch.tensor([item["x"] for item in chosen], device="cuda")
            n = torch.tensor([item["modulus"] for item in chosen], device="cuda")
            target = torch.tensor([item["result"] for item in chosen], device="cuda")
            model.train()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(module._bits(x).float())
            distribution = exact_residue_distribution(logits, n, maximum_modulus)
            target_probability = distribution.gather(1, target[:, None]).squeeze(1).clamp_min(1e-12)
            loss = -target_probability.log().mean()
            muon.zero_grad(set_to_none=True)
            adam.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            muon.step()
            adam.step()
            progress = min(1.0, max(0.0, (step - 1000) / 4000))
            muon.param_groups[0]["lr"] = 0.002 + 0.018 * 0.5 * (1.0 + math.cos(math.pi * progress))
            if step in checkpoints:
                train_metrics = evaluate(module, model, train, "cuda")
                test_metrics = evaluate(module, model, test, "cuda")
                row = {"step": step, "loss": float(loss), "train": train_metrics, "test": test_metrics}
                print(json.dumps(row), flush=True)
                metrics.write(json.dumps(row) + "\n")
                metrics.flush()
                torch.save({f"squarer.{key}": value.cpu() for key, value in model.state_dict().items()}, args.output / f"step_{step:06d}.pt")


if __name__ == "__main__":
    main()

"""One-variable Neural GPU ablations on the frozen binary 0..99 task."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim._muon import _zeropower_via_newtonschulz

from train_transformer_multiplication_factorial import carry_count, split_examples

BITS, OUTPUT_BITS = 7, 14


def bits(value, width):
    return [(value >> position) & 1 for position in range(width)]


def rows(examples, input_bits=BITS):
    return [(bits(a, input_bits), bits(b, input_bits), bits(a * b, 2 * input_bits), a, b)
            for a, b in examples]


def sampled_pair_split(input_bits, train_size, validation_size, audit_size, seed):
    """Deterministic non-competition unordered-pair sample without overlap."""
    rng = random.Random(seed + 40_000)
    limit, needed = 1 << input_bits, train_size + validation_size + audit_size
    pairs = set()
    while len(pairs) < needed:
        a, b = rng.randrange(limit), rng.randrange(limit)
        pairs.add((min(a, b), max(a, b)))
    pairs = list(pairs)
    rng.shuffle(pairs)
    return (pairs[:train_size], pairs[train_size:train_size + validation_size],
            pairs[train_size + validation_size:])


def square_value_split(input_bits, train_size, validation_size, audit_size, seed):
    """Deterministic disjoint-x split for direct fixed-width squaring."""
    limit, needed = 1 << input_bits, train_size + validation_size + audit_size
    if needed > limit:
        raise ValueError(f"square split requests {needed} values but only {limit} exist")
    values = list(range(limit))
    random.Random(seed + 50_000).shuffle(values)
    pairs = [(value, value) for value in values[:needed]]
    return (pairs[:train_size], pairs[train_size:train_size + validation_size],
            pairs[train_size + validation_size:])


def binary_carry_count(a, b, width):
    left, right, carry, active = bits(a, width), bits(b, width), 0, 0
    for column in range(2 * width - 1):
        total = carry + sum(left[i] * right[column - i] for i in range(width)
                            if 0 <= column - i < width)
        carry = total // 2
        active += int(carry > 0)
    return active


def hard_sigmoid(value):
    return torch.clamp((value + 1.0) / 2.0, 0.0, 1.0)


class Cell(nn.Module):
    def __init__(self, channels, diagonal=False, hard=False):
        super().__init__()
        self.channels, self.diagonal, self.hard = channels, diagonal, hard
        self.update = nn.Conv2d(channels, channels, 3, padding=1)
        self.reset = nn.Conv2d(channels, channels, 3, padding=1)
        self.candidate = nn.Conv2d(channels, channels, 3, padding=1)

    def transport(self, hidden):
        if not self.diagonal:
            return hidden
        cut1, cut2 = self.channels // 3, 2 * self.channels // 3
        left, stay, right = hidden[:, :cut1], hidden[:, cut1:cut2], hidden[:, cut2:]
        return torch.cat((F.pad(left[..., 1:], (0, 1)), stay,
                          F.pad(right[..., :-1], (1, 0))), dim=1)

    def forward(self, hidden, dropout_mask=None):
        state = self.transport(hidden)
        update_pre, reset_pre = self.update(state), self.reset(state)
        activation = hard_sigmoid if self.hard else torch.sigmoid
        update, reset = activation(update_pre), activation(reset_pre)
        candidate_pre = self.candidate(reset * state)
        candidate = F.hardtanh(candidate_pre) if self.hard else torch.tanh(candidate_pre)
        if dropout_mask is not None:
            candidate = candidate * dropout_mask
        output = (1 - update) * state + update * candidate
        saturation = output.new_zeros(())
        if self.hard:
            saturation = sum(F.relu(value.abs() - 0.9).mean()
                             for value in (update_pre, reset_pre, candidate_pre))
        return output, saturation


class Model(nn.Module):
    def __init__(self, variant, channels, steps=14):
        super().__init__()
        self.variant, self.channels, self.steps = variant, channels, steps
        self.embedding = nn.Embedding(2, channels)
        self.left_marker = nn.Parameter(torch.randn(channels) * 0.02)
        self.right_marker = nn.Parameter(torch.randn(channels) * 0.02)
        copies = 6 if variant == "sharing_relaxation" else 4 if variant == "microprogram" else 1
        self.cells = nn.ModuleList([Cell(channels, variant in ("diagonal", "muon_dropout_diagonal"), variant == "hard")
                                    for _ in range(copies)])
        self.readout = nn.Conv1d(channels, 1, 1)
        self.memory_gate = nn.Conv2d(channels * 2, channels, 1) if variant == "sparse_memory" else None

    def forward(self, left, right, dropout=0.0):
        batch, width = left.shape
        hidden = torch.zeros(batch, self.channels, 4, 2 * width, device=left.device)
        hidden[:, :, 0, :width] = self.embedding(left).transpose(1, 2) + self.left_marker[None, :, None]
        hidden[:, :, 1, :width] = self.embedding(right).transpose(1, 2) + self.right_marker[None, :, None]
        mask = None
        if self.training and dropout:
            keep = 1.0 - dropout
            mask = torch.empty(batch, self.channels, 1, 1, device=left.device).bernoulli_(keep) / keep
        history, saturation = [], hidden.new_zeros(())
        for step in range(self.steps):
            if self.memory_gate is not None and step >= 4:
                past = history[-4]
                gate = torch.sigmoid(self.memory_gate(torch.cat((hidden, past), dim=1)))
                hidden = hidden + gate * past
            history.append(hidden)
            hidden, penalty = self.cells[step % len(self.cells)](hidden, mask)
            saturation = saturation + penalty
        return self.readout(hidden[:, :, 0]).squeeze(1), saturation / self.steps

    def sharing_cost(self):
        if self.variant != "sharing_relaxation":
            return self.left_marker.new_zeros(())
        cost, count = self.left_marker.new_zeros(()), 0
        named = [dict(cell.named_parameters()) for cell in self.cells]
        for name in named[0]:
            mean = torch.stack([parameters[name] for parameters in named]).mean(0)
            for parameters in named:
                cost = cost + (parameters[name] - mean).pow(2).mean()
                count += 1
        return cost / count


def tensors(items, device):
    return (torch.tensor([item[0] for item in items], device=device),
            torch.tensor([item[1] for item in items], device=device),
            torch.tensor([item[2] for item in items], dtype=torch.float32, device=device))


def buckets(counts):
    return {str(key): {"correct": value[0], "total": value[1], "accuracy": value[0] / value[1]}
            for key, value in sorted(counts.items(), key=lambda item: str(item[0]))}


@torch.no_grad()
def evaluate(model, data, device, input_bits=BITS):
    model.eval()
    exact = bit_correct = bit_total = 0
    output_bits = 2 * input_bits
    positions = [0] * output_bits
    carry = defaultdict(lambda: [0, 0])
    lengths = defaultdict(lambda: [0, 0])
    for start in range(0, len(data), 512):
        chunk = data[start:start + 512]
        left, right, target = tensors(chunk, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = model(left, right)[0].gt(0).long()
        matches = prediction == target.long()
        rows_exact = matches.all(-1)
        exact += int(rows_exact.sum())
        bit_correct += int(matches.sum())
        bit_total += matches.numel()
        for position in range(output_bits):
            positions[position] += int(matches[:, position].sum())
        for item, correct in zip(chunk, rows_exact.tolist()):
            a, b = item[3], item[4]
            carry_key = carry_count(a, b) if input_bits == 7 else binary_carry_count(a, b, input_bits)
            length_key = len(str(a * b)) if input_bits == 7 else (a * b).bit_length()
            for table, key in ((carry, carry_key), (lengths, length_key)):
                table[key][0] += int(correct)
                table[key][1] += 1
    total = len(data)
    return {"exact": exact / total, "bit_accuracy": bit_correct / bit_total,
            "bit_accuracy_lsd_first": [value / total for value in positions],
            "carry": buckets(carry), "product_length": buckets(lengths), "examples": total}


class ConvMuon(Optimizer):
    """PyTorch Muon update extended to flatten convolutional kernels."""
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=1e-5):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.reshape(parameter.shape[0], -1)
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(gradient)
                buffer = state["momentum_buffer"]
                buffer.lerp_(gradient, 1 - group["momentum"])
                update = gradient.lerp(buffer, group["momentum"])
                update = _zeropower_via_newtonschulz(
                    update, (3.4445, -4.775, 2.0315), 5, 1e-7
                )
                adjusted_lr = group["lr"] * max(1.0, gradient.shape[0] / gradient.shape[1]) ** 0.5
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(update.reshape_as(parameter), alpha=-adjusted_lr)


def make_optimizers(model, variant):
    if variant != "muon":
        return [torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)]
    matrix = [parameter for parameter in model.parameters() if parameter.ndim >= 2]
    scalar = [parameter for parameter in model.parameters() if parameter.ndim < 2]
    return [ConvMuon(matrix, lr=0.02, weight_decay=1e-5),
            torch.optim.AdamW(scalar, lr=3e-4, weight_decay=1e-5)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["baseline", "diagonal", "dropout", "hard",
                        "gradient_noise", "wide", "sharing_relaxation", "muon",
                        "sparse_memory", "microprogram", "muon_decay", "muon_dropout",
                        "muon_dropout_diagonal"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--input-bits", type=int, default=7)
    parser.add_argument("--task", choices=["multiply", "square"], default="multiply")
    parser.add_argument("--train-size", type=int, default=200_000)
    parser.add_argument("--validation-size", type=int, default=10_000)
    parser.add_argument("--audit-size", type=int, default=10_000)
    parser.add_argument("--train-eval-size", type=int, default=0,
                        help="Use a fixed train prefix for progress only; zero evaluates all train rows.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if args.task == "square":
        train_examples, validation_examples, audit_examples = square_value_split(
            args.input_bits, args.train_size, args.validation_size, args.audit_size, args.seed)
    elif args.input_bits == 7:
        train_examples, test_examples = split_examples(args.seed)
        test_rng = random.Random(args.seed + 20_000)
        test_rng.shuffle(test_examples)
        validation_examples, audit_examples = test_examples[:1003], test_examples[1003:]
    else:
        train_examples, validation_examples, audit_examples = sampled_pair_split(
            args.input_bits, args.train_size, args.validation_size, args.audit_size, args.seed)
    train = rows(train_examples, args.input_bits)
    validation = rows(validation_examples, args.input_bits)
    audit = rows(audit_examples, args.input_bits)
    train_monitor = train[:args.train_eval_size] if args.train_eval_size else train
    channels = 192 if args.variant == "wide" else 128
    model = Model(args.variant, channels, steps=2 * args.input_bits).to(args.device)
    if args.compile:
        model = torch.compile(model)
    optimizer_variant = "muon" if args.variant in ("muon_decay", "muon_dropout",
                                                      "muon_dropout_diagonal") else args.variant
    optimizers = make_optimizers(model, optimizer_variant)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    curve, best_validation, best_state, best_step = [], -1.0, None, 0
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        indices = torch.randint(len(train), (512,), generator=generator).tolist()
        left, right, target = tensors([train[index] for index in indices], args.device)
        model.train()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, saturation = model(left, right, dropout=0.09 if args.variant in
                                       ("dropout", "muon_dropout", "muon_dropout_diagonal") else 0.0)
            loss = F.binary_cross_entropy_with_logits(logits, target)
        if args.variant == "hard":
            loss = loss + 1e-3 * saturation
        if args.variant == "sharing_relaxation":
            loss = loss + min(1.0, step / 15_000) * model.sharing_cost()
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.variant == "gradient_noise":
            std = 0.03 / ((1 + step) ** 0.55)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.add_(torch.randn_like(parameter.grad), alpha=std)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        for optimizer in optimizers:
            optimizer.step()
        if args.variant in ("muon_decay", "muon_dropout", "muon_dropout_diagonal"):
            progress = min(1.0, max(0.0, (step - 1000) / 4000))
            muon_lr = 0.002 + 0.018 * 0.5 * (1 + math.cos(progress * math.pi))
            optimizers[0].param_groups[0]["lr"] = muon_lr
        if step == 1 or step % args.eval_every == 0:
            train_metrics = evaluate(model, train_monitor, args.device, args.input_bits)
            validation_metrics = evaluate(model, validation, args.device, args.input_bits)
            record = {"step": step, "examples": step * 512,
                      "elapsed_seconds": time.perf_counter() - started,
                      "loss": float(loss.detach()),
                      "train_exact": train_metrics["exact"],
                      "validation_exact": validation_metrics["exact"]}
            curve.append(record)
            if record["validation_exact"] > best_validation:
                best_validation, best_step = record["validation_exact"], step
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, args.out / "model_best.pt")
            print(json.dumps({"type": "progress", **record}), flush=True)
    final = {"train": evaluate(model, train, args.device, args.input_bits),
             "validation": evaluate(model, validation, args.device, args.input_bits)}
    final_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    selected = {"train": evaluate(model, train, args.device, args.input_bits),
                "validation": evaluate(model, validation, args.device, args.input_bits),
                "audit": evaluate(model, audit, args.device, args.input_bits)}
    report = {"task": args.task, "variant": args.variant, "seed": args.seed, "steps": args.steps,
              "input_bits": args.input_bits, "recurrent_updates": 2 * args.input_bits,
              "parameters": sum(parameter.numel() for parameter in model.parameters()),
              "split": {"train": len(train), "validation": len(validation), "audit": len(audit)},
              "best_step": best_step, "curve": curve, "selected": selected, "final": final,
              "elapsed_seconds": time.perf_counter() - started}
    (args.out / "eval_report.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save(best_state, args.out / "model_best.pt")
    torch.save(final_state, args.out / "model_final.pt")
    (args.out / "source.py").write_text(Path(__file__).read_text())
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

"""Kaiser-Sutskever Neural GPU on fixed 0..99 decimal multiplication.

Architecture follows arXiv:1511.08228: a [w,n,m] mental image, input embedded
in its first width column, two 3x3 CGRU layers repeated n times, hard-cutoff
sigmoid gates, recurrent dropout, and six relaxed recurrent parameter sets.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from train_transformer_multiplication_factorial import carry_count, split_examples


PAD, MUL = 10, 11
VOCAB, IMAGE_WIDTH, MAPS, SEQUENCE_LENGTH = 12, 4, 24, 5
RELAX_COPIES, CGRU_LAYERS = 6, 2


def digits(value, width=2):
    return [int(character) for character in f"{value:0{width}d}"][::-1]


def rows(examples):
    return [(digits(a) + [MUL] + digits(b), digits(a * b, 4) + [PAD], a, b) for a, b in examples]


def hard_sigmoid(value):
    return (1.2 * torch.sigmoid(value) - .1).clamp(0, 1)


class CGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.update = nn.Conv2d(MAPS, MAPS, 3, padding=1)
        self.reset = nn.Conv2d(MAPS, MAPS, 3, padding=1)
        self.candidate = nn.Conv2d(MAPS, MAPS, 3, padding=1)

    def forward(self, state):
        update = hard_sigmoid(self.update(state))
        reset = hard_sigmoid(self.reset(state))
        candidate = torch.tanh(self.candidate(reset * state))
        return update * state + (1 - update) * candidate


class PaperNeuralGPU(nn.Module):
    def __init__(self, dropout=.09):
        super().__init__()
        self.dropout = dropout
        self.embedding = nn.Embedding(VOCAB, MAPS)
        self.relaxed = nn.ModuleList([
            nn.ModuleList([CGRU() for _ in range(CGRU_LAYERS)])
            for _ in range(RELAX_COPIES)
        ])
        self.output = nn.Linear(MAPS, VOCAB)

    def forward(self, input_ids):
        batch, length = input_ids.shape
        state = torch.zeros(batch, MAPS, IMAGE_WIDTH, length, device=input_ids.device)
        state[:, :, 0, :] = self.embedding(input_ids).transpose(1, 2)
        for time_step in range(length):
            cells = self.relaxed[time_step % RELAX_COPIES]
            for cell in cells:
                state = cell(state)
            if self.training and self.dropout:
                state = F.dropout(state, p=self.dropout, training=True)
        return self.output(state[:, :, 0, :].transpose(1, 2))

    def relaxation_cost(self):
        cost = torch.zeros((), device=self.embedding.weight.device)
        parameter_groups = list(zip(*(list(copy.parameters()) for copy in self.relaxed)))
        for parameters in parameter_groups:
            stacked = torch.stack(parameters)
            mean = stacked.mean(0)
            cost = cost + ((stacked - mean) ** 2).mean()
        return cost


def tensors(items, device):
    return (
        torch.tensor([row[0] for row in items], device=device),
        torch.tensor([row[1] for row in items], device=device),
    )


def bucket_report(counts):
    return {str(key): {"correct": value[0], "total": value[1], "accuracy": value[0] / value[1]}
            for key, value in sorted(counts.items(), key=lambda item: str(item[0]))}


@torch.no_grad()
def evaluate(model, data, device, batch_size=1024):
    model.eval()
    exact = total = 0
    digit_correct = [0] * 4
    operand_buckets = defaultdict(lambda: [0, 0])
    product_buckets = defaultdict(lambda: [0, 0])
    carry_buckets = defaultdict(lambda: [0, 0])
    predictions = {}
    for start in range(0, len(data), batch_size):
        chunk = data[start:start + batch_size]
        source, target = tensors(chunk, device)
        prediction = model(source).argmax(-1)[:, :4]
        matches = prediction == target[:, :4]
        for position in range(4):
            digit_correct[position] += int(matches[:, position].sum())
        for row, predicted, correct in zip(chunk, prediction.tolist(), matches.all(-1).tolist()):
            a, b = row[2], row[3]
            value = sum(digit * 10**position for position, digit in enumerate(predicted))
            predictions[(a, b)] = value
            exact += int(correct)
            total += 1
            for buckets, key in (
                (operand_buckets, f"{len(str(a))}x{len(str(b))}"),
                (product_buckets, len(str(a * b))),
                (carry_buckets, carry_count(a, b)),
            ):
                buckets[key][0] += int(correct)
                buckets[key][1] += 1
    comm_correct = comm_total = 0
    for (a, b), value in predictions.items():
        if a <= b and (b, a) in predictions:
            comm_correct += int(value == predictions[(b, a)])
            comm_total += 1
    return {
        "exact_numerical_accuracy": exact / total,
        "digit_accuracy_lsd_first": [value / total for value in digit_correct],
        "by_operand_digit_lengths": bucket_report(operand_buckets),
        "by_product_digit_length": bucket_report(product_buckets),
        "by_carry_columns": bucket_report(carry_buckets),
        "commutativity_consistency": comm_correct / comm_total,
        "examples": total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_examples, test_examples = split_examples(args.seed)
    train, test = rows(train_examples), rows(test_examples)
    model = PaperNeuralGPU().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    curve = []
    for step in range(1, args.steps + 1):
        indices = torch.randint(len(train), (256,), generator=generator).tolist()
        sample = [train[index] for index in indices]
        source, target = tensors(sample, args.device)
        model.train()
        logits = model(source)
        task_loss = F.cross_entropy(logits.reshape(-1, VOCAB), target.reshape(-1))
        pull = .1 * min(step / args.steps, 1.0)
        relaxation_loss = model.relaxation_cost()
        loss = task_loss + pull * relaxation_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 400 == 0:
            record = {"step": step, "task_loss": float(task_loss.detach()),
                      "relaxation_loss": float(relaxation_loss.detach()), "pull": pull}
            curve.append(record)
            print(json.dumps({"type": "progress", **record}), flush=True)
    report = {
        "classification": "RESEARCH ONLY - paper Neural GPU decimal multiplication",
        "paper": "Kaiser and Sutskever 2015, arXiv:1511.08228",
        "architecture": {"image_width": IMAGE_WIDTH, "maps": MAPS, "kernel": 3,
                         "cgru_layers": CGRU_LAYERS, "relaxation_copies": RELAX_COPIES,
                         "recurrent_steps": SEQUENCE_LENGTH, "dropout": .09},
        "representation": "LSD-first decimal: aa MUL bb; four LSD product digits plus PAD",
        "split": "same 80/20 unordered-pair groups as prior baselines",
        "seed": args.seed,
        "steps": args.steps,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "curve": curve,
        "train": evaluate(model, train, args.device),
        "test": evaluate(model, test, args.device),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "eval_report.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save(model.state_dict(), args.out / "model.pt")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

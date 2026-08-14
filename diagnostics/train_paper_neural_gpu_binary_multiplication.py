"""Paper Neural GPU binary multiplication on the shared 0..99 numeric split."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from train_paper_neural_gpu_multiplication import CGRU
from train_transformer_multiplication_factorial import carry_count, split_examples


PAD, MUL, VOCAB = 2, 3, 4
BITS, LENGTH, MAPS, IMAGE_WIDTH, RELAX_COPIES, CGRU_LAYERS = 7, 15, 24, 4, 6, 2


def bits(value, width):
    return [(value >> position) & 1 for position in range(width)]


def rows(examples):
    return [(bits(a, BITS) + [MUL] + bits(b, BITS), bits(a * b, 2 * BITS) + [PAD], a, b)
            for a, b in examples]


class BinaryPaperNeuralGPU(nn.Module):
    def __init__(self, dropout=.09):
        super().__init__()
        self.dropout = dropout
        self.embedding = nn.Embedding(VOCAB, MAPS)
        self.relaxed = nn.ModuleList([
            nn.ModuleList([CGRU() for _ in range(CGRU_LAYERS)]) for _ in range(RELAX_COPIES)
        ])
        self.output = nn.Linear(MAPS, VOCAB)

    def forward(self, input_ids):
        batch, length = input_ids.shape
        state = torch.zeros(batch, MAPS, IMAGE_WIDTH, length, device=input_ids.device)
        state[:, :, 0, :] = self.embedding(input_ids).transpose(1, 2)
        for time_step in range(length):
            for cell in self.relaxed[time_step % RELAX_COPIES]:
                state = cell(state)
            if self.training:
                state = F.dropout(state, p=self.dropout, training=True)
        return self.output(state[:, :, 0, :].transpose(1, 2))

    def relaxation_cost(self):
        cost = torch.zeros((), device=self.embedding.weight.device)
        for parameters in zip(*(list(copy.parameters()) for copy in self.relaxed)):
            stacked = torch.stack(parameters)
            cost = cost + ((stacked - stacked.mean(0)) ** 2).mean()
        return cost


def tensors(items, device):
    return (torch.tensor([row[0] for row in items], device=device),
            torch.tensor([row[1] for row in items], device=device))


def bucket_report(counts):
    return {str(k): {"correct": v[0], "total": v[1], "accuracy": v[0] / v[1]}
            for k, v in sorted(counts.items(), key=lambda item: str(item[0]))}


@torch.no_grad()
def evaluate(model, data, device, batch_size=512):
    model.eval()
    exact = total = 0
    bit_correct = [0] * (2 * BITS)
    product_buckets = defaultdict(lambda: [0, 0])
    carry_buckets = defaultdict(lambda: [0, 0])
    for start in range(0, len(data), batch_size):
        chunk = data[start:start + batch_size]
        source, target = tensors(chunk, device)
        prediction = model(source).argmax(-1)[:, :2 * BITS]
        matches = prediction == target[:, :2 * BITS]
        for position in range(2 * BITS):
            bit_correct[position] += int(matches[:, position].sum())
        for row, correct in zip(chunk, matches.all(-1).tolist()):
            a, b = row[2], row[3]
            exact += int(correct); total += 1
            for buckets, key in ((product_buckets, len(str(a * b))), (carry_buckets, carry_count(a, b))):
                buckets[key][0] += int(correct); buckets[key][1] += 1
    return {"exact_numerical_accuracy": exact / total,
            "bit_accuracy_lsd_first": [value / total for value in bit_correct],
            "by_product_digit_length": bucket_report(product_buckets),
            "by_decimal_carry_columns": bucket_report(carry_buckets), "examples": total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    train_examples, test_examples = split_examples(args.seed)
    train, test = rows(train_examples), rows(test_examples)
    model = BinaryPaperNeuralGPU().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    curve = []
    for step in range(1, args.steps + 1):
        indices = torch.randint(len(train), (256,), generator=generator).tolist()
        source, target = tensors([train[index] for index in indices], args.device)
        model.train(); logits = model(source)
        task_loss = F.cross_entropy(logits.reshape(-1, VOCAB), target.reshape(-1))
        pull = .1 * step / args.steps; relaxation_loss = model.relaxation_cost()
        loss = task_loss + pull * relaxation_loss
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step == 1 or step % 400 == 0:
            record = {"step": step, "task_loss": float(task_loss.detach()),
                      "relaxation_loss": float(relaxation_loss.detach()), "pull": pull}
            curve.append(record); print(json.dumps({"type": "progress", **record}), flush=True)
    report = {"classification": "RESEARCH ONLY - paper Neural GPU binary multiplication",
              "paper": "Kaiser and Sutskever 2015, arXiv:1511.08228",
              "architecture": {"image_width": 4, "maps": 24, "kernel": 3, "cgru_layers": 2,
                               "relaxation_copies": 6, "recurrent_steps": LENGTH, "dropout": .09},
              "representation": "two padded 7-bit LSD-first operands, MUL, 14 product bits, PAD",
              "split": "same numeric unordered-pair split as decimal arms", "seed": args.seed,
              "steps": args.steps, "parameters": sum(p.numel() for p in model.parameters()),
              "curve": curve, "train": evaluate(model, train, args.device),
              "test": evaluate(model, test, args.device)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "eval_report.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save(model.state_dict(), args.out / "model.pt"); print(json.dumps(report), flush=True)


if __name__ == "__main__": main()

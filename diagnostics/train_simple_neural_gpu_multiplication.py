"""Minimal convolutional Neural GPU for fixed-width 0..99 multiplication."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from train_serial_gru_multiplication import evaluate, rows, tensors
from train_transformer_multiplication_factorial import split_examples


WIDTH, HIDDEN = 4, 64


class ConvGRUCell(nn.Module):
    def __init__(self):
        super().__init__()
        self.left = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.self_ = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.right = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.norm = nn.LayerNorm(HIDDEN)
        self.gru = nn.GRUCell(HIDDEN, HIDDEN)

    def forward(self, state, immutable):
        zero = torch.zeros_like(state[:, :1])
        left = torch.cat((zero, state[:, :-1]), dim=1)
        right = torch.cat((state[:, 1:], zero), dim=1)
        message = self.norm(self.left(left) + self.self_(state) + self.right(right) + immutable)
        return self.gru(message.reshape(-1, HIDDEN), state.reshape(-1, HIDDEN)).reshape_as(state)


class SimpleNeuralGPU(nn.Module):
    def __init__(self, recurrent_steps=8):
        super().__init__()
        self.recurrent_steps = recurrent_steps
        self.digit = nn.Embedding(10, HIDDEN)
        self.place = nn.Embedding(WIDTH, HIDDEN)
        self.inject = nn.Sequential(nn.Linear(2 * HIDDEN, HIDDEN), nn.LayerNorm(HIDDEN), nn.GELU())
        self.boundary = nn.Parameter(torch.randn(2, HIDDEN) * .02)
        self.cell = ConvGRUCell()
        self.output_norm = nn.LayerNorm(HIDDEN)
        self.head = nn.Linear(HIDDEN, 10)

    def forward(self, left, right):
        positions = torch.arange(WIDTH, device=left.device)
        place = self.place(positions)[None]
        immutable = self.inject(torch.cat((self.digit(left) + place, self.digit(right) + place), dim=-1))
        immutable = immutable.clone()
        immutable[:, 0] += self.boundary[0]
        immutable[:, -1] += self.boundary[1]
        state = torch.zeros_like(immutable)
        for _ in range(self.recurrent_steps):
            state = self.cell(state, immutable)
        return self.head(self.output_norm(state))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--recurrent-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_examples, test_examples = split_examples(args.seed)
    train, test = rows(train_examples), rows(test_examples)
    model = SimpleNeuralGPU(args.recurrent_steps).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(.9, .95), weight_decay=.01)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    curve = []
    for step in range(1, args.steps + 1):
        indices = torch.randint(len(train), (256,), generator=generator).tolist()
        sample = [train[index] for index in indices]
        left, right, target = tensors(sample, args.device)
        model.train()
        logits = model(left, right)
        loss = F.cross_entropy(logits.reshape(-1, 10), target.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 400 == 0:
            record = {"step": step, "loss": float(loss.detach())}
            curve.append(record)
            print(json.dumps({"type": "progress", **record}), flush=True)
    report = {
        "classification": "RESEARCH ONLY - fixed-width minimal Neural GPU multiplication",
        "architecture": "one 64-wide tape, tied local ConvGRU, both operands injected, final digit decode",
        "split": "same 80/20 unordered-pair groups as Transformer factorial",
        "seed": args.seed,
        "steps": args.steps,
        "recurrent_steps": args.recurrent_steps,
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

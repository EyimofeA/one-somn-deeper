"""Final-label-only T=1 test of phase-factorized information flow.

The matched arms differ only in whether N is visible during the square phase.
Arithmetic is used solely to generate/evaluate labels, never as model state or
an intermediate supervision target.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from train_t1_representation import REPRESENTATIONS, batch, decode, encode, moduli, rows


class PhaseSquareReduce(nn.Module):
    def __init__(self, arm: str, width: int = 4, vocab: int = 10, d: int = 128,
                 square_steps: int = 4, reduce_steps: int = 4):
        super().__init__()
        self.arm = arm
        self.width = width
        self.vocab = vocab
        self.d = d
        self.square_steps = square_steps
        self.reduce_steps = reduce_steps
        self.token = nn.Embedding(vocab, 32)
        self.place = nn.Embedding(width, 32)
        self.n_local = nn.Sequential(nn.Linear(32, d), nn.LayerNorm(d), nn.GELU())
        self.x_local = nn.Sequential(nn.Linear(32, d), nn.LayerNorm(d), nn.GELU())
        self.init = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d), nn.GELU())
        self.null_square_context = nn.Parameter(torch.zeros(1, width, d))
        self.square_mix = nn.Linear(3 * d, d)
        self.square_cell = nn.GRUCell(2 * d, d)
        self.reduce_mix = nn.Linear(3 * d, d)
        self.reduce_cell = nn.GRUCell(2 * d, d)
        self.decoder = nn.Sequential(nn.Linear(d + 32, d), nn.GELU(), nn.Linear(d, vocab))

    @staticmethod
    def neighbors(h):
        zeros = torch.zeros_like(h[:, :1])
        return torch.cat((zeros, h[:, :-1]), 1), torch.cat((h[:, 1:], zeros), 1)

    def recurrent_step(self, h, context, mixer, cell):
        left, right = self.neighbors(h)
        local = mixer(torch.cat((left, h, right), -1))
        updated = cell(torch.cat((local, context), -1).reshape(-1, 2 * self.d),
                       h.reshape(-1, self.d))
        return updated.reshape_as(h)

    def forward(self, n, x):
        pos = torch.arange(self.width, device=n.device)
        pe = self.place(pos)
        n_state = self.n_local(self.token(n) + pe)
        h = self.init(self.x_local(self.token(x) + pe))
        square_context = n_state if self.arm == "entangled" else self.null_square_context.expand(h.shape[0], -1, -1)
        for _ in range(self.square_steps):
            h = self.recurrent_step(h, square_context, self.square_mix, self.square_cell)
        for _ in range(self.reduce_steps):
            h = self.recurrent_step(h, n_state, self.reduce_mix, self.reduce_cell)
        return self.decoder(torch.cat((h, pe[None].expand(h.shape[0], -1, -1)), -1))


@torch.no_grad()
def evaluate(model, data, device):
    model.eval()
    exact = token_exact = total = 0
    started = time.perf_counter()
    for start in range(0, len(data), 512):
        sample = data[start:start + 512]
        n, x, y, target = batch(sample, len(sample), 0, device)
        pred = model(n, x).argmax(-1)
        token_exact += int((pred == y).sum())
        exact += sum(decode(row, "decimal") == int(value) for row, value in zip(pred.tolist(), target.tolist()))
        total += len(sample)
    elapsed = time.perf_counter() - started
    return {
        "exact": exact / total,
        "token_accuracy": token_exact / (total * 4),
        "examples": total,
        "seconds": elapsed,
        "examples_per_second": total / max(elapsed, 1e-9),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("factored", "entangled"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--square-steps", type=int, default=4)
    parser.add_argument("--reduce-steps", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    ms = moduli()
    train_ms, unseen_ms = ms[:18], ms[18:]
    train = rows(train_ms, "decimal", True)
    held = rows(train_ms, "decimal", False)
    unseen = rows(unseen_ms, "decimal", True) + rows(unseen_ms, "decimal", False)
    model = PhaseSquareReduce(args.arm, square_steps=args.square_steps,
                              reduce_steps=args.reduce_steps).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, betas=(.9, .95), weight_decay=.05)
    started = time.monotonic()
    step = 0
    curve = []
    while time.monotonic() - started < args.seconds:
        n, x, y, _ = batch(train, 512, step, device)
        model.train()
        logits = model(n, x)
        loss = F.cross_entropy(logits.reshape(-1, 10), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        step += 1
        if step == 1 or step % 100 == 0:
            curve.append({
                "step": step,
                "seconds": round(time.monotonic() - started, 2),
                "loss": float(loss.detach()),
                "last_batch_exact": float((logits.argmax(-1) == y).all(-1).float().mean()),
            })
    elapsed = time.monotonic() - started
    result = {
        "arm": args.arm,
        "seed": args.seed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "square_steps": args.square_steps,
        "reduce_steps": args.reduce_steps,
        "steps": step,
        "seconds": elapsed,
        "steps_per_second": step / max(elapsed, 1e-9),
        "curve": curve,
        "train": evaluate(model, train, device),
        "held_out_x": evaluate(model, held, device),
        "unseen_N": evaluate(model, unseen, device),
    }
    report = {
        "classification": "RESEARCH ONLY — T=1 final-label phase information-flow test",
        "one_variable": "N is hidden from versus exposed to the square phase",
        "intermediate_supervision": False,
        "train_moduli": train_ms,
        "unseen_moduli": unseen_ms,
        "result": result,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "eval_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

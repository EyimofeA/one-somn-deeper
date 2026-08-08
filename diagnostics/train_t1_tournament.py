"""T=1-only state-topology and discrete-refinement tournament.

Research-only diagnostic. The diffusion/refinement branch corrupts target
tokens during training and must pass a separate competition-legality review.
The model forwards contain no arithmetic; exact arithmetic is used only by the
synthetic data generator/evaluator.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from train_structured_latent_vdf import StateModel, W, digits, make_moduli, rows


class DiscreteRefiner(nn.Module):
    """Shared discrete digit-state refinement, initialized from all masks."""

    def __init__(self, d: int = 128):
        super().__init__(); self.d = d
        self.digit = nn.Embedding(10, 32); self.place = nn.Embedding(W, 32)
        self.state_token = nn.Embedding(11, d)  # 0..9 plus mask token 10
        self.condition = nn.Sequential(nn.Linear(64, d), nn.LayerNorm(d), nn.GELU(), nn.Linear(d, d))
        self.step = nn.GRUCell(2 * d, d)
        self.head = nn.Sequential(nn.Linear(d + 32, d), nn.GELU(), nn.Linear(d, 10))

    def forward(self, n, x, steps: int, initial=None):
        pos = torch.arange(W, device=n.device); pe = self.place(pos)
        cond = self.condition(torch.cat((self.digit(n) + pe, self.digit(x) + pe), -1))
        if initial is None:
            tokens = torch.full((n.shape[0], W), 10, dtype=torch.long, device=n.device)
            h = self.state_token(tokens)
        else:
            h = self.state_token(initial)
        outputs = []
        digit_values = self.state_token.weight[:10]
        for _ in range(steps):
            h = self.step(torch.cat((h, cond), -1).reshape(-1, 2 * self.d), h.reshape(-1, self.d)).reshape_as(h)
            logits = self.head(torch.cat((h, pe[None].expand(n.shape[0], -1, -1)), -1))
            outputs.append(logits)
            h = logits.softmax(-1) @ digit_values
        return outputs[-1], outputs


def split_t1(moduli):
    train = rows(moduli, True); held = rows(moduli, False)
    return train, held


def t1_batch(items, size, step, device):
    sample = [items[(step * size + i) % len(items)] for i in range(size)]
    n = torch.tensor([digits(row[0]) for row in sample], device=device)
    x = torch.tensor([digits(row[1]) for row in sample], device=device)
    y = torch.tensor([row[3][0] for row in sample], device=device)
    return n, x, y


@torch.no_grad()
def score(model, data, device, refiner_steps=None):
    model.eval(); correct = 0; total = 0; started = time.perf_counter()
    for start in range(0, len(data), 512):
        sample = data[start:start + 512]; n, x, y = t1_batch(sample, len(sample), 0, device)
        if refiner_steps is None:
            logits, _ = model(n, x, torch.ones(len(sample), dtype=torch.long, device=device), return_trace=True)
        else:
            logits, _ = model(n, x, refiner_steps)
        correct += int((logits.argmax(-1) == y).all(-1).sum()); total += len(sample)
    elapsed = time.perf_counter() - started
    return {"exact": correct / total, "examples": total, "seconds": elapsed, "examples_per_second": total / max(elapsed, 1e-9), "milliseconds_per_example": 1000 * elapsed / total}


def train_state(kind, train, held, unseen, seconds, device, seed):
    torch.manual_seed(seed); model = StateModel(kind).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, betas=(.9, .95), weight_decay=.05)
    started, step, history = time.monotonic(), 0, []
    while time.monotonic() - started < seconds:
        n, x, y = t1_batch(train, 512, step, device); model.train()
        logits, _ = model(n, x, torch.ones(512, dtype=torch.long, device=device), return_trace=True)
        loss = F.cross_entropy(logits.reshape(-1, 10), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); step += 1
        if step == 1 or step % 100 == 0:
            history.append({"step": step, "seconds": round(time.monotonic() - started, 2), "loss": float(loss.detach()), "train_exact": float((logits.argmax(-1) == y).all(-1).float().mean())})
    return {"kind": kind, "parameters": sum(p.numel() for p in model.parameters()), "steps": step, "seconds": time.monotonic() - started, "steps_per_second": step / max(time.monotonic() - started, 1e-9), "curve": history, "held_out_x": score(model, held, device), "unseen_N": score(model, unseen, device)}


def train_refiner(train, held, unseen, seconds, device, seed):
    torch.manual_seed(seed); model = DiscreteRefiner().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, betas=(.9, .95), weight_decay=.05)
    started, step, history = time.monotonic(), 0, []; mask_rate = .5
    while time.monotonic() - started < seconds:
        n, x, y = t1_batch(train, 512, step, device); target = y
        corrupt = torch.rand_like(target.float()) < mask_rate
        initial = torch.where(corrupt, torch.full_like(target, 10), target)
        model.train(); logits, _ = model(n, x, 4, initial)
        loss = F.cross_entropy(logits.reshape(-1, 10), target.reshape(-1))
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); step += 1
        if step == 1 or step % 100 == 0:
            history.append({"step": step, "seconds": round(time.monotonic() - started, 2), "loss": float(loss.detach()), "train_exact": float((logits.argmax(-1) == target).all(-1).float().mean())})
    return {"kind": "discrete_refiner", "parameters": sum(p.numel() for p in model.parameters()), "steps": step, "seconds": time.monotonic() - started, "steps_per_second": step / max(time.monotonic() - started, 1e-9), "mask_rate": mask_rate, "curve": history, "held_out_x_by_K": {str(k): score(model, held, device, k) for k in (1, 2, 4, 8)}, "unseen_N_by_K": {str(k): score(model, unseen, device, k) for k in (1, 2, 4, 8)}}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--kind", choices=("register", "global", "structured", "refiner"), required=True); parser.add_argument("--seconds", type=float, default=120); parser.add_argument("--seed", type=int, default=0); parser.add_argument("--device", default="cuda"); args = parser.parse_args()
    device = torch.device(args.device); moduli = make_moduli(0); train_moduli, unseen_moduli = moduli[:18], moduli[18:]
    train, held = split_t1(train_moduli); unseen_train, unseen_held = split_t1(unseen_moduli); unseen = unseen_train + unseen_held
    result = train_refiner(train, held, unseen, args.seconds, device, args.seed) if args.kind == "refiner" else train_state(args.kind, train, held, unseen, args.seconds, device, args.seed)
    report = {"classification": "RESEARCH ONLY — T=1 tournament", "train_moduli": train_moduli, "unseen_moduli": unseen_moduli, "train_rows": len(train), "held_out_x_rows": len(held), "unseen_rows": len(unseen), "result": result}
    args.out.mkdir(parents=True, exist_ok=True); (args.out / "eval_report.json").write_text(json.dumps(report, indent=2) + "\n"); print(json.dumps(report), flush=True)


if __name__ == "__main__": main()

"""Matched T=1 decimal, binary, and fixed-limb representation diagnostic."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


REPRESENTATIONS = {
    "decimal": {"width": 4, "vocab": 10, "base": 10},
    "binary": {"width": 7, "vocab": 2, "base": 2},
    # Fixed before the run: 2 little-endian 4-bit limbs, 8 representable bits.
    "limb4": {"width": 2, "vocab": 16, "base": 16},
}


def moduli(seed=0):
    primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47)
    values = sorted({p * q for i, p in enumerate(primes) for q in primes[i + 1:] if 21 <= p * q <= 99})
    return random.Random(seed).sample(values, 26)


def encode(value, kind):
    spec = REPRESENTATIONS[kind]
    if kind == "decimal":
        return [int(c) for c in f"{value:0{spec['width']}d}"][::-1]
    if kind == "binary":
        return [(value >> index) & 1 for index in range(spec["width"])]
    return [(value >> (4 * index)) & 15 for index in range(spec["width"])]


def decode(tokens, kind):
    if kind == "decimal":
        return sum(token * (10 ** index) for index, token in enumerate(tokens))
    if kind == "binary":
        return sum(token << index for index, token in enumerate(tokens))
    return sum(token << (4 * index) for index, token in enumerate(tokens))


def rows(moduli_list, kind, train_split):
    output = []
    for modulus in moduli_list:
        values = list(range(modulus)); random.Random(45 + modulus).shuffle(values)
        cut = int(.8 * len(values)); values = values[:cut] if train_split else values[cut:]
        for x in values:
            target = (x * x) % modulus
            output.append((encode(modulus, kind), encode(x, kind), encode(target, kind), target))
    return output


class RepresentationTape(nn.Module):
    def __init__(self, width, vocab, d=128):
        super().__init__(); self.width, self.vocab, self.d = width, vocab, d
        self.token = nn.Embedding(vocab, 32); self.place = nn.Embedding(width, 32)
        self.n_local = nn.Sequential(nn.Linear(32, d), nn.LayerNorm(d), nn.GELU())
        self.x_local = nn.Sequential(nn.Linear(32, d), nn.LayerNorm(d), nn.GELU())
        self.init = nn.Sequential(nn.Linear(2 * d, d), nn.LayerNorm(d), nn.GELU())
        self.local_mix = nn.Linear(3 * d, d)
        self.step = nn.GRUCell(2 * d, d)
        self.decoder = nn.Sequential(nn.Linear(d + 32, d), nn.GELU(), nn.Linear(d, vocab))

    def forward(self, n, x):
        pos = torch.arange(self.width, device=n.device); pe = self.place(pos)
        n_state = self.n_local(self.token(n) + pe); h = self.init(torch.cat((n_state, self.x_local(self.token(x) + pe)), -1))
        zeros = torch.zeros_like(h[:, :1]); left = torch.cat((zeros, h[:, :-1]), 1); right = torch.cat((h[:, 1:], zeros), 1)
        h = self.step(torch.cat((self.local_mix(torch.cat((left, h, right), -1)), n_state), -1).reshape(-1, 2 * self.d), h.reshape(-1, self.d)).reshape_as(h)
        return self.decoder(torch.cat((h, pe[None].expand(h.shape[0], -1, -1)), -1))


def batch(items, size, step, device):
    sample = [items[(step * size + i) % len(items)] for i in range(size)]
    n = torch.tensor([row[0] for row in sample], device=device); x = torch.tensor([row[1] for row in sample], device=device)
    y = torch.tensor([row[2] for row in sample], device=device); target = torch.tensor([row[3] for row in sample], device=device)
    return n, x, y, target


@torch.no_grad()
def evaluate(model, data, kind, device):
    model.eval(); exact = token_exact = total = 0; started = time.perf_counter()
    for start in range(0, len(data), 512):
        sample = data[start:start + 512]; n, x, y, target = batch(sample, len(sample), 0, device); logits = model(n, x); pred = logits.argmax(-1)
        token_exact += int((pred == y).sum()); exact += sum(decode(row, kind) == int(value) for row, value in zip(pred.tolist(), target.tolist())); total += len(sample)
    elapsed = time.perf_counter() - started
    return {"exact": exact / total, "token_accuracy": token_exact / (total * REPRESENTATIONS[kind]["width"]), "examples": total, "seconds": elapsed, "examples_per_second": total / max(elapsed, 1e-9), "milliseconds_per_example": 1000 * elapsed / total}


def run(kind, train, held, unseen, seconds, device, seed):
    spec = REPRESENTATIONS[kind]; torch.manual_seed(seed); model = RepresentationTape(spec["width"], spec["vocab"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, betas=(.9, .95), weight_decay=.05); started, step, curve = time.monotonic(), 0, []
    while time.monotonic() - started < seconds:
        n, x, y, target = batch(train, 512, step, device); model.train(); logits = model(n, x); loss = F.cross_entropy(logits.reshape(-1, spec["vocab"]), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); step += 1
        if step == 1 or step % 100 == 0: curve.append({"step": step, "seconds": round(time.monotonic() - started, 2), "loss": float(loss.detach()), "last_batch_exact": float((logits.argmax(-1) == y).all(-1).float().mean())})
    elapsed = time.monotonic() - started
    return {"representation": kind, "width": spec["width"], "vocab": spec["vocab"], "parameters": sum(p.numel() for p in model.parameters()), "steps": step, "seconds": elapsed, "steps_per_second": step / max(elapsed, 1e-9), "curve": curve, "train_last_batch_exact": curve[-1]["last_batch_exact"], "train": evaluate(model, train, kind, device), "held_out_x": evaluate(model, held, kind, device), "unseen_N": evaluate(model, unseen, kind, device)}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--kind", choices=tuple(REPRESENTATIONS)); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--seconds", type=float, default=120); parser.add_argument("--seed", type=int, default=0); parser.add_argument("--device", default="cuda"); args = parser.parse_args()
    device = torch.device(args.device); ms = moduli(); train_ms, unseen_ms = ms[:18], ms[18:]; train = rows(train_ms, args.kind, True); held = rows(train_ms, args.kind, False); unseen = rows(unseen_ms, args.kind, True) + rows(unseen_ms, args.kind, False)
    report = {"classification": "RESEARCH ONLY — T=1 representation comparison", "representation_spec": REPRESENTATIONS[args.kind], "train_moduli": train_ms, "unseen_moduli": unseen_ms, "train_rows": len(train), "held_out_x_rows": len(held), "unseen_rows": len(unseen), "result": run(args.kind, train, held, unseen, args.seconds, device, args.seed)}
    args.out.mkdir(parents=True, exist_ok=True); (args.out / "eval_report.json").write_text(json.dumps(report, indent=2) + "\n"); print(json.dumps(report), flush=True)


if __name__ == "__main__": main()

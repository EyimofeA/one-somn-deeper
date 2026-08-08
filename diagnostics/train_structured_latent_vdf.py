"""Matched small-N VDF state-topology and trace-identifiability diagnostic.

This file is research-only. Exact arithmetic appears only in the synthetic
data generator and evaluator; no model forward contains arithmetic.
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

W = 4
DEPTHS = (1, 2, 4, 8)


def digits(value: int) -> list[int]:
    return [int(c) for c in f"{value:0{W}d}"][::-1]


def make_moduli(seed: int) -> list[int]:
    primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47)
    values = sorted({p * q for i, p in enumerate(primes) for q in primes[i + 1:] if 21 <= p * q <= 99})
    return random.Random(seed).sample(values, 26)


def rows(moduli: list[int], train_split: bool, split_seed: int = 45):
    output = []
    for modulus in moduli:
        values = list(range(modulus))
        random.Random(split_seed + modulus).shuffle(values)
        cut = int(0.8 * len(values))
        values = values[:cut] if train_split else values[cut:]
        for x in values:
            state = x
            trace = []
            for depth in DEPTHS:
                state = x
                trace = []
                for _ in range(depth):
                    state = state * state % modulus
                    trace.append(digits(state))
                output.append((modulus, x, depth, trace))
    return output


class StateModel(nn.Module):
    """Global, independent-register, or local LSD-aligned recurrent state."""

    def __init__(self, kind: str, d: int = 128):
        super().__init__()
        self.kind, self.d = kind, d
        self.digit = nn.Embedding(10, 32)
        self.place = nn.Embedding(W, 32)
        self.n_local = nn.Sequential(nn.Linear(32, d), nn.LayerNorm(d), nn.GELU())
        self.x_local = nn.Sequential(nn.Linear(32, d), nn.LayerNorm(d), nn.GELU())
        self.global_enc = nn.Sequential(nn.Linear(2 * W * 32, d), nn.LayerNorm(d), nn.GELU(), nn.Linear(d, d))
        self.init_local = nn.Sequential(nn.Linear(2 * d, d), nn.LayerNorm(d), nn.GELU())
        self.step = nn.GRUCell(2 * d, d)
        self.local_mix = nn.Linear(3 * d, d) if kind == "structured" else None
        self.decoder = nn.Sequential(nn.Linear(d + 32, d), nn.GELU(), nn.Linear(d, 10))

    def forward(self, n, x, t, return_trace=False):
        positions = torch.arange(W, device=n.device)
        pe = self.place(positions)
        n_tokens = self.digit(n) + pe
        x_tokens = self.digit(x) + pe
        n_state = self.n_local(n_tokens)
        if self.kind == "global":
            h = self.global_enc(torch.cat((n_tokens, x_tokens), -1).flatten(1))
        else:
            h = self.init_local(torch.cat((n_state, self.x_local(x_tokens)), -1))
            if self.kind == "register":
                # Independent position slots, matching the historical control.
                pass
        all_logits = []
        for step in range(int(t.max().item())):
            if self.kind == "global":
                update_input = torch.cat((h, n_state.mean(1)), -1)
                new = self.step(update_input, h)
            else:
                if self.kind == "structured":
                    zeros = torch.zeros_like(h[:, :1])
                    left = torch.cat((zeros, h[:, :-1]), 1)
                    right = torch.cat((h[:, 1:], zeros), 1)
                    local = self.local_mix(torch.cat((left, h, right), -1))
                else:
                    local = h
                inp = torch.cat((local, n_state), -1)
                new = self.step(inp.reshape(-1, 2 * self.d), h.reshape(-1, self.d)).reshape_as(h)
            active = (step < t)
            h = torch.where(active.reshape(-1, *([1] if self.kind == "global" else [1, 1])), new, h)
            if self.kind == "global":
                decoded = self.decoder(torch.cat((h[:, None].expand(-1, W, -1), pe[None].expand(h.shape[0], -1, -1)), -1))
            else:
                decoded = self.decoder(torch.cat((h, pe[None].expand(h.shape[0], -1, -1)), -1))
            all_logits.append(decoded)
        return all_logits[-1], all_logits


def batch(items, size, step, device):
    sample = [items[(step * size + i) % len(items)] for i in range(size)]
    n = torch.tensor([digits(row[0]) for row in sample], device=device)
    x = torch.tensor([digits(row[1]) for row in sample], device=device)
    t = torch.tensor([row[2] for row in sample], device=device)
    trace = torch.full((len(sample), max(DEPTHS), W), -1, dtype=torch.long, device=device)
    for index, row in enumerate(sample):
        trace[index, :row[2]] = torch.tensor(row[3], device=device)
    return n, x, t, trace


@torch.no_grad()
def evaluate(model, data, device):
    model.eval(); result = {}; transition_by_step = {}
    for start in range(0, len(data), 512):
        n, x, t, trace = batch(data[start:start + 512], min(512, len(data) - start), 0, device)
        logits, all_logits = model(n, x, t, return_trace=True)
        for index, step_logits in enumerate(all_logits):
            active = t > index
            if active.any():
                value = (step_logits[active].argmax(-1) == trace[active, index]).all(-1).float().mean()
                transition_by_step.setdefault(str(index + 1), []).append(float(value))
        for depth in DEPTHS:
            active = t == depth
            if active.any():
                final_exact = (logits[active].argmax(-1) == trace[active, depth - 1]).all(-1).float().mean()
                transition_exact = (all_logits[depth - 1][active].argmax(-1) == trace[active, depth - 1]).all(-1).float().mean()
                result.setdefault(str(depth), {"final_exact": [], "transition_exact": []})
                result[str(depth)]["final_exact"].append(float(final_exact))
                result[str(depth)]["transition_exact"].append(float(transition_exact))
    return {"by_depth": {depth: {key: sum(values) / len(values) for key, values in metrics.items()} for depth, metrics in result.items()}, "transition_exact_by_step": {step: sum(values) / len(values) for step, values in transition_by_step.items()}}


def train(model, train_data, held_x, unseen, seconds, mode, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, betas=(.9, .95), weight_decay=.05)
    started, step, history = time.monotonic(), 0, []
    while time.monotonic() - started < seconds:
        n, x, t, trace = batch(train_data, 512, step, device)
        model.train(); final_logits, all_logits = model(n, x, t, return_trace=True)
        final_loss = F.cross_entropy(final_logits.reshape(-1, 10), trace[torch.arange(trace.shape[0], device=device), t - 1].reshape(-1))
        trace_loss = torch.zeros((), device=device)
        if mode == "trace":
            parts = []
            for index, logits in enumerate(all_logits):
                active = t > index
                if active.any():
                    parts.append(F.cross_entropy(logits[active].reshape(-1, 10), trace[active, index].reshape(-1)))
            trace_loss = torch.stack(parts).mean()
        loss = final_loss + trace_loss if mode == "trace" else final_loss
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); step += 1
        if step == 1 or step % 100 == 0:
            history.append({"step": step, "seconds": round(time.monotonic() - started, 2), "loss": float(loss), "final_loss": float(final_loss), "trace_loss": float(trace_loss), "train_exact": float((final_logits.argmax(-1) == trace[torch.arange(trace.shape[0], device=device), t - 1]).all(-1).float().mean())})
    return {"parameters": sum(parameter.numel() for parameter in model.parameters()), "steps": step, "seconds": time.monotonic() - started, "throughput_steps_per_second": step / max(time.monotonic() - started, 1e-9), "mode": mode, "kind": model.kind, "curve": history, "held_out_x": evaluate(model, held_x, device), "unseen_N": evaluate(model, unseen, device)}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--kind", choices=("global", "register", "structured"), required=True); parser.add_argument("--mode", choices=("final", "trace"), default="final"); parser.add_argument("--seconds", type=float, default=120); parser.add_argument("--seed", type=int, default=0); parser.add_argument("--device", default="cuda"); args = parser.parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed); device = torch.device(args.device)
    moduli = make_moduli(0); train_moduli, unseen_moduli = moduli[:18], moduli[18:]
    train_data = rows(train_moduli, True); held_x = rows(train_moduli, False); unseen = rows(unseen_moduli, False)
    model = StateModel(args.kind).to(device)
    report = {"classification": "RESEARCH ONLY — synthetic small-N state-topology diagnostic", "train_moduli": train_moduli, "unseen_moduli": unseen_moduli, "depths": DEPTHS, "train_rows": len(train_data), "held_out_x_rows": len(held_x), "unseen_rows": len(unseen), "result": train(model, train_data, held_x, unseen, args.seconds, args.mode, device)}
    args.out.mkdir(parents=True, exist_ok=True); (args.out / "eval_report.json").write_text(json.dumps(report, indent=2) + "\n"); print(json.dumps(report), flush=True)


if __name__ == "__main__": main()

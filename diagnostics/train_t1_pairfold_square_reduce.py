"""Final-label T=1 pair-routed square tape followed by learned reduction."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from train_t1_phase_square_reduce import evaluate
from train_t1_representation import batch, moduli, rows


class PairFoldSquareReduce(nn.Module):
    """Learn digit-pair categories, fold schoolbook columns, then scan LSD-first.

    Pair routing supplies topology only: pair values, carries, quotients, and
    residues are neither computed nor supervised.
    """

    def __init__(self, width: int = 4, vocab: int = 10, d: int = 128,
                 reduce_steps: int = 4):
        super().__init__()
        self.width, self.vocab, self.d = width, vocab, d
        self.reduce_steps = reduce_steps
        self.token = nn.Embedding(vocab, 32)
        self.place = nn.Embedding(width, 32)
        self.n_local = nn.Sequential(nn.Linear(32, d), nn.LayerNorm(d), nn.GELU())
        self.pair_table = nn.Embedding(vocab * vocab, d)
        self.pair_fold = nn.GRUCell(d, d)
        self.fold_initial = nn.Parameter(torch.zeros(d))
        self.square_carry = nn.GRUCell(d, d)
        self.carry_initial = nn.Parameter(torch.zeros(d))
        self.reduce_mix = nn.Linear(3 * d, d)
        self.reduce_cell = nn.GRUCell(2 * d, d)
        self.decoder = nn.Sequential(nn.Linear(d + 32, d), nn.GELU(), nn.Linear(d, vocab))
        nn.init.normal_(self.pair_table.weight, std=.02)

    @staticmethod
    def neighbors(h):
        zeros = torch.zeros_like(h[:, :1])
        return torch.cat((zeros, h[:, :-1]), 1), torch.cat((h[:, 1:], zeros), 1)

    def forward(self, n, x):
        batch_size = x.shape[0]
        pos = torch.arange(self.width, device=x.device)
        pe = self.place(pos)
        n_state = self.n_local(self.token(n) + pe)

        column_terms = [[] for _ in range(2 * self.width - 1)]
        for left in range(self.width):
            for right in range(self.width):
                pair_id = x[:, left] * self.vocab + x[:, right]
                column_terms[left + right].append(self.pair_table(pair_id))
        columns = []
        for terms in column_terms:
            folded = self.fold_initial[None].expand(batch_size, -1)
            for term in terms:
                folded = self.pair_fold(term, folded)
            columns.append(folded)

        carry = self.carry_initial[None].expand(batch_size, -1)
        square_tape = []
        for column in columns:
            carry = self.square_carry(column, carry)
            square_tape.append(carry)
        h = torch.stack(square_tape[:self.width], 1)

        for _ in range(self.reduce_steps):
            left, right = self.neighbors(h)
            local = self.reduce_mix(torch.cat((left, h, right), -1))
            h = self.reduce_cell(
                torch.cat((local, n_state), -1).reshape(-1, 2 * self.d),
                h.reshape(-1, self.d),
            ).reshape_as(h)
        return self.decoder(torch.cat((h, pe[None].expand(batch_size, -1, -1)), -1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    ms = moduli()
    train_ms, unseen_ms = ms[:18], ms[18:]
    train = rows(train_ms, "decimal", True)
    held = rows(train_ms, "decimal", False)
    unseen = rows(unseen_ms, "decimal", True) + rows(unseen_ms, "decimal", False)
    model = PairFoldSquareReduce().to(device)
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
        if step == 1 or step % 250 == 0:
            curve.append({"step": step, "seconds": round(time.monotonic() - started, 2),
                          "loss": float(loss.detach()),
                          "last_batch_exact": float((logits.argmax(-1) == y).all(-1).float().mean())})
    elapsed = time.monotonic() - started
    result = {
        "arm": "pairfold_square_reduce", "seed": args.seed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "steps": step, "seconds": elapsed, "steps_per_second": step / elapsed,
        "curve": curve, "train": evaluate(model, train, device),
        "held_out_x": evaluate(model, held, device),
        "unseen_N": evaluate(model, unseen, device),
    }
    report = {
        "classification": "RESEARCH ONLY — T=1 final-label pair-routed square tape",
        "intermediate_supervision": False,
        "routing_only": "learned pair categories grouped by i+j, learned fold and carry",
        "train_moduli": train_ms, "unseen_moduli": unseen_ms, "result": result,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "eval_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

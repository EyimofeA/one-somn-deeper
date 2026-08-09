"""Identity-initialized learned-residual reducer on public Easy T=1."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from train_t1_phase_square_reduce import PhaseSquareReduce, evaluate
from train_t1_phase_square_reduce_e5 import load_rows
from train_t1_representation import batch


class IdentityInitReducerPhaseSquareReduce(PhaseSquareReduce):
    """Factored tape whose reduction residual starts almost closed."""

    def __init__(self, initial_reduce_gate: float):
        super().__init__("factored")
        if not 0.0 < initial_reduce_gate < 1.0:
            raise ValueError("initial_reduce_gate must be strictly between 0 and 1")
        initial_logit = torch.logit(torch.tensor(initial_reduce_gate))
        self.reduce_residual_logit = nn.Parameter(initial_logit)

    @property
    def reduce_gate(self):
        return self.reduce_residual_logit.sigmoid()

    def forward(self, n, x):
        pos = torch.arange(self.width, device=n.device)
        pe = self.place(pos)
        n_state = self.n_local(self.token(n) + pe)
        h = self.init(self.x_local(self.token(x) + pe))
        square_context = self.null_square_context.expand(h.shape[0], -1, -1)
        for _ in range(self.square_steps):
            h = self.recurrent_step(
                h, square_context, self.square_mix, self.square_cell
            )
        gate = self.reduce_gate.to(dtype=h.dtype)
        for _ in range(self.reduce_steps):
            candidate = self.recurrent_step(
                h, n_state, self.reduce_mix, self.reduce_cell
            )
            h = h + gate * (candidate - h)
        return self.decoder(
            torch.cat((h, pe[None].expand(h.shape[0], -1, -1)), -1)
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--initial-reduce-gate", type=float, default=0.01)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train = load_rows(args.data_root / "train.jsonl")
    seen = load_rows(args.data_root / "depth_t_1.jsonl")
    unseen = load_rows(args.data_root / "depth_ood_n_t_1.jsonl")
    model = IdentityInitReducerPhaseSquareReduce(args.initial_reduce_gate).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-3, betas=(0.9, 0.95), weight_decay=0.05
    )

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
            curve.append(
                {
                    "step": step,
                    "seconds": round(time.monotonic() - started, 2),
                    "loss": float(loss.detach()),
                    "last_batch_exact": float(
                        (logits.argmax(-1) == y).all(-1).float().mean()
                    ),
                    "reduce_gate": float(model.reduce_gate.detach()),
                }
            )

    elapsed = time.monotonic() - started
    result = {
        "arm": "factored_e5_identity_init_reducer",
        "initial_reduce_gate": args.initial_reduce_gate,
        "final_reduce_gate": float(model.reduce_gate.detach()),
        "seed": args.seed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "steps": step,
        "seconds": elapsed,
        "steps_per_second": step / max(elapsed, 1e-9),
        "train_rows": len(train),
        "curve": curve,
        "train": evaluate(model, train, device),
        "seen_N_t1": evaluate(model, seen, device),
        "ood_N_t1": evaluate(model, unseen, device),
    }
    report = {
        "classification": "RESEARCH ONLY — identity-initialized reducer ablation",
        "one_variable": "learned residual reduction gate initialized at 0.01",
        "intermediate_supervision": False,
        "data_root": str(args.data_root),
        "result": result,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "eval_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

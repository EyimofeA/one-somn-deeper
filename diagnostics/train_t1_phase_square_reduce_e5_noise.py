"""Training-only square/reduce interface-noise ablation on public Easy T=1."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from train_t1_phase_square_reduce import PhaseSquareReduce, evaluate
from train_t1_phase_square_reduce_e5 import load_rows
from train_t1_representation import batch


class NoisyInterfacePhaseSquareReduce(PhaseSquareReduce):
    """The factored tape with noise only at its square/reduce boundary."""

    def __init__(self, interface_noise_std: float):
        super().__init__("factored")
        self.interface_noise_std = interface_noise_std

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
        if self.training and self.interface_noise_std:
            h = h + torch.randn_like(h) * self.interface_noise_std
        for _ in range(self.reduce_steps):
            h = self.recurrent_step(h, n_state, self.reduce_mix, self.reduce_cell)
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
    parser.add_argument("--interface-noise-std", type=float, default=0.1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train = load_rows(args.data_root / "train.jsonl")
    seen = load_rows(args.data_root / "depth_t_1.jsonl")
    unseen = load_rows(args.data_root / "depth_ood_n_t_1.jsonl")
    model = NoisyInterfacePhaseSquareReduce(args.interface_noise_std).to(device)
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
                }
            )

    elapsed = time.monotonic() - started
    result = {
        "arm": "factored_e5_interface_noise",
        "interface_noise_std": args.interface_noise_std,
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
        "classification": "RESEARCH ONLY — square/reduce interface-noise ablation",
        "one_variable": "training-only Gaussian noise at the square/reduce interface",
        "intermediate_supervision": False,
        "data_root": str(args.data_root),
        "result": result,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "eval_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

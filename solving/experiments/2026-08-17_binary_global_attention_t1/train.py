"""T=1 binary work-state diagnostic with a tied global-attention transition.

The deterministic rows and split policy match ``2026-08-16_binary_workstate_matched``.
The model sees only x and N bits and is supervised only on final residue bits.
No square, quotient, carry, comparison, or subtraction trace is provided.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


BASE_PATH = (
    Path(__file__).resolve().parents[1]
    / "2026-08-16_binary_workstate_matched"
    / "train.py"
)
_spec = importlib.util.spec_from_file_location("binary_matched_base", BASE_PATH)
assert _spec is not None and _spec.loader is not None
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

OPERAND_BITS = base.OPERAND_BITS
WORKSPACE_BITS = base.WORKSPACE_BITS


class RMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(value, (value.shape[-1],), self.weight)


class TiedAttentionCell(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.width = width
        self.heads = heads
        self.dropout = dropout
        self.attention_norm = RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.attention_out = nn.Linear(width, width, bias=False)
        self.mlp_norm = RMSNorm(width)
        self.up = nn.Linear(width, 4 * width, bias=False)
        self.down = nn.Linear(4 * width, width, bias=False)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        residual = state
        value = self.attention_norm(state)
        batch, length, _ = value.shape
        q, k, v = self.qkv(value).chunk(3, dim=-1)
        head_width = self.width // self.heads
        q = q.view(batch, length, self.heads, head_width).transpose(1, 2)
        k = k.view(batch, length, self.heads, head_width).transpose(1, 2)
        v = v.view(batch, length, self.heads, head_width).transpose(1, 2)
        value = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
        )
        value = value.transpose(1, 2).contiguous().view(batch, length, self.width)
        state = residual + self.attention_out(value)
        value = F.silu(self.up(self.mlp_norm(state)))
        return state + self.down(value)


class GlobalBinaryWorkState(nn.Module):
    def __init__(
        self, width: int, updates: int, heads: int, dropout: float
    ) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.updates = updates
        self.bit_embedding = nn.Embedding(2, width)
        self.position_embedding = nn.Embedding(WORKSPACE_BITS, width)
        self.role_embedding = nn.Embedding(3, width)
        self.cell = TiedAttentionCell(width, heads, dropout)
        self.final_norm = RMSNorm(width)
        self.readout = nn.Linear(width, 1)

    def _tape(self, bits: torch.Tensor, role: int) -> torch.Tensor:
        positions = torch.arange(WORKSPACE_BITS, device=bits.device)
        role_id = torch.tensor(role, device=bits.device)
        return (
            self.bit_embedding(bits)
            + self.position_embedding(positions)[None]
            + self.role_embedding(role_id)[None, None]
        )

    def forward(
        self, source_bits: torch.Tensor, modulus_bits: torch.Tensor
    ) -> torch.Tensor:
        source = self._tape(source_bits, 0)
        modulus = self._tape(modulus_bits, 1)
        work = self._tape(source_bits, 2)
        for _ in range(self.updates):
            visible = torch.cat((source, modulus, work), dim=1)
            updated = self.cell(visible)
            work = updated[:, 2 * WORKSPACE_BITS :]
        return self.readout(self.final_norm(work[:, :OPERAND_BITS])).squeeze(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=74)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--muon-learning-rate", type=float, default=0.006)
    parser.add_argument("--muon-weight-decay", type=float, default=0.1)
    parser.add_argument("--muon-warmup-steps", type=int, default=250)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")

    train_x, heldout_x = base.split_values(args.seed)
    train_n, unseen_n = base.split_moduli(args.seed)
    train = base.make_rows(train_n, train_x, 100_000, args.seed + 1)
    validation = base.make_rows(train_n, heldout_x, 5_000, args.seed + 2)
    evaluation_sets = {
        "train": train,
        "validation_unseen_x_seen_n": validation,
        "audit_seen_x_unseen_n": base.make_rows(
            unseen_n, train_x, 5_000, args.seed + 3
        ),
        "audit_unseen_x_unseen_n": base.make_rows(
            unseen_n, heldout_x, 5_000, args.seed + 4
        ),
    }

    raw_model = GlobalBinaryWorkState(
        args.width, args.updates, args.heads, args.dropout
    ).to(device)
    model = torch.compile(raw_model, dynamic=False) if args.compile else raw_model
    matrix = [parameter for parameter in model.parameters() if parameter.ndim >= 2]
    scalar = [parameter for parameter in model.parameters() if parameter.ndim < 2]
    optimizers = [
        base.ConvMuon(
            matrix,
            lr=args.muon_learning_rate,
            weight_decay=args.muon_weight_decay,
        ),
        torch.optim.AdamW(
            scalar, lr=3e-4, betas=(0.9, 0.95), weight_decay=1e-5
        ),
    ]

    generator = torch.Generator(device="cpu").manual_seed(args.seed + 5)
    best_validation = -1.0
    best_step = 0
    best_state = None
    curve = []
    started = time.perf_counter()

    for step in range(1, args.steps + 1):
        optimizers[0].param_groups[0]["lr"] = args.muon_learning_rate * min(
            1.0, step / args.muon_warmup_steps
        )
        indices = torch.randint(
            len(train), (args.batch_size,), generator=generator
        ).tolist()
        sample = [train[index] for index in indices]
        source, modulus, target = base.tensor_batch(sample, "fused_x", device)
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(source, modulus)
            loss = F.binary_cross_entropy_with_logits(logits, target)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for optimizer in optimizers:
            optimizer.step()

        if step == 1 or step % args.eval_every == 0:
            train_probe = base.evaluate(model, train[:5000], "fused_x", device)
            validation_metrics = base.evaluate(model, validation, "fused_x", device)
            record = {
                "type": "progress",
                "step": step,
                "examples": step * args.batch_size,
                "seconds": time.perf_counter() - started,
                "loss": float(loss.detach()),
                "learning_rate": optimizers[0].param_groups[0]["lr"],
                "train_exact": train_probe["exact"],
                "validation_exact": validation_metrics["exact"],
            }
            curve.append(record)
            print(json.dumps(record), flush=True)
            if validation_metrics["exact"] > best_validation:
                best_validation = validation_metrics["exact"]
                best_step = step
                best_state = copy.deepcopy(model.state_dict())

    assert best_state is not None
    model.load_state_dict(best_state)
    report = {
        "architecture": "tied_global_attention",
        "supervision": "final residue bits only",
        "seed": args.seed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "width": args.width,
        "heads": args.heads,
        "updates": args.updates,
        "dropout": args.dropout,
        "steps": args.steps,
        "examples": args.steps * args.batch_size,
        "best_step": best_step,
        "elapsed_seconds": time.perf_counter() - started,
        "split": {name: len(rows) for name, rows in evaluation_sets.items()},
        "selected": {
            name: base.evaluate(model, rows, "fused_x", device)
            for name, rows in evaluation_sets.items()
        },
        "curve": curve,
    }
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(state, args.out / "model_best.pt")
    (args.out / "eval_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"type": "final", **report}, default=str), flush=True)


if __name__ == "__main__":
    main()

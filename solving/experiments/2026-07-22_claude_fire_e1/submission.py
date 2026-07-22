"""FIRE relative attention bias, no other positional signal — claude code.

Li et al. 2023 (arXiv 2310.04418): a learned relative-position bias added to
attention logits, produced by a small MLP over a normalized relative
distance — "progressive interpolation" keeps the MLP's input bounded at any
sequence length. Reference studied at `learnings/playground/fire.py`:

    b(i, j) = f_θ( ψ(i − j) / ψ(max(L, i)) ),   ψ(x) = log(c·x + 1)

One deliberate departure from that reference: `fire.py` is causal
(`dist = (i − j).clamp(min=0)`, triu-masked). This task's attention is
bidirectional — full self-attention, answer read off the tail positions of
the input, no causal mask anywhere in this project (see `claude_pv_evalk4`,
`claude_std_rope_e1`). The causal formula assumes i ≥ j always, so relative
distance is trivially non-negative; that assumption breaks here. This uses a
signed generalization instead:

    d = i − j                                    (signed, no clamp)
    b(i, j) = f_θ( sign(d) · ψ(|d|) / ψ(max(L, i, j)) )

— same ψ, same learnable c and L, but the normalizer uses max(i, j) instead
of the causal max(L, i), and sign is carried through separately so the MLP
can still tell "query is left of key" from "query is right of key." This is
a bidirectional extension, not a literal implementation of the paper's Eq. 4.

Pure ablation, matching `claude_abacus_e1`'s discipline: token embedding
only, no Abacus, no absolute position embedding, no depth embedding — FIRE
is the *only* source of sequence-order information. One FIRE module shared
across all 4 layers (not one per layer), same as RoPE's shared cos/sin
table in the anchor card, to keep the position-scheme swap the single
changed variable. Same base otherwise: 4 independent Pre-LN layers, d=32,
heads=4, matching `claude_std_rope_e1`.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from benchmark import (
    ModelSpec,
    OptimizerBundle,
    OptimizerSpec,
    Submission,
    assert_model_state,
)


D_MODEL = 32
NUM_HEADS = 4
NUM_LAYERS = 4
FIRE_MLP_HIDDEN = 32


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class RMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight)


class FIRE(nn.Module):
    """Per-head additive attention bias from (signed, bidirectional) relative position."""

    def __init__(self, num_heads: int, mlp_hidden: int = FIRE_MLP_HIDDEN) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.log_c = nn.Parameter(torch.zeros(1))
        self.raw_L = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(1, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, num_heads),
        )

    def _psi(self, x: Tensor) -> Tensor:
        c = F.softplus(self.log_c) + 1e-6
        return torch.log(c * x + 1.0)

    def bias(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        pos = torch.arange(1, seq_len + 1, device=device, dtype=torch.float32)
        i = pos.view(-1, 1)
        j = pos.view(1, -1)
        d = i - j
        L = F.softplus(self.raw_L) + 1.0
        normalizer = torch.maximum(L, torch.maximum(i, j))
        x = torch.sign(d) * self._psi(d.abs()) / self._psi(normalizer)
        out = self.mlp(x.unsqueeze(-1).to(dtype))
        return out.permute(2, 0, 1).unsqueeze(0)


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.out = nn.Linear(D_MODEL, D_MODEL)
        self.mixer_norm = RMSNorm(D_MODEL)
        self.up = nn.Linear(D_MODEL, 4 * D_MODEL)
        self.down = nn.Linear(4 * D_MODEL, D_MODEL)

    def forward(self, x: Tensor, attention_mask: Tensor | None, fire_bias: Tensor) -> Tensor:
        residual = x
        x = self.attention_norm(x)
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        k = k.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        v = v.view(batch, length, NUM_HEADS, -1).transpose(1, 2)

        bias = fire_bias.to(dtype=x.dtype)
        if attention_mask is not None:
            if attention_mask.shape == (batch, length):
                pad = attention_mask[:, None, None, :]
            elif attention_mask.shape == (batch, length, length):
                pad = attention_mask[:, None, :, :]
            else:
                raise ValueError("invalid attention_mask shape")
            pad = pad.to(device=x.device, dtype=torch.bool)
            pad_bias = torch.zeros(pad.shape, device=x.device, dtype=x.dtype)
            pad_bias = pad_bias.masked_fill(~pad, float("-inf"))
            bias = bias + pad_bias

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        x = x.transpose(1, 2).contiguous().view(batch, length, D_MODEL)
        x = residual + self.out(x)
        return x + self.down(F.gelu(self.up(self.mixer_norm(x))))


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.token_embedding = nn.Embedding(spec.vocab_size, D_MODEL)
        self.fire = FIRE(NUM_HEADS)
        self.layers = nn.ModuleList([Block() for _ in range(NUM_LAYERS)])
        self.final_norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight
        nn.init.normal_(self.token_embedding.weight, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        x = self.token_embedding(input_ids)
        fire_bias = self.fire.bias(input_ids.shape[1], input_ids.device, x.dtype)
        for layer in self.layers:
            x = layer(x, attention_mask, fire_bias)
        return self.head(self.final_norm(x)), None


WARMUP_FRACTION = 0.05
FINAL_LR_FRACTION = 0.01


def _build_scheduler(
    optimizer: torch.optim.Optimizer, spec: OptimizerSpec
) -> torch.optim.lr_scheduler.LRScheduler:
    import time

    total_seconds = max(1.0, float(spec.training_time_seconds))
    started = time.monotonic()

    def factor(_step: int) -> float:
        progress = (time.monotonic() - started) / total_seconds
        progress = min(max(progress, 0.0), 1.0)
        if progress < WARMUP_FRACTION:
            return FINAL_LR_FRACTION + (1.0 - FINAL_LR_FRACTION) * (
                progress / WARMUP_FRACTION
            )
        tail = (progress - WARMUP_FRACTION) / (1.0 - WARMUP_FRACTION)
        cosine = 0.5 * (1.0 + math.cos(math.pi * tail))
        return FINAL_LR_FRACTION + (1.0 - FINAL_LR_FRACTION) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-3,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        capturable=spec.device_type == "cuda",
    )
    return OptimizerBundle(optimizer, scheduler=_build_scheduler(optimizer, spec))


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    batch_size=256,
    eval_batch_size=512,
)

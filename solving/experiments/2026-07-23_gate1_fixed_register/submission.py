"""Standard (non-recurrent) Transformer, plain token embedding, RoPE — claude code.

Sanity-check baseline, deliberately reset away from every prior card. Two
axes change together against the `depth_d32_k4_ut` anchor, both under one
umbrella hypothesis ("does a textbook architecture learn this at all"),
not as an isolated one-variable ablation:

  1. Weight-tying removed. `depth_d32_k4_ut` applies ONE shared Block four
     times (Universal-Transformer style). This card stacks 4 INDEPENDENTLY
     parameterized layers — an ordinary Pre-LN Transformer encoder.
  2. Position encoding replaced. No learned absolute position embedding, no
     depth embedding. Position enters only through RoPE (Su et al. 2021)
     inside attention — genuinely relative, translation-invariant.

Same d_model=32, same heads=4, same depth=4 as the anchor, so op count and
launch count per step should be comparable and any throughput delta is
attributable to weight-tying, not size.

Before embedding, every row is packed into the right edge of one fixed
`max_seq_len` register. RoPE therefore sees stable tail coordinates across
input lengths. Returned logits are mapped back to the evaluator's original
per-row positions and tensor width.

Everything else (AdamW, batch 256) matches prior cards. Scheduler is the
wall-clock fix validated in `learnings/concepts/15-lr-schedules-wallclock.md`
rather than the older step-count-based `t_max = seconds * 8` — chosen here
because this is a fresh card, not a continuation of the calibrated lineage.
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
HEAD_DIM = D_MODEL // NUM_HEADS
NUM_LAYERS = 4
ROPE_BASE = 10000.0


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


def _rope_cos_sin(
    seq_len: int, head_dim: int, device: torch.device, dtype: torch.dtype
) -> tuple[Tensor, Tensor]:
    inv_freq = 1.0 / (
        ROPE_BASE
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


def _valid_token_mask(
    input_ids: Tensor,
    attention_mask: Tensor | None,
) -> Tensor:
    if attention_mask is None:
        return torch.ones_like(input_ids, dtype=torch.bool)
    if attention_mask.shape == input_ids.shape:
        return attention_mask.to(device=input_ids.device, dtype=torch.bool)
    if attention_mask.shape == (
        input_ids.shape[0],
        input_ids.shape[1],
        input_ids.shape[1],
    ):
        return attention_mask.to(
            device=input_ids.device,
            dtype=torch.bool,
        ).any(dim=1)
    raise ValueError("invalid attention_mask shape")


def _right_aligned_register(
    input_ids: Tensor,
    attention_mask: Tensor | None,
    max_seq_len: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Pack valid row tokens at the register tail and retain reverse mapping."""
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape (batch, length)")
    if input_ids.shape[1] > max_seq_len:
        raise ValueError("input sequence exceeds the fixed register")

    valid = _valid_token_mask(input_ids, attention_mask)
    valid_lengths = valid.to(torch.long).sum(dim=1)
    source_rank = valid.to(torch.long).cumsum(dim=1) - 1
    register_start = max_seq_len - valid_lengths
    register_positions = register_start[:, None] + source_rank
    safe_positions = torch.where(
        valid,
        register_positions,
        torch.zeros_like(register_positions),
    )

    register_ids = torch.zeros(
        input_ids.shape[0],
        max_seq_len,
        device=input_ids.device,
        dtype=input_ids.dtype,
    )
    register_ids.scatter_add_(
        1,
        safe_positions,
        torch.where(valid, input_ids, torch.zeros_like(input_ids)),
    )
    grid = torch.arange(max_seq_len, device=input_ids.device)[None, :]
    register_mask = grid >= register_start[:, None]
    return register_ids, register_mask, safe_positions, valid


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.out = nn.Linear(D_MODEL, D_MODEL)
        self.mixer_norm = RMSNorm(D_MODEL)
        self.up = nn.Linear(D_MODEL, 4 * D_MODEL)
        self.down = nn.Linear(4 * D_MODEL, D_MODEL)

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor | None,
        cos: Tensor,
        sin: Tensor,
    ) -> Tensor:
        residual = x
        x = self.attention_norm(x)
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        k = k.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        v = v.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)
        mask = None
        if attention_mask is not None:
            if attention_mask.shape == (batch, length):
                mask = attention_mask[:, None, None, :]
            elif attention_mask.shape == (batch, length, length):
                mask = attention_mask[:, None, :, :]
            else:
                raise ValueError("invalid attention_mask shape")
            mask = mask.to(device=x.device, dtype=torch.bool)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x.transpose(1, 2).contiguous().view(batch, length, D_MODEL)
        x = residual + self.out(x)
        return x + self.down(F.gelu(self.up(self.mixer_norm(x))))


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.token_embedding = nn.Embedding(spec.vocab_size, D_MODEL)
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
        register_ids, register_mask, source_to_register, valid = (
            _right_aligned_register(
                input_ids,
                attention_mask,
                self.config.max_seq_len,
            )
        )
        x = self.token_embedding(register_ids)
        cos, sin = _rope_cos_sin(
            self.config.max_seq_len,
            HEAD_DIM,
            input_ids.device,
            x.dtype,
        )
        for layer in self.layers:
            x = layer(x, register_mask, cos, sin)
        register_logits = self.head(self.final_norm(x))
        gather_positions = source_to_register[:, :, None].expand(
            -1,
            -1,
            register_logits.shape[-1],
        )
        logits = register_logits.gather(1, gather_positions)
        logits = logits.masked_fill(~valid[:, :, None], 0.0)
        return logits, None


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

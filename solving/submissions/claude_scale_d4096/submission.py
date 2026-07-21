"""Throughput calibration at d=4096 — claude code.

NOT a Hard candidate on its own. Purpose: measure how many optimiser steps an H100
completes in the 60s Easy clock at this width, so the Hard width can be chosen from
data instead of guessed.

Why this matters: on Easy, d=128 completed 394 steps against d=32's 377 — 13x the
parameters at no throughput cost, because 32x32 matmuls leave the GPU kernel-launch-
bound rather than FLOP-bound. Width is free until it isn't; this finds the knee.

Identical to `claude_hard_h1` except D_MODEL/NUM_HEADS/BASE_LR.
"""

from __future__ import annotations

import math
import time

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


D_MODEL = 4096
NUM_HEADS = 32
MIN_LOOPS = 4
MAX_LOOPS = 16
BASE_LR = 2.0e-4  # 3e-3 at d=32, scaled by ~1/sqrt(256/32)
WARMUP_FRACTION = 0.05
FINAL_LR_FRACTION = 0.01

# Structural token ids are public benchmark markers (not the trapdoor).
MARKER_LO = 2  # N
MARKER_HI = 4  # T  (N=2, X=3, T=4 are contiguous and in sequence order)
PAD_ID = 0
DIGIT_LO = 7
DIGIT_HI = 16
MAX_PLACES = 64


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


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.out = nn.Linear(D_MODEL, D_MODEL)
        self.mixer_norm = RMSNorm(D_MODEL)
        self.up = nn.Linear(D_MODEL, 4 * D_MODEL)
        self.down = nn.Linear(4 * D_MODEL, D_MODEL)

    def forward(self, x: Tensor, attention_mask: Tensor | None) -> Tensor:
        residual = x
        x = self.attention_norm(x)
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        k = k.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        v = v.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
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


def _field_and_place(input_ids: Tensor) -> tuple[Tensor, Tensor]:
    """Derive (field_id, place_id) from input_ids alone. claude code.

    field_id in {0,1,2,3}: 0 = pre-N, 1 = N span, 2 = X span, 3 = T span.
    place_id in {0..MAX_PLACES-1}: 0 for non-digits; for a digit it is
    1 + (distance from the least-significant digit of its span), so the units digit of
    every operand shares place_id == 1 regardless of operand length.

    Cheap form: one cumsum for `field` (markers are ordered N < X < T, so the sum of
    three cumsums is the cumsum of the sum) and one reverse-cummin for span boundaries.
    Verified bit-identical to the naive per-field-loop version on train/test/ood at
    three padding widths.
    """
    device = input_ids.device
    batch, length = input_ids.shape
    positions = torch.arange(length, device=device)[None, :].expand(batch, length)

    markers = (input_ids >= MARKER_LO) & (input_ids <= MARKER_HI)
    field = markers.cumsum(dim=1).clamp_(0, 3)

    boundary = markers | (input_ids == PAD_ID)
    marked = torch.where(boundary, positions, torch.full_like(positions, length))
    next_boundary = marked.flip(1).cummin(dim=1).values.flip(1)
    strictly_after = torch.cat(
        [
            next_boundary[:, 1:],
            torch.full((batch, 1), length, device=device, dtype=next_boundary.dtype),
        ],
        dim=1,
    )
    span_end = strictly_after - 1

    is_digit = (input_ids >= DIGIT_LO) & (input_ids <= DIGIT_HI)
    place = torch.where(
        is_digit,
        (span_end - positions + 1).clamp(1, MAX_PLACES - 1),
        torch.zeros_like(positions),
    )
    return field, place


def _exponent_value(input_ids: Tensor, field: Tensor, place: Tensor) -> Tensor:
    """Parse the integer T from the T-marker span. claude code.

    Digits carry value `id - DIGIT_LO`; `place` is 1-indexed from the units end, so the
    span's value is sum(digit * 10**(place-1)). Places cap at 3 so a hidden three-digit
    T still parses without overflow.
    """
    is_digit = (input_ids >= DIGIT_LO) & (input_ids <= DIGIT_HI)
    t_mask = is_digit & (field == 3)
    digit_val = (input_ids - DIGIT_LO).clamp(0, 9)
    exponent = (place - 1).clamp(0, 3)
    weight = torch.pow(torch.full_like(exponent, 10), exponent)
    return (digit_val * weight * t_mask).sum(dim=1)


def _depth_table(width: int, size: int) -> Tensor:
    """Fixed, deterministic, near-orthogonal depth codes. claude code.

    Distinct at every k and defined at every k, so no loop depth is ever untrained.
    Random high-dimensional vectors are near-orthogonal (measured max off-diagonal
    cosine 0.52 at d=32); a sinusoidal encoding measured 0.96, i.e. adjacent loops look
    nearly identical, which is the degeneration that collapsed `claude_evalk4_zeroinit`.
    """
    generator = torch.Generator().manual_seed(0)
    return torch.randn(size, width, generator=generator)


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.token_embedding = nn.Embedding(spec.vocab_size, D_MODEL)
        self.position_embedding = nn.Embedding(spec.max_seq_len, D_MODEL)
        self.segment_embedding = nn.Embedding(4, D_MODEL)
        self.place_embedding = nn.Embedding(MAX_PLACES, D_MODEL)
        self.depth_gain = nn.Parameter(torch.ones(()))
        self.register_buffer("depth_table", _depth_table(D_MODEL, MAX_LOOPS))
        self.block = Block()
        self.final_norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight
        # The head is TIED to token_embedding, so the embedding init sets the logit
        # scale. PyTorch's default N(0,1) gives logits with std ~sqrt(d): initial loss
        # measured 83.4 at d=256 (and ~12 at d=32) against ln(17) = 2.83 for a
        # well-scaled init. Early training is then spent shrinking logits rather than
        # learning. std=0.02 puts the starting loss at the theoretical floor. claude code
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        nn.init.zeros_(self.segment_embedding.weight)
        nn.init.zeros_(self.place_embedding.weight)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        device = input_ids.device
        positions = torch.arange(input_ids.shape[1], device=device)
        field, place = _field_and_place(input_ids)
        x = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)
            + self.segment_embedding(field)
            + self.place_embedding(place)
        )

        # One block application per squaring, floored and capped. claude code
        t_val = _exponent_value(input_ids, field, place)
        n_loops = t_val.clamp(MIN_LOOPS, MAX_LOOPS)
        steps = int(n_loops.max().item())

        for k in range(steps):
            depth = self.depth_gain * self.depth_table[min(k, MAX_LOOPS - 1)]
            updated = self.block(x + depth, attention_mask)
            active = (n_loops > k)[:, None, None]
            x = torch.where(active, updated, x)
        return self.head(self.final_norm(x)), None


def _build_scheduler(
    optimizer: torch.optim.Optimizer, spec: OptimizerSpec
) -> torch.optim.lr_scheduler.LRScheduler:
    """Wall-clock cosine. Anneals exactly once over the real budget. claude code."""
    total_seconds = max(1.0, float(spec.training_time_seconds))
    started = time.monotonic()

    def factor(_step: int) -> float:
        progress = (time.monotonic() - started) / total_seconds
        progress = min(max(progress, 0.0), 1.0)
        if progress < WARMUP_FRACTION:
            ramp = progress / WARMUP_FRACTION
            return FINAL_LR_FRACTION + (1.0 - FINAL_LR_FRACTION) * ramp
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
        lr=BASE_LR,
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

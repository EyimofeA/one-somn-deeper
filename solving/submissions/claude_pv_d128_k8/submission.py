"""Scaled place-value UT loop (d=128, K=8) — claude code.

The Medium bet. Two constants change vs `claude_pv_fast`: width 32 -> 128 and loops
4 -> 8. Everything else, including the bit-exact cheap (field, place) derivation, is
unchanged.

Rationale (claude code): d=128 was tested on Easy and LOST badly — e1 2.00% vs d=32's
5.83%. That result is real but it is *60-second-specific*. Easy training dies with the
loss still falling (~380 steps), so width loses purely by buying fewer steps against a
clock that was already the binding constraint. Medium gives 600s — ten times the
budget — which is exactly the regime where that trade should invert.

K=8 for the same reason: Easy's T in {1,2,3} needs little depth, but Medium runs
T = 4/8/16 and a 4-loop block cannot perform 16 sequential squarings.

Prediction, recorded before scoring (claude code): this should lose on Easy, roughly
reproducing the d=128 result, and that is not evidence against it for Medium. If it
loses on Easy AND on Medium, the scaling axis is dead at both clocks and we stop
reaching for parameters. Compare against `claude_pv_tadapt`, which spends the same
600s on adaptive depth instead of width.
"""

from __future__ import annotations

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


D_MODEL = 128
NUM_HEADS = 8
NUM_LOOPS = 8

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
    """Derive (field_id, place_id) from input_ids alone — cheap form. claude code.

    Bit-identical to C1's `_field_and_place`; see module docstring.

    field_id in {0,1,2,3}: 0 = pre-N, 1 = N span, 2 = X span, 3 = T span.
    place_id in {0..MAX_PLACES-1}: 0 for non-digits; for a digit it is
    1 + (distance from the least-significant digit of its span).
    """
    device = input_ids.device
    batch, length = input_ids.shape
    positions = torch.arange(length, device=device)[None, :].expand(batch, length)

    markers = (input_ids >= MARKER_LO) & (input_ids <= MARKER_HI)
    field = markers.cumsum(dim=1).clamp_(0, 3)

    # A span ends one position before the next marker, or before the padding.
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


class Model(nn.Module):
    num_loops = NUM_LOOPS

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.token_embedding = nn.Embedding(spec.vocab_size, D_MODEL)
        self.position_embedding = nn.Embedding(spec.max_seq_len, D_MODEL)
        self.segment_embedding = nn.Embedding(4, D_MODEL)
        self.place_embedding = nn.Embedding(MAX_PLACES, D_MODEL)
        self.depth_embedding = nn.Embedding(self.num_loops, D_MODEL)
        self.block = Block()
        self.final_norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight
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
        for k in range(self.num_loops):
            depth = self.depth_embedding(
                torch.tensor(k, device=device, dtype=torch.long)
            )
            x = x + depth
            x = self.block(x, attention_mask)
        return self.head(self.final_norm(x)), None


def _build_scheduler(
    optimizer: torch.optim.Optimizer, spec: OptimizerSpec
) -> torch.optim.lr_scheduler.LRScheduler:
    t_max = max(100, int(spec.training_time_seconds * 8))
    warmup_steps = max(1, int(0.05 * t_max))
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_max - warmup_steps
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
    )


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

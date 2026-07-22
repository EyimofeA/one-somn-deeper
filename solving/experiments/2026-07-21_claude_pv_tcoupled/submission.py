"""T-coupled loop count on the place-value UT loop — claude code.

One-change ablation vs `claude_pv_k4_ut` (C1). C1 keeps a FIXED K=4 loops for
every example, so the block has no mechanism for "do one more squaring" — the
compute it spends is independent of the exponent T it was asked for. That makes
unseen-T generalisation structurally unreachable: an OOD T=4 row gets exactly the
same 4 applications a T=1 row gets.

Change (claude code): derive T from `input_ids` (public T-marker span, no dataset
access) and run `n_loops = T + 1` applications per example, gating the residual
update per row so finished examples stop changing.

Compute control (claude code): loop count is `max(T)+1` over the batch. With
Easy T in {1,2,3} that is 4 — EXACTLY the anchor's fixed K=4 — so wall-clock per
step is matched to C1 and any delta is bias, not a compute trade. Rows with small
T get *less* compute, which is the point.

Depth embeddings are indexed `min(k, 3)` so loops beyond the in-distribution max
reuse a trained slot instead of hitting an untrained one — this is what lets the
recurrence extrapolate to T > 3 at eval.

Everything else (d=32, 4 heads, shared Block, place-value + field embeddings,
AdamW warmup+cosine, batch 256) is held identical to C1.
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


D_MODEL = 32
NUM_HEADS = 4
NUM_DEPTH_EMB = 4  # trained depth slots; loop k uses min(k, NUM_DEPTH_EMB-1)
MAX_LOOPS = 8  # hard ceiling so an absurd parsed T cannot hang the run

# Structural token ids are public benchmark markers (not the trapdoor).
MARKER_N = 2
MARKER_X = 3
MARKER_T = 4
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
    """Derive (field_id, place_id) per position from input_ids alone. claude code."""
    device = input_ids.device
    length = input_ids.shape[1]
    positions = torch.arange(length, device=device)

    seen_n = (input_ids == MARKER_N).cumsum(dim=1)
    seen_x = (input_ids == MARKER_X).cumsum(dim=1)
    seen_t = (input_ids == MARKER_T).cumsum(dim=1)
    field = (seen_n + seen_x + seen_t).clamp_(0, 3).long()

    is_digit = (input_ids >= DIGIT_LO) & (input_ids <= DIGIT_HI)
    place = torch.zeros_like(input_ids, dtype=torch.long)
    pos_row = positions[None, :]
    for f in (1, 2, 3):
        fmask = is_digit & (field == f)
        masked_pos = torch.where(fmask, pos_row, torch.full_like(pos_row, -1))
        last_pos = masked_pos.max(dim=1, keepdim=True).values
        place_f = (last_pos - pos_row + 1).clamp_(1, MAX_PLACES - 1)
        place = torch.where(fmask, place_f, place)
    return field, place


def _exponent_value(input_ids: Tensor, field: Tensor, place: Tensor) -> Tensor:
    """Parse the integer T from the T-marker span. claude code.

    Digits carry value `id - DIGIT_LO`; `place` is 1-indexed from the units end,
    so the span's value is sum(digit * 10**(place-1)). Places are capped at 2
    (T < 100) to keep the power table small and overflow-free.
    """
    is_digit = (input_ids >= DIGIT_LO) & (input_ids <= DIGIT_HI)
    t_mask = is_digit & (field == 3)
    digit_val = (input_ids - DIGIT_LO).clamp_(0, 9)
    exponent = (place - 1).clamp_(0, 2)
    weight = torch.pow(torch.full_like(exponent, 10), exponent)
    return (digit_val * weight * t_mask).sum(dim=1)


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.token_embedding = nn.Embedding(spec.vocab_size, D_MODEL)
        self.position_embedding = nn.Embedding(spec.max_seq_len, D_MODEL)
        self.segment_embedding = nn.Embedding(4, D_MODEL)
        self.place_embedding = nn.Embedding(MAX_PLACES, D_MODEL)
        self.depth_embedding = nn.Embedding(NUM_DEPTH_EMB, D_MODEL)
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

        # Per-row loop budget tracks the exponent. claude code
        t_val = _exponent_value(input_ids, field, place)
        n_loops = (t_val + 1).clamp_(1, MAX_LOOPS)
        steps = int(n_loops.max().item())

        for k in range(steps):
            depth_idx = min(k, NUM_DEPTH_EMB - 1)
            depth = self.depth_embedding(
                torch.tensor(depth_idx, device=device, dtype=torch.long)
            )
            updated = self.block(x + depth, attention_mask)
            active = (n_loops > k)[:, None, None]
            x = torch.where(active, updated, x)
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

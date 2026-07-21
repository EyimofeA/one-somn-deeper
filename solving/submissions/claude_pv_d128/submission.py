"""Width-scaled place-value UT loop (d=128) — claude code.

One-change ablation vs `claude_pv_k4_ut` (C1): width only, 32 -> 128, heads 4 -> 8.

Motivation (claude code): every card in this repo sits at d=32 (~14K state) against
a 500M ceiling — roughly four orders of magnitude of unused capacity. The 60s clock
is the real budget, not parameters, so the open question is whether the extra width
pays for the steps it costs. Loop count (K=4), place-value/field embeddings, optimizer,
schedule and batch size are all held identical to C1.
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
NUM_LOOPS = 4

# Structural token ids are public benchmark markers (not the trapdoor).
MARKER_N = 2
MARKER_X = 3
MARKER_T = 4
DIGIT_LO = 7
DIGIT_HI = 16
MAX_PLACES = 64  # supports numbers up to 63 digits; index 0 = non-digit/marker/pad


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
    """Derive (field_id, place_id) per position from input_ids alone. claude code.

    field_id in {0,1,2,3}: 0 = before N marker (BOS/pad), 1 = N span, 2 = X span,
    3 = T span. Computed by cumulative marker counts, so each marker groups with
    the digits that follow it.

    place_id in {0..MAX_PLACES-1}: 0 for any non-digit/marker/pad position; for a
    digit it is 1 + (distance from the least-significant digit of its span), so the
    units digit of every number shares place_id == 1 regardless of number length.
    """
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
        last_pos = masked_pos.max(dim=1, keepdim=True).values  # units digit position
        place_f = (last_pos - pos_row + 1).clamp_(1, MAX_PLACES - 1)
        place = torch.where(fmask, place_f, place)
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
        # Start structural signals near-zero so init matches the UT baseline. claude code
        nn.init.zeros_(self.segment_embedding.weight)
        nn.init.zeros_(self.place_embedding.weight)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        field, place = _field_and_place(input_ids)
        x = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)
            + self.segment_embedding(field)
            + self.place_embedding(place)
        )
        for k in range(self.num_loops):
            depth = self.depth_embedding(
                torch.tensor(k, device=input_ids.device, dtype=torch.long)
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

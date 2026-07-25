"""Pair/N interaction model with per-column first-square carry supervision."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from benchmark import (
    ModelSpec,
    OptimizerBundle,
    OptimizerSpec,
    Submission,
    assert_model_state,
)


D_MODEL = 64
NUM_HEADS = 4
NUM_LAYERS = 4
N_TOKEN = 2
X_TOKEN = 3
T_TOKEN = 4
DIGIT_OFFSET = 7
AUX_LOSS_WEIGHT = 1.0
CARRY_SCALE = 20.0


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
        self.attention_out = nn.Linear(D_MODEL, D_MODEL)
        self.mlp_norm = RMSNorm(D_MODEL)
        self.up = nn.Linear(D_MODEL, 4 * D_MODEL)
        self.down = nn.Linear(4 * D_MODEL, D_MODEL)

    def forward(self, x: Tensor, attention_mask: Tensor | None) -> Tensor:
        residual = x
        normalized = self.attention_norm(x)
        batch, length, _ = normalized.shape
        q, k, v = self.qkv(normalized).chunk(3, dim=-1)
        q = q.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        k = k.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        v = v.view(batch, length, NUM_HEADS, -1).transpose(1, 2)

        mask = None
        if attention_mask is not None:
            mask = attention_mask[:, None, None, :].bool()
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        attended = attended.transpose(1, 2).contiguous().view(
            batch, length, D_MODEL
        )
        x = residual + self.attention_out(attended)
        return x + self.down(F.gelu(self.up(self.mlp_norm(x))))


def _field_masks(input_ids: Tensor) -> tuple[Tensor, Tensor]:
    is_digit = (input_ids >= DIGIT_OFFSET) & (input_ids < DIGIT_OFFSET + 10)
    after_n = (input_ids == N_TOKEN).cumsum(dim=1) > 0
    before_x = (input_ids == X_TOKEN).cumsum(dim=1) == 0
    after_x = (input_ids == X_TOKEN).cumsum(dim=1) > 0
    before_t = (input_ids == T_TOKEN).cumsum(dim=1) == 0
    return after_n & before_x & is_digit, after_x & before_t & is_digit


@torch.no_grad()
def _square_carry_targets(input_ids: Tensor) -> tuple[Tensor, Tensor]:
    """First-square carry-in/out, aligned LSD-first from the sequence end."""
    _, x_mask = _field_masks(input_ids)
    batch, length = input_ids.shape
    relative_msd = x_mask.long().cumsum(dim=1) - 1
    digit_count = x_mask.long().sum(dim=1)
    relative_lsd = digit_count[:, None] - 1 - relative_msd
    digit_values = torch.where(
        x_mask, input_ids - DIGIT_OFFSET, torch.zeros_like(input_ids)
    )
    digits_lsd = torch.zeros_like(input_ids)
    digits_lsd.scatter_add_(
        1, relative_lsd.clamp(min=0, max=length - 1), digit_values
    )

    column_sums = torch.zeros(
        batch, 2 * length, device=input_ids.device, dtype=torch.long
    )
    for left in range(length):
        for right in range(length):
            column_sums[:, left + right] += (
                digits_lsd[:, left] * digits_lsd[:, right]
            )

    carry = torch.zeros(batch, device=input_ids.device, dtype=torch.long)
    carry_in = torch.zeros_like(column_sums)
    carry_out = torch.zeros_like(column_sums)
    for column in range(2 * length):
        carry_in[:, column] = carry
        carry = torch.div(
            column_sums[:, column] + carry, 10, rounding_mode="floor"
        )
        carry_out[:, column] = carry

    targets = torch.zeros(
        batch, length, 2, device=input_ids.device, dtype=torch.float32
    )
    active = torch.zeros(
        batch, length, device=input_ids.device, dtype=torch.bool
    )
    for column in range(length):
        position = length - 1 - column
        is_active = column < (2 * digit_count)
        active[:, position] = is_active
        targets[:, position, 0] = carry_in[:, column].float() / CARRY_SCALE
        targets[:, position, 1] = carry_out[:, column].float() / CARRY_SCALE
    return active, targets


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.token_embedding = nn.Embedding(spec.vocab_size, D_MODEL)
        self.position_embedding = nn.Embedding(spec.max_seq_len, D_MODEL)
        self.blocks = nn.ModuleList(Block() for _ in range(NUM_LAYERS))
        self.pair_left = nn.Linear(D_MODEL, D_MODEL)
        self.pair_right = nn.Linear(D_MODEL, D_MODEL)
        self.pair_out = nn.Sequential(
            nn.Linear(D_MODEL, 2 * D_MODEL),
            nn.GELU(),
            nn.Linear(2 * D_MODEL, D_MODEL),
        )
        self.mod_query = nn.Linear(D_MODEL, D_MODEL)
        self.mod_key = nn.Linear(D_MODEL, D_MODEL)
        self.mod_value = nn.Linear(D_MODEL, D_MODEL)
        self.context = nn.Sequential(
            nn.Linear(3 * D_MODEL, 2 * D_MODEL),
            nn.GELU(),
            nn.Linear(2 * D_MODEL, D_MODEL),
        )
        self.context_gate = nn.Linear(D_MODEL, D_MODEL)
        self.final_norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.carry_head = nn.Linear(D_MODEL, 2)

    def _pair_n_context(self, hidden: Tensor, input_ids: Tensor) -> Tensor:
        n_mask, x_mask = _field_masks(input_ids)
        left = self.pair_left(hidden)[:, :, None, :]
        right = self.pair_right(hidden)[:, None, :, :]
        pair_states = F.gelu(left + right)
        pair_mask = (x_mask[:, :, None] & x_mask[:, None, :]).unsqueeze(-1)
        pair_weight = pair_mask.to(hidden.dtype)
        pair_pooled = (pair_states * pair_weight).sum(dim=(1, 2))
        pair_pooled = pair_pooled / pair_weight.sum(dim=(1, 2)).clamp_min(1.0)
        pair_pooled = self.pair_out(pair_pooled)

        query = self.mod_query(pair_pooled)
        keys = self.mod_key(hidden)
        values = self.mod_value(hidden)
        scores = torch.einsum("bd,bld->bl", query, keys) / math.sqrt(D_MODEL)
        scores = scores.masked_fill(~n_mask, -1e4)
        weights = F.softmax(scores, dim=-1)
        n_context = torch.einsum("bl,bld->bd", weights, values)
        return self.context(
            torch.cat((pair_pooled, n_context, pair_pooled * n_context), dim=-1)
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden = self.blocks[0](hidden, attention_mask)
        hidden = self.blocks[1](hidden, attention_mask)
        context = self._pair_n_context(hidden, input_ids)
        hidden = hidden + torch.tanh(self.context_gate(context))[:, None, :]
        hidden = self.blocks[2](hidden, attention_mask)
        hidden = self.blocks[3](hidden, attention_mask)
        hidden = self.final_norm(hidden)
        carry_mask, carry_target = _square_carry_targets(input_ids)
        return self.head(hidden), (
            self.carry_head(hidden),
            carry_target,
            carry_mask,
        )


def training_loss(
    loss_logits: Tensor, loss_labels: Tensor, auxiliary: object
) -> Tensor:
    main_loss = F.cross_entropy(loss_logits, loss_labels)
    carry_pred, carry_target, carry_mask = auxiliary  # type: ignore[misc]
    per_slot = (carry_pred - carry_target.to(carry_pred.dtype)).square().mean(-1)
    carry_loss = (per_slot * carry_mask.to(per_slot.dtype)).sum()
    carry_loss = carry_loss / carry_mask.sum().clamp_min(1)
    return main_loss + AUX_LOSS_WEIGHT * carry_loss


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
    estimated_steps = max(1000, int(spec.training_time_seconds * 70))
    warmup = max(1, int(0.05 * estimated_steps))

    def schedule(step: int) -> float:
        if step < warmup:
            return 0.01 + 0.99 * step / warmup
        progress = min((step - warmup) / max(1, estimated_steps - warmup), 1.0)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return OptimizerBundle(
        optimizer,
        torch.optim.lr_scheduler.LambdaLR(optimizer, schedule),
    )


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    batch_size=512,
    eval_batch_size=1024,
)

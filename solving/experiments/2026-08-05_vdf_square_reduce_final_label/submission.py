"""Final-label-only, tied learned Square -> Reduce recurrent VDF candidate."""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from benchmark import ModelSpec, OptimizerBundle, OptimizerSpec, Submission, assert_model_state


PAD, N_MARK, X_MARK, T_MARK, DIGIT_OFFSET = 0, 2, 3, 4, 7
WIDTH, HEADS, MAX_STEPS, MAX_PLACE = 64, 4, 64, 32


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size, self.max_seq_len = vocab_size, max_seq_len


class RMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, value: Tensor) -> Tensor:
        return F.rms_norm(value, (value.shape[-1],), self.weight)


def prompt_layout(ids: Tensor, attention_mask: Tensor | None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Learned-model features only: field, LSD-relative place, T loop count, register."""
    mask = ids.ne(PAD) if attention_mask is None else attention_mask.bool()
    batch, length = ids.shape
    position = torch.arange(length, device=ids.device)[None, :]
    x_mark = ((ids == X_MARK) & mask).long().argmax(-1, keepdim=True).clamp_min(1)
    t_mark = ((ids == T_MARK) & mask).long().argmax(-1, keepdim=True)
    row_end = mask.long().sum(-1, keepdim=True)
    digits = ids.ge(DIGIT_OFFSET) & mask
    n_field = (position > 0) & (position < x_mark) & digits
    x_field = (position > x_mark) & (position < t_mark) & digits
    t_field = (position > t_mark) & (position < row_end) & digits
    field = n_field.long() + 2 * x_field.long() + 3 * t_field.long()
    place = torch.zeros_like(ids)
    for current in (n_field, x_field, t_field):
        from_right = torch.flip(torch.flip(current.long(), (-1,)).cumsum(-1), (-1,)) - 1
        place = torch.where(current, from_right, place)
    value = torch.zeros(batch, dtype=torch.long, device=ids.device)
    raw_t = torch.where(t_field, ids - DIGIT_OFFSET, torch.full_like(ids, -1))
    for index in range(length):
        digit = raw_t[:, index]
        value = torch.where(digit.ge(0), value * 10 + digit.clamp_min(0), value)
    n_width = (x_mark - 1).clamp_min(1)
    register = (position >= row_end - n_width) & (position < row_end)
    return field, place.clamp_max(MAX_PLACE - 1), value.clamp(1, MAX_STEPS), register


def reverse_valid(value: Tensor, attention_mask: Tensor) -> Tensor:
    """Reverse each row's valid prefix while leaving right padding outside the scan."""
    batch, length, width = value.shape
    valid_length = attention_mask.long().sum(-1, keepdim=True)
    position = torch.arange(length, device=value.device)[None, :]
    reverse_position = torch.where(position < valid_length, valid_length - 1 - position, position)
    return value.gather(1, reverse_position[:, :, None].expand(batch, length, width))


class SerialCell(nn.Module):
    """One learned arithmetic phase: attention then an LSD-to-MSD recurrent scan."""
    def __init__(self) -> None:
        super().__init__()
        self.attention_norm, self.mlp_norm = RMSNorm(WIDTH), RMSNorm(WIDTH)
        self.qkv, self.out = nn.Linear(WIDTH, 3 * WIDTH), nn.Linear(WIDTH, WIDTH)
        self.up, self.down = nn.Linear(WIDTH, 3 * WIDTH), nn.Linear(3 * WIDTH, WIDTH)
        self.scan = nn.GRU(WIDTH, WIDTH, batch_first=True)

    def forward(self, value: Tensor, attention_mask: Tensor) -> Tensor:
        residual, normed = value, self.attention_norm(value)
        batch, length, _ = normed.shape
        query, key, val = self.qkv(normed).chunk(3, dim=-1)
        query = query.view(batch, length, HEADS, -1).transpose(1, 2)
        key = key.view(batch, length, HEADS, -1).transpose(1, 2)
        val = val.view(batch, length, HEADS, -1).transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, key, val, attn_mask=attention_mask[:, None, None, :])
        mixed = residual + self.out(attended.transpose(1, 2).reshape(batch, length, WIDTH))
        mixed = mixed + self.down(F.silu(self.up(self.mlp_norm(mixed))))
        reversed_mixed = reverse_valid(mixed, attention_mask)
        reversed_scans, _ = self.scan(reversed_mixed)
        scans = reverse_valid(reversed_scans, attention_mask)
        return mixed + scans * attention_mask[:, :, None]


class VDFModel(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.token = nn.Embedding(spec.vocab_size, WIDTH)
        self.position = nn.Embedding(spec.max_seq_len, WIDTH)
        self.field, self.place = nn.Embedding(4, WIDTH), nn.Embedding(MAX_PLACE, WIDTH)
        self.register_projection = nn.Linear(spec.vocab_size, WIDTH, bias=False)
        self.square, self.reduce = SerialCell(), SerialCell()
        self.norm = RMSNorm(WIDTH)
        self.head = nn.Linear(WIDTH, spec.vocab_size, bias=False)
        self.head.weight = self.token.weight

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> tuple[Tensor, None]:
        batch, length = input_ids.shape
        mask = input_ids.ne(PAD) if attention_mask is None else attention_mask.bool()
        field, place, steps, register = prompt_layout(input_ids, mask)
        base = self.token(input_ids) + self.position(torch.arange(length, device=input_ids.device)) + self.field(field) + self.place(place)
        state = torch.zeros(batch, length, self.config.vocab_size, device=input_ids.device, dtype=base.dtype)
        hidden = base
        for depth in range(int(steps.max().item())):
            indices = torch.nonzero(steps > depth, as_tuple=True)[0]
            active_base = base.index_select(0, indices)
            active_state = state.index_select(0, indices)
            active_mask = mask.index_select(0, indices)
            active_register = register.index_select(0, indices)
            squared = self.square(active_base + self.register_projection(active_state), active_mask)
            reduced = self.reduce(active_base + squared, active_mask)
            logits = self.head(self.norm(reduced))
            soft = logits.softmax(-1)
            hard = F.one_hot(logits.argmax(-1), self.config.vocab_size).to(soft.dtype)
            next_state = hard + soft - soft.detach() if self.training else hard
            next_state = torch.where(active_register[:, :, None], next_state, active_state)
            state = state.index_copy(0, indices, next_state)
            hidden = hidden.index_copy(0, indices, reduced)
        return self.head(self.norm(hidden)), None


class Schedule:
    def __init__(self, optimizer: torch.optim.Optimizer, seconds: float) -> None:
        self.optimizer, self.start, self.seconds = optimizer, time.monotonic(), max(1.0, float(seconds))
        self.base = [group["lr"] for group in optimizer.param_groups]

    def step(self) -> None:
        fraction = min(1.0, (time.monotonic() - self.start) / self.seconds)
        scale = 0.08 + 0.92 * 0.5 * (1.0 + math.cos(math.pi * fraction))
        for group, rate in zip(self.optimizer.param_groups, self.base):
            group["lr"] = rate * scale


def build_model(spec: ModelSpec) -> VDFModel:
    model = VDFModel(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, betas=(0.9, 0.95), weight_decay=0.05, capturable=spec.device_type == "cuda")
    return OptimizerBundle(optimizer, scheduler=Schedule(optimizer, spec.training_time_seconds))


SUBMISSION = Submission(build_model=build_model, build_optimizer=build_optimizer, batch_size=512, eval_batch_size=512)

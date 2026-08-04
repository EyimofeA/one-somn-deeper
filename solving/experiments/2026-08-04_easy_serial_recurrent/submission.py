"""Compact Easy candidate: learned LSD-first tied recurrent arithmetic.

The model receives only evaluator prompt tokens (N, x, T).  It uses no
precomputed square, modular reduction, factorization, lookup, or arithmetic
helper.  T is used solely to choose how many times to apply the shared learned
transition cell.
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from benchmark import ModelSpec, OptimizerBundle, OptimizerSpec, Submission, assert_model_state


PAD, N_MARK, X_MARK, T_MARK, DIGIT_OFFSET = 0, 2, 3, 4, 7
D_MODEL, HEADS, MAX_T, MAX_PLACE = 32, 4, 4, 16


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int):
        self.vocab_size, self.max_seq_len = vocab_size, max_seq_len


class RMSNorm(nn.Module):
    def __init__(self, width: int):
        super().__init__(); self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight)


def prompt_features(ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Field and right-relative decimal place, plus T used only as loop count."""
    seen_n, seen_x, seen_t = (ids == N_MARK).cumsum(-1), (ids == X_MARK).cumsum(-1), (ids == T_MARK).cumsum(-1)
    field = (seen_n + seen_x + seen_t).clamp(max=3)
    digit = ids >= DIGIT_OFFSET
    place = torch.zeros_like(ids)
    for current in (1, 2, 3):
        mask = (field == current) & digit
        from_right = torch.flip(torch.flip(mask.long(), (-1,)).cumsum(-1), (-1,)) - 1
        place = torch.where(mask, from_right, place)
    t_digits = torch.where((field == 3) & digit, ids - DIGIT_OFFSET, torch.full_like(ids, -1))
    steps = torch.zeros(ids.shape[0], dtype=torch.long, device=ids.device)
    for position in range(ids.shape[1]):
        value = t_digits[:, position]
        steps = torch.where(value >= 0, steps * 10 + value.clamp_min(0), steps)
    return field, place.clamp(max=MAX_PLACE - 1), steps.clamp(min=1, max=MAX_T)


class TiedArithmeticCell(nn.Module):
    """One learned bidirectional interaction followed by an LSD-to-MSD scan."""
    def __init__(self):
        super().__init__()
        self.norm_a, self.norm_m = RMSNorm(D_MODEL), RMSNorm(D_MODEL)
        self.qkv, self.out = nn.Linear(D_MODEL, 3 * D_MODEL), nn.Linear(D_MODEL, D_MODEL)
        self.up, self.down = nn.Linear(D_MODEL, 4 * D_MODEL), nn.Linear(4 * D_MODEL, D_MODEL)
        self.scan = nn.GRUCell(D_MODEL, D_MODEL)

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        residual, normalized = x, self.norm_a(x)
        batch, length, _ = normalized.shape
        q, k, v = self.qkv(normalized).chunk(3, dim=-1)
        q = q.view(batch, length, HEADS, -1).transpose(1, 2)
        k = k.view(batch, length, HEADS, -1).transpose(1, 2)
        v = v.view(batch, length, HEADS, -1).transpose(1, 2)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = residual + self.out(attended.transpose(1, 2).reshape(batch, length, D_MODEL))
        x = x + self.down(F.gelu(self.up(self.norm_m(x))))
        state = torch.zeros(batch, D_MODEL, device=x.device, dtype=x.dtype)
        scanned = []
        for position in range(length - 1, -1, -1):
            state = self.scan(x[:, position], state)
            scanned.append(state)
        return x + torch.stack(list(reversed(scanned)), dim=1)


class Model(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.token = nn.Embedding(spec.vocab_size, D_MODEL)
        self.position = nn.Embedding(spec.max_seq_len, D_MODEL)
        self.field = nn.Embedding(4, D_MODEL)
        self.place = nn.Embedding(MAX_PLACE, D_MODEL)
        self.state_projection = nn.Linear(spec.vocab_size, D_MODEL, bias=False)
        self.cell, self.norm = TiedArithmeticCell(), RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.head.weight = self.token.weight

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> tuple[Tensor, None]:
        batch, length = input_ids.shape
        field, place, steps = prompt_features(input_ids)
        positions = torch.arange(length, device=input_ids.device)
        base = self.token(input_ids) + self.position(positions) + self.field(field) + self.place(place)
        mask = attention_mask[:, None, None, :].bool() if attention_mask is not None and attention_mask.ndim == 2 else attention_mask
        state = torch.zeros(batch, length, self.config.vocab_size, device=input_ids.device, dtype=base.dtype)
        hidden = base
        for depth in range(MAX_T):
            candidate = self.cell(base + self.state_projection(state), mask)
            logits = self.head(self.norm(candidate))
            soft = logits.softmax(-1)
            hard = F.one_hot(logits.argmax(-1), self.config.vocab_size).to(soft.dtype)
            next_state = hard + soft - soft.detach() if self.training else hard
            active = (depth < steps)[:, None, None]
            state = torch.where(active, next_state, state)
            hidden = torch.where(active, candidate, hidden)
        return self.head(self.norm(hidden)), None


class Schedule:
    def __init__(self, optimizer, seconds):
        self.optimizer, self.started = optimizer, time.monotonic()
        self.seconds, self.base = max(1.0, float(seconds)), [group["lr"] for group in optimizer.param_groups]

    def step(self):
        progress = min(1.0, (time.monotonic() - self.started) / self.seconds)
        scale = 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))
        for group, lr in zip(self.optimizer.param_groups, self.base): group["lr"] = lr * scale


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec); assert_model_state(model, spec); return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(.9, .95), weight_decay=.1, capturable=spec.device_type == "cuda")
    return OptimizerBundle(optimizer, scheduler=Schedule(optimizer, spec.training_time_seconds))


SUBMISSION = Submission(build_model=build_model, build_optimizer=build_optimizer, batch_size=512, eval_batch_size=512)

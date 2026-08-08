"""T-independent recurrent transition with a genuine T=1 curriculum.

The transition never receives the requested T value. T controls only how many
times the same learned cell executes. The first half of training uses only
T=1 final labels and ordinary cross-entropy; no intermediate arithmetic label,
oracle, lookup, or hard-coded numeric operation is used.
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
    TokenLossBatch,
    assert_model_state,
)

PAD, BOS, N_MARK, X_MARK, T_MARK, ANS_MARK, EOS = 0, 1, 2, 3, 4, 5, 6
DIGIT_OFFSET = 7

D_MODEL = 256
NUM_HEADS = 4
STEP_LAYERS = 2
MAX_LOOPS = 64
TRAIN_LOOP_CAP = 16
INIT_SCALE = 0.4
WARMUP_FRACTION = 0.05
FINAL_LR_FRACTION = 0.01

# These three constants define the registered ablation arms. The committed
# source is the proposed treatment and the only version eligible for upload.
HIDE_T_FROM_TRANSITION = True
T1_ONLY_FRACTION = 0.50
T1_LATE_WEIGHT = 4.0

_training_started = 0.0
_training_total_seconds = 1.0


def _training_progress() -> float:
    return min(max((time.monotonic() - _training_started) / _training_total_seconds, 0.0), 1.0)


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

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        residual = x
        x = self.attention_norm(x)
        b, length, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, length, NUM_HEADS, -1).transpose(1, 2)
        k = k.view(b, length, NUM_HEADS, -1).transpose(1, 2)
        v = v.view(b, length, NUM_HEADS, -1).transpose(1, 2)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x.transpose(1, 2).contiguous().view(b, length, D_MODEL)
        x = residual + self.out(x)
        return x + self.down(F.gelu(self.up(self.mixer_norm(x))))


class StepBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(Block() for _ in range(STEP_LAYERS))

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


def _derived_features(input_ids: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    ids = input_ids
    is_n = (ids == N_MARK).cumsum(-1)
    is_x = (ids == X_MARK).cumsum(-1)
    is_t = (ids == T_MARK).cumsum(-1)
    is_ans = (ids == ANS_MARK).cumsum(-1)
    field = (is_n + is_x + is_t).clamp(max=3)
    is_digit = ids >= DIGIT_OFFSET
    place = torch.zeros_like(ids)
    for field_id in (1, 2, 3):
        selected = (field == field_id) & is_digit & (is_ans == 0)
        reverse_count = torch.flip(
            torch.flip(selected.long(), dims=[-1]).cumsum(-1), dims=[-1]
        )
        place = place + torch.where(selected, reverse_count - 1, torch.zeros_like(reverse_count))
    place = place.clamp(max=15)
    t_region = (is_t > 0) & (is_ans == 0)
    t_digits = torch.where(
        t_region & is_digit, ids - DIGIT_OFFSET, torch.full_like(ids, -1)
    )
    t_value = torch.zeros(ids.shape[0], dtype=torch.long, device=ids.device)
    for position in range(ids.shape[1]):
        digit = t_digits[:, position]
        keep = digit >= 0
        t_value = torch.where(keep, t_value * 10 + digit.clamp(min=0), t_value)
    return field, place, t_value.clamp(min=1, max=MAX_LOOPS), t_region


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.token_embedding = nn.Embedding(spec.vocab_size, D_MODEL)
        self.position_embedding = nn.Embedding(spec.max_seq_len, D_MODEL)
        self.field_embedding = nn.Embedding(4, D_MODEL)
        self.place_embedding = nn.Embedding(16, D_MODEL)
        self.step = StepBlock()
        self.state_proj = nn.Linear(spec.vocab_size, D_MODEL, bias=False)
        self.final_norm = RMSNorm(D_MODEL)
        self.head_fwd = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.head_rev = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.fuse_gate = nn.Linear(D_MODEL, 1)
        self.head_fwd.weight = self.token_embedding.weight
        self.auxiliary: object | None = None
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.head_fwd:
                nn.init.normal_(module.weight, std=INIT_SCALE * module.weight.shape[1] ** -0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _quantize(logits: Tensor) -> Tensor:
        hard = F.one_hot(logits.argmax(-1), logits.shape[-1]).to(logits.dtype)
        soft = logits.softmax(-1)
        return hard + (soft - soft.detach())

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> tuple[Tensor, object | None]:
        batch_size, length = input_ids.shape
        field, place, t_value, t_region = _derived_features(input_ids)
        in_t1_phase = self.training and _training_progress() < T1_ONLY_FRACTION
        if in_t1_phase:
            effective_t = torch.ones_like(t_value)
        elif self.training:
            effective_t = t_value.clamp(max=TRAIN_LOOP_CAP)
        else:
            effective_t = t_value

        positions = torch.arange(length, device=input_ids.device)
        base = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)
            + self.field_embedding(field)
            + self.place_embedding(place)
        )
        if HIDE_T_FROM_TRANSITION:
            base = torch.where(t_region[:, :, None], torch.zeros_like(base), base)
        mask = None
        if attention_mask is not None:
            mask = attention_mask[:, None, None, :].to(torch.bool)

        maximum_t = int(effective_t.max().item())
        detach_prefix = (
            int(torch.randint(0, maximum_t, ()).item())
            if self.training and maximum_t > 1
            else 0
        )
        state = torch.zeros(
            batch_size,
            length,
            self.config.vocab_size,
            dtype=base.dtype,
            device=base.device,
        )
        hidden = base
        for step_index in range(maximum_t):
            hidden = self.step(base + self.state_proj(state), mask)
            step_logits = self.head_fwd(self.final_norm(hidden))
            proposed_state = self._quantize(step_logits)
            active = (step_index < effective_t).view(batch_size, 1, 1).to(proposed_state.dtype)
            state = active * proposed_state + (1.0 - active) * state
            if self.training and step_index < detach_prefix:
                state = state.detach()

        hidden = self.final_norm(hidden)
        logits_fwd = self.head_fwd(hidden)
        logits_rev = torch.flip(self.head_rev(torch.flip(hidden, dims=[1])), dims=[1])
        gate = torch.sigmoid(self.fuse_gate(hidden))
        logits = gate * logits_fwd + (1.0 - gate) * logits_rev
        self.auxiliary = {"t_value": t_value}
        return logits, self.auxiliary


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    token_ce = F.cross_entropy(
        batch.logits.transpose(1, 2),
        batch.labels,
        ignore_index=-100,
        reduction="none",
    )
    valid = batch.valid_mask.to(token_ce.dtype)
    row_ce = (token_ce * valid).sum(1) / valid.sum(1).clamp_min(1.0)
    auxiliary = batch.auxiliary
    if not isinstance(auxiliary, dict) or "t_value" not in auxiliary:
        return row_ce.mean()
    is_t1 = (auxiliary["t_value"] == 1).to(row_ce.dtype)
    if _training_progress() < T1_ONLY_FRACTION:
        # Zero remains connected to logits if an unusual batch has no T=1 row.
        return (row_ce * is_t1).sum() / is_t1.sum().clamp_min(1.0)
    weights = 1.0 + (T1_LATE_WEIGHT - 1.0) * is_t1
    return (row_ce * weights).sum() / weights.sum().clamp_min(1.0)


class WallClockSchedule:
    def __init__(self, optimizer: torch.optim.Optimizer, total_seconds: float) -> None:
        self.optimizer = optimizer
        self.total_seconds = max(1.0, float(total_seconds))
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.started = time.monotonic()

    def step(self) -> None:
        progress = min(max((time.monotonic() - self.started) / self.total_seconds, 0.0), 1.0)
        if progress < WARMUP_FRACTION:
            factor = FINAL_LR_FRACTION + (1.0 - FINAL_LR_FRACTION) * progress / WARMUP_FRACTION
        else:
            tail = (progress - WARMUP_FRACTION) / (1.0 - WARMUP_FRACTION)
            factor = FINAL_LR_FRACTION + (1.0 - FINAL_LR_FRACTION) * 0.5 * (
                1.0 + math.cos(math.pi * tail)
            )
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * factor


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    global _training_started, _training_total_seconds
    _training_started = time.monotonic()
    _training_total_seconds = max(1.0, float(spec.training_time_seconds))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-3,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        capturable=spec.device_type == "cuda",
    )
    return OptimizerBundle(optimizer, WallClockSchedule(optimizer, spec.training_time_seconds))


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    token_training_loss=token_training_loss,
    batch_size=512,
    eval_batch_size=1024,
)

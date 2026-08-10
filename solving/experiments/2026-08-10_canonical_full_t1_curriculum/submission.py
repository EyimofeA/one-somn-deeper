"""Canonical T-independent modular-state recurrence.

The mutable register is initialized from x once. Every subsequent application
of the tied cell receives only the current LSD-first digit state and immutable
N digits. Requested T controls only the number of applications. The state
logits are also the answer logits, preventing a separate output shortcut.

Training uses only evaluator-provided final labels with ordinary cross-entropy.
No arithmetic trace, oracle, answer lookup, or hard-coded numeric operation is
used by the model or loss.
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

N_MARK, X_MARK, T_MARK = 2, 3, 4
DIGIT_OFFSET = 7

D_MODEL = 256
NUM_HEADS = 4
STEP_LAYERS = 2
MAX_DIGITS = 16
MAX_LOOPS = 64
TRAIN_LOOP_CAP = 16
INIT_SCALE = 0.4
WARMUP_FRACTION = 0.05
FINAL_LR_FRACTION = 0.01

T1_ONLY_FRACTION = 1.00
T1_LATE_WEIGHT = 4.0

_training_started = 0.0
_training_total_seconds = 1.0


def _training_progress() -> float:
    elapsed = time.monotonic() - _training_started
    return min(max(elapsed / _training_total_seconds, 0.0), 1.0)


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

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        residual = x
        h = self.attention_norm(x)
        batch, length, _ = h.shape
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        k = k.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        v = v.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        h = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        h = h.transpose(1, 2).contiguous().view(batch, length, D_MODEL)
        x = residual + self.out(h)
        return x + self.down(F.gelu(self.up(self.mixer_norm(x))))


class StepBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(Block() for _ in range(STEP_LAYERS))

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


def _parse_prompt(
    input_ids: Tensor, num_digits: int
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return packed N/x tokens, valid N slots, and requested T.

    Packing only routes observed digit tokens into fixed LSD-first slots. It
    performs no arithmetic and creates no training label.
    """

    ids = input_ids
    is_n = (ids == N_MARK).cumsum(-1)
    is_x = (ids == X_MARK).cumsum(-1)
    is_t = (ids == T_MARK).cumsum(-1)
    field = (is_n + is_x + is_t).clamp(max=3)
    is_digit = ids >= DIGIT_OFFSET

    place = torch.zeros_like(ids)
    for field_id in (1, 2, 3):
        selected = (field == field_id) & is_digit
        reverse_count = torch.flip(
            torch.flip(selected.long(), dims=[-1]).cumsum(-1), dims=[-1]
        )
        place = place + torch.where(
            selected, reverse_count - 1, torch.zeros_like(reverse_count)
        )
    place = place.clamp(max=num_digits - 1)

    batch = ids.shape[0]
    zero_token = torch.full(
        (batch, num_digits), DIGIT_OFFSET, dtype=torch.long, device=ids.device
    )
    n_tokens = zero_token.clone()
    x_tokens = zero_token.clone()
    n_valid = torch.zeros(
        batch, num_digits, dtype=torch.bool, device=ids.device
    )

    for position in range(ids.shape[1]):
        slot = place[:, position : position + 1]
        token = ids[:, position : position + 1]
        for field_id, packed, valid_slots in (
            (1, n_tokens, n_valid),
            (2, x_tokens, None),
        ):
            selected = ((field[:, position] == field_id) & is_digit[:, position])[:, None]
            previous = packed.gather(1, slot)
            packed.scatter_(1, slot, torch.where(selected, token, previous))
            if valid_slots is not None:
                was_valid = valid_slots.gather(1, slot)
                valid_slots.scatter_(1, slot, selected | was_valid)

    t_digits = torch.where(
        (field == 3) & is_digit,
        ids - DIGIT_OFFSET,
        torch.full_like(ids, -1),
    )
    t_value = torch.zeros(batch, dtype=torch.long, device=ids.device)
    for position in range(ids.shape[1]):
        digit = t_digits[:, position]
        keep = digit >= 0
        t_value = torch.where(keep, t_value * 10 + digit.clamp(min=0), t_value)

    return n_tokens, x_tokens, n_valid, t_value.clamp(min=1, max=MAX_LOOPS)


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        # A prompt is N digits + x digits + three markers (+ T digits). This
        # conservative bound avoids paying quadratic attention for 16 slots on
        # small suites while automatically expanding for larger evaluator specs.
        self.num_digits = min(MAX_DIGITS, max(4, (spec.max_seq_len - 3) // 2))
        self.token_embedding = nn.Embedding(spec.vocab_size, D_MODEL)
        self.place_embedding = nn.Embedding(MAX_DIGITS, D_MODEL)
        self.role_embedding = nn.Embedding(2, D_MODEL)
        self.state_proj = nn.Linear(spec.vocab_size, D_MODEL, bias=False)
        self.step = StepBlock()
        self.final_norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight
        self.auxiliary: object | None = None

        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(
                    module.weight,
                    std=INIT_SCALE * D_MODEL ** -0.5,
                )
            elif isinstance(module, nn.Linear) and module is not self.head:
                nn.init.normal_(
                    module.weight,
                    std=INIT_SCALE * module.weight.shape[1] ** -0.5,
                )
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
        del attention_mask
        batch, length = input_ids.shape
        n_tokens, x_tokens, n_valid, t_value = _parse_prompt(
            input_ids, self.num_digits
        )
        batch_has_t1 = bool((t_value == 1).any().item())
        in_t1_phase = (
            self.training
            and batch_has_t1
            and _training_progress() < T1_ONLY_FRACTION
        )
        if in_t1_phase:
            effective_t = torch.ones_like(t_value)
        elif self.training:
            effective_t = t_value.clamp(max=TRAIN_LOOP_CAP)
        else:
            effective_t = t_value

        places = torch.arange(self.num_digits, device=input_ids.device)
        place_features = self.place_embedding(places)[None, :, :]
        n_context = (
            self.token_embedding(n_tokens)
            + place_features
            + self.role_embedding.weight[1][None, None, :]
        )
        state = F.one_hot(x_tokens, self.config.vocab_size).to(n_context.dtype)

        # All mutable state slots are valid. Only observed N slots are exposed.
        key_valid = torch.cat(
            [
                torch.ones(
                    batch,
                    self.num_digits,
                    dtype=torch.bool,
                    device=input_ids.device,
                ),
                n_valid,
            ],
            dim=1,
        )
        attention = key_valid[:, None, None, :]
        maximum_t = int(effective_t.max().item())
        output_state_logits = torch.zeros(
            batch,
            self.num_digits,
            self.config.vocab_size,
            dtype=n_context.dtype,
            device=input_ids.device,
        )

        for step_index in range(maximum_t):
            mutable = (
                self.state_proj(state)
                + place_features
                + self.role_embedding.weight[0][None, None, :]
            )
            joined = torch.cat([mutable, n_context], dim=1)
            hidden = self.step(joined, attention)[:, : self.num_digits, :]
            proposed_logits = self.head(self.final_norm(hidden))
            proposed_state = self._quantize(proposed_logits)
            active = (step_index < effective_t)[:, None, None]
            state = torch.where(active, proposed_state, state)
            output_state_logits = torch.where(
                active, proposed_logits, output_state_logits
            )

        # Separate-output labels occupy the prompt tail in MSD-to-LSD order.
        # Mapping each sequence position by distance from the end makes the
        # output logits identical to the corresponding recurrent state digit.
        output_places = (length - 1 - torch.arange(length, device=input_ids.device)).clamp(
            max=self.num_digits - 1
        )
        logits = output_state_logits[:, output_places, :]
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
    if _training_progress() < T1_ONLY_FRACTION and bool((is_t1 > 0).any().item()):
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
        progress = min(
            max((time.monotonic() - self.started) / self.total_seconds, 0.0), 1.0
        )
        if progress < WARMUP_FRACTION:
            factor = FINAL_LR_FRACTION + (
                (1.0 - FINAL_LR_FRACTION) * progress / WARMUP_FRACTION
            )
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
    return OptimizerBundle(
        optimizer, WallClockSchedule(optimizer, spec.training_time_seconds)
    )


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    token_training_loss=token_training_loss,
    batch_size=512,
    eval_batch_size=1024,
)

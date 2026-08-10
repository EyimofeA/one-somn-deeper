"""Competition-legal multi-lane local recurrent grid.

The model uses generic learned scratch lanes and tied local updates. It has no
arithmetic trace, carry target, task solver, or hard-coded numeric transition.
Training uses only evaluator-provided final labels.
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

FEATURE_DIM = 32
STATE_DIM = 64
LANES = 6
MICROSTEPS = 4
MAX_DIGITS = 16
MAX_LOOPS = 64
TRAIN_LOOP_CAP = 16
INIT_SCALE = 0.4
WARMUP_FRACTION = 0.05
FINAL_LR_FRACTION = 0.01
T1_ONLY_FRACTION = 0.50
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


def _parse_prompt(
    input_ids: Tensor, num_digits: int
) -> tuple[Tensor, Tensor, Tensor]:
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
    for position in range(ids.shape[1]):
        slot = place[:, position : position + 1]
        token = ids[:, position : position + 1]
        for field_id, packed in ((1, n_tokens), (2, x_tokens)):
            selected = ((field[:, position] == field_id) & is_digit[:, position])[:, None]
            previous = packed.gather(1, slot)
            packed.scatter_(1, slot, torch.where(selected, token, previous))

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
    return n_tokens, x_tokens, t_value.clamp(min=1, max=MAX_LOOPS)


class LocalGridCell(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Linear(STATE_DIM, STATE_DIM, bias=False)
        self.self_ = nn.Linear(STATE_DIM, STATE_DIM, bias=False)
        self.right = nn.Linear(STATE_DIM, STATE_DIM, bias=False)
        self.lane_mix = nn.Linear(LANES, LANES, bias=False)
        self.norm = nn.LayerNorm(STATE_DIM)
        self.update = nn.GRUCell(STATE_DIM, STATE_DIM)

    def forward(self, state: Tensor, immutable: Tensor) -> Tensor:
        zero = torch.zeros_like(state[:, :1])
        left = torch.cat([zero, state[:, :-1]], dim=1)
        right = torch.cat([state[:, 1:], zero], dim=1)
        local = self.left(left) + self.self_(state) + self.right(right)
        lanes = self.lane_mix(state.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        proposal = self.norm(local + lanes + immutable)
        shape = state.shape
        return self.update(
            proposal.reshape(-1, STATE_DIM), state.reshape(-1, STATE_DIM)
        ).reshape(shape)


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.num_digits = min(MAX_DIGITS, max(4, (spec.max_seq_len - 3) // 2))
        self.token_embedding = nn.Embedding(spec.vocab_size, FEATURE_DIM)
        self.n_inject = nn.Linear(FEATURE_DIM, LANES * STATE_DIM)
        self.x_inject = nn.Linear(FEATURE_DIM, LANES * STATE_DIM)
        self.lane_roles = nn.Parameter(torch.empty(LANES, STATE_DIM))
        self.boundaries = nn.Parameter(torch.empty(2, LANES, STATE_DIM))
        self.cell = LocalGridCell()
        self.output_norm = nn.LayerNorm(STATE_DIM)
        self.head = nn.Linear(STATE_DIM, spec.vocab_size)
        self.auxiliary: object | None = None

        nn.init.normal_(self.lane_roles, std=0.02)
        nn.init.normal_(self.boundaries, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=INIT_SCALE * FEATURE_DIM ** -0.5)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(
                    module.weight,
                    std=INIT_SCALE * math.prod(module.weight.shape[1:]) ** -0.5,
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _quantize(logits: Tensor) -> Tensor:
        hard = F.one_hot(logits.argmax(-1), logits.shape[-1]).to(logits.dtype)
        soft = logits.softmax(-1)
        return hard + (soft - soft.detach())

    def _macrostep(self, digit_state: Tensor, n_features: Tensor) -> Tensor:
        batch, positions, _ = digit_state.shape
        x_features = digit_state @ self.token_embedding.weight
        x = self.x_inject(x_features).view(batch, positions, LANES, STATE_DIM)
        n = self.n_inject(n_features).view(batch, positions, LANES, STATE_DIM)
        immutable = x + n + self.lane_roles[None, None]
        immutable = immutable.clone()
        immutable[:, 0] = immutable[:, 0] + self.boundaries[0]
        immutable[:, -1] = immutable[:, -1] + self.boundaries[1]
        scratch = torch.zeros_like(immutable)
        for _ in range(MICROSTEPS):
            scratch = self.cell(scratch, immutable)
        return self.head(self.output_norm(scratch[:, :, 0]))

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> tuple[Tensor, object | None]:
        del attention_mask
        batch, length = input_ids.shape
        n_tokens, x_tokens, t_value = _parse_prompt(input_ids, self.num_digits)
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

        n_features = self.token_embedding(n_tokens)
        digit_state = F.one_hot(x_tokens, self.config.vocab_size).to(n_features.dtype)
        output_logits = torch.zeros(
            batch,
            self.num_digits,
            self.config.vocab_size,
            dtype=n_features.dtype,
            device=input_ids.device,
        )
        for step_index in range(int(effective_t.max().item())):
            proposed_logits = self._macrostep(digit_state, n_features)
            proposed_state = self._quantize(proposed_logits)
            active = (step_index < effective_t)[:, None, None]
            digit_state = torch.where(active, proposed_state, digit_state)
            output_logits = torch.where(active, proposed_logits, output_logits)

        output_places = (length - 1 - torch.arange(length, device=input_ids.device)).clamp(
            max=self.num_digits - 1
        )
        logits = output_logits[:, output_places, :]
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

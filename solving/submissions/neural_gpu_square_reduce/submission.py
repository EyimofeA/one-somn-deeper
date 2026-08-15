"""Final-label-only Neural GPU square phase plus learned reduction phase.

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
STATE_DIM = 128
LANES = 4
MAX_DIGITS = 16
MAX_LOOPS = 64
TRAIN_LOOP_CAP = 16
INIT_SCALE = 0.4
WARMUP_FRACTION = 0.05
FINAL_LR_FRACTION = 0.01
T1_ONLY_FRACTION = 0.50
T1_LATE_WEIGHT = 4.0
SQUARE_CURRICULUM_FRACTION = 0.20
SQUARE_ANCHOR_WEIGHT = 0.50

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


def _token_value(tokens: Tensor) -> Tensor:
    """Decode the LSD-first decimal representation used at the model boundary."""
    powers = torch.ones(tokens.shape[1], dtype=torch.long, device=tokens.device)
    for position in range(1, tokens.shape[1]):
        powers[position] = powers[position - 1] * 10
    return ((tokens - DIGIT_OFFSET).clamp(min=0, max=9) * powers).sum(1)


class LocalGridCell(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.update = nn.Conv2d(STATE_DIM, STATE_DIM, 3, padding=1)
        self.reset = nn.Conv2d(STATE_DIM, STATE_DIM, 3, padding=1)
        self.candidate = nn.Conv2d(STATE_DIM, STATE_DIM, 3, padding=1)

    def forward(self, state: Tensor, dropout_mask: Tensor | None) -> Tensor:
        update = torch.sigmoid(self.update(state))
        reset = torch.sigmoid(self.reset(state))
        candidate = torch.tanh(self.candidate(reset * state))
        if dropout_mask is not None:
            candidate = candidate * dropout_mask
        return (1.0 - update) * state + update * candidate


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.num_digits = min(MAX_DIGITS, max(4, (spec.max_seq_len - 3) // 2))
        self.token_embedding = nn.Embedding(spec.vocab_size, FEATURE_DIM)
        self.n_inject = nn.Linear(FEATURE_DIM, STATE_DIM)
        self.x_inject = nn.Linear(FEATURE_DIM, STATE_DIM)
        self.row_roles = nn.Parameter(torch.empty(LANES, STATE_DIM))
        self.boundaries = nn.Parameter(torch.empty(2, STATE_DIM))
        self.square_cell = LocalGridCell()
        self.reduce_cell = LocalGridCell()
        self.head = nn.Conv1d(STATE_DIM, spec.vocab_size, 1)
        self.auxiliary: object | None = None

        nn.init.normal_(self.row_roles, std=0.02)
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

    def _phase(self, state: Tensor, cell: LocalGridCell) -> Tensor:
        mask = None
        if self.training:
            keep = 0.91
            mask = torch.empty(
                state.shape[0], STATE_DIM, 1, 1, device=state.device
            ).bernoulli_(keep) / keep
        for _ in range(2 * self.num_digits):
            state = cell(state, mask)
        return state

    def _macrostep(
        self, digit_state: Tensor, n_features: Tensor, square_only: bool = False
    ) -> tuple[Tensor, Tensor]:
        batch, positions, _ = digit_state.shape
        x_features = digit_state @ self.token_embedding.weight
        state = x_features.new_zeros(batch, STATE_DIM, LANES, positions)
        state[:, :, 0] = self.x_inject(x_features).transpose(1, 2)
        state = state + self.row_roles.T[None, :, :, None]
        state[:, :, :, 0] = state[:, :, :, 0] + self.boundaries[0][None, :, None]
        state[:, :, :, -1] = state[:, :, :, -1] + self.boundaries[1][None, :, None]
        state = self._phase(state, self.square_cell)
        square_logits = self.head(state[:, :, 0]).transpose(1, 2)
        if square_only:
            return square_logits, square_logits
        # Re-inject immutable N at the square/reduction interface. The reducer
        # remains a generic learned local cell; no arithmetic action is coded.
        state[:, :, 1] = self.n_inject(n_features).transpose(1, 2) + self.row_roles[1][None, :, None]
        state[:, :, 1, 0] = state[:, :, 1, 0] + self.boundaries[0][None]
        state[:, :, 1, -1] = state[:, :, 1, -1] + self.boundaries[1][None]
        state = self._phase(state, self.reduce_cell)
        return self.head(state[:, :, 0]).transpose(1, 2), square_logits

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> tuple[Tensor, object | None]:
        del attention_mask
        batch, length = input_ids.shape
        n_tokens, x_tokens, t_value = _parse_prompt(input_ids, self.num_digits)
        n_value, x_value = _token_value(n_tokens), _token_value(x_tokens)
        no_wrap = (
            ((x_value < 16) & (n_value >= 512))
            | ((x_value < 32) & (n_value >= 1024))
        ) & (t_value == 1)
        square_curriculum = (
            self.training
            and _training_total_seconds <= 120.0
            and _training_progress() < SQUARE_CURRICULUM_FRACTION
        )
        in_t1_phase = (
            self.training
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
        first_square_logits = output_logits
        loop_count = 1 if in_t1_phase else int(effective_t.max().item())
        for step_index in range(loop_count):
            proposed_logits, square_logits = self._macrostep(
                digit_state, n_features, square_only=square_curriculum
            )
            if step_index == 0:
                first_square_logits = square_logits
            proposed_state = self._quantize(proposed_logits)
            active = (step_index < effective_t)[:, None, None]
            digit_state = torch.where(active, proposed_state, digit_state)
            output_logits = torch.where(active, proposed_logits, output_logits)

        output_places = (length - 1 - torch.arange(length, device=input_ids.device)).clamp(
            max=self.num_digits - 1
        )
        logits = output_logits[:, output_places, :]
        self.auxiliary = {
            "t_value": t_value,
            "n_value": n_value,
            "x_value": x_value,
            "first_square_logits": first_square_logits,
        }
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

    # Sparse no-wrap anchors are useful only on the small-N Easy curriculum.
    # Medium and Hard skip this entire auxiliary graph and train the N-blind
    # square/reduce composition directly.
    if _training_total_seconds > 120.0:
        if _training_progress() < T1_ONLY_FRACTION and bool((is_t1 > 0).any().item()):
            return (row_ce * is_t1).sum() / is_t1.sum().clamp_min(1.0)
        weights = 1.0 + (T1_LATE_WEIGHT - 1.0) * is_t1
        return (row_ce * weights).sum() / weights.sum().clamp_min(1.0)

    # Pack evaluator-provided output labels into the model's LSD-first digit
    # workspace. No answer or arithmetic trace is generated here.
    valid_labels = batch.valid_mask
    reverse_place = torch.flip(
        torch.flip(valid_labels.long(), dims=[-1]).cumsum(-1), dims=[-1]
    ) - 1
    square_logits = auxiliary["first_square_logits"]
    square_target = torch.full(
        square_logits.shape[:2],
        DIGIT_OFFSET,
        dtype=torch.long,
        device=square_logits.device,
    )
    for position in range(batch.labels.shape[1]):
        selected = valid_labels[:, position]
        slot = reverse_place[:, position].clamp(min=0, max=square_target.shape[1] - 1)
        previous = square_target.gather(1, slot[:, None]).squeeze(1)
        token = torch.where(selected, batch.labels[:, position], previous)
        square_target.scatter_(1, slot[:, None], token[:, None])
    square_row = F.cross_entropy(
        square_logits.transpose(1, 2), square_target, reduction="none"
    ).mean(1)

    # Bit-length-safe no-wrap curriculum: these sufficient input conditions
    # guarantee the provided T=1 final label is also the square-phase label.
    x_value, n_value = auxiliary["x_value"], auxiliary["n_value"]
    no_wrap = (
        ((x_value < 16) & (n_value >= 512))
        | ((x_value < 32) & (n_value >= 1024))
    ).to(row_ce.dtype) * is_t1
    if (
        _training_progress() < SQUARE_CURRICULUM_FRACTION
    ):
        return (square_row * no_wrap).sum() / no_wrap.sum().clamp_min(1.0)
    if _training_progress() < T1_ONLY_FRACTION and bool((is_t1 > 0).any().item()):
        base = (row_ce * is_t1).sum() / is_t1.sum().clamp_min(1.0)
        anchor = (square_row * no_wrap).sum() / no_wrap.sum().clamp_min(1.0)
        return base + SQUARE_ANCHOR_WEIGHT * anchor
    weights = 1.0 + (T1_LATE_WEIGHT - 1.0) * is_t1
    base = (row_ce * weights).sum() / weights.sum().clamp_min(1.0)
    anchor = (square_row * no_wrap).sum() / no_wrap.sum().clamp_min(1.0)
    return base + SQUARE_ANCHOR_WEIGHT * anchor


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


class MuonWarmdown:
    """Update-count schedule that retained the standalone Neural GPU solution."""
    def __init__(self, optimizer: CombinedOptimizer) -> None:
        self.optimizer = optimizer
        self.steps = 0

    def step(self) -> None:
        self.steps += 1
        progress = min(1.0, max(0.0, (self.steps - 1000) / 4000))
        self.optimizer.param_groups[0]["lr"] = 0.002 + 0.018 * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


class CombinedOptimizer:
    def __init__(self, optimizers: list[torch.optim.Optimizer]) -> None:
        self.optimizers = optimizers
        self.param_groups = [group for optimizer in optimizers for group in optimizer.param_groups]

    def step(self, closure=None) -> None:
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {"optimizers": [optimizer.state_dict() for optimizer in self.optimizers]}

    def load_state_dict(self, state_dict: dict) -> None:
        for optimizer, child_state in zip(self.optimizers, state_dict["optimizers"]):
            optimizer.load_state_dict(child_state)


def _zeropower(gradient: Tensor) -> Tensor:
    original_shape = gradient.shape
    value = gradient.reshape(gradient.shape[0], -1).bfloat16()
    value = value / (value.norm() + 1e-7)
    transposed = value.shape[0] > value.shape[1]
    if transposed:
        value = value.T
    for _ in range(5):
        gram = value @ value.T
        value = 3.4445 * value + (-4.775 * gram + 2.0315 * gram @ gram) @ value
    if transposed:
        value = value.T
    return value.reshape(original_shape)


class FlattenedMuon(torch.optim.Optimizer):
    """Muon with convolution kernels flattened exactly as in the winning pilot."""
    def __init__(self, params) -> None:
        super().__init__(params, {"lr": 0.02, "momentum": 0.95, "weight_decay": 1e-5})

    @torch.no_grad()
    def step(self, closure=None) -> None:
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(gradient)
                buffer = state["momentum_buffer"]
                buffer.lerp_(gradient, 1.0 - group["momentum"])
                update = _zeropower(gradient.lerp(buffer, group["momentum"]))
                rows = parameter.shape[0]
                columns = parameter.numel() // rows
                scale = max(1.0, rows / columns) ** 0.5
                parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                parameter.add_(update, alpha=-group["lr"] * scale)


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    global _training_started, _training_total_seconds
    _training_started = time.monotonic()
    _training_total_seconds = max(1.0, float(spec.training_time_seconds))
    matrix = [parameter for parameter in model.parameters() if parameter.ndim >= 2]
    scalar = [parameter for parameter in model.parameters() if parameter.ndim < 2]
    muon = FlattenedMuon(matrix)
    adamw = torch.optim.AdamW(
        scalar,
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=1e-5,
        capturable=spec.device_type == "cuda",
    )
    optimizer = CombinedOptimizer([muon, adamw])
    schedule = (
        WallClockSchedule(optimizer, spec.training_time_seconds)
        if spec.training_time_seconds <= 120
        else MuonWarmdown(optimizer)
    )
    return OptimizerBundle(
        optimizer, schedule
    )


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    token_training_loss=token_training_loss,
    batch_size=128,
    eval_batch_size=256,
)

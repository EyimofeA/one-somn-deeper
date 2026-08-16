"""Binary fused work-state recurrence: width 256 + tuned flattened Muon.

The model receives exact binary representations of evaluator-provided x and N,
then learns the complete transition from final labels only.  It has no explicit
squarer, reducer, arithmetic trace, or task oracle.
"""
from __future__ import annotations

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

N_MARK, X_MARK, T_MARK, DIGIT_OFFSET = 2, 3, 4, 7
BITS, WORKSPACE_BITS, CHANNELS, LANES, UPDATES = 11, 22, 128, 4, 11
T1_FRACTION, T1_WEIGHT = 0.50, 4.0

_started, _seconds = 0.0, 1.0


def _progress() -> float:
    return min(max((time.monotonic() - _started) / _seconds, 0.0), 1.0)


def _parse(ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    field = (
        (ids == N_MARK).cumsum(-1)
        + (ids == X_MARK).cumsum(-1)
        + (ids == T_MARK).cumsum(-1)
    ).clamp(max=3)
    digit = ids >= DIGIT_OFFSET
    values = []
    for which in (1, 2, 3):
        value = torch.zeros(ids.shape[0], dtype=torch.long, device=ids.device)
        for position in range(ids.shape[1]):
            selected = (field[:, position] == which) & digit[:, position]
            next_digit = (ids[:, position] - DIGIT_OFFSET).clamp(min=0, max=9)
            value = torch.where(selected, 10 * value + next_digit, value)
        values.append(value)
    return values[0], values[1], values[2].clamp(min=1, max=8)


def _bits(value: Tensor, width: int = WORKSPACE_BITS) -> Tensor:
    shifts = torch.arange(width, device=value.device)
    return torch.bitwise_and(torch.bitwise_right_shift(value[:, None], shifts), 1)


def _straight_through_bits(logits: Tensor) -> Tensor:
    soft = logits.sigmoid()
    hard = (soft >= 0.5).to(soft.dtype)
    return hard + soft - soft.detach()


class Cell(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.update = nn.Conv2d(CHANNELS, CHANNELS, 3, padding=1)
        self.reset = nn.Conv2d(CHANNELS, CHANNELS, 3, padding=1)
        self.candidate = nn.Conv2d(CHANNELS, CHANNELS, 3, padding=1)

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
        self.config = type(
            "Config", (), {"vocab_size": spec.vocab_size, "max_seq_len": spec.max_seq_len}
        )()
        self.bit_embedding = nn.Embedding(2, CHANNELS)
        self.source_role = nn.Parameter(torch.randn(CHANNELS) * 0.02)
        self.modulus_role = nn.Parameter(torch.randn(CHANNELS) * 0.02)
        self.work_role = nn.Parameter(torch.randn(CHANNELS) * 0.02)
        self.boundaries = nn.Parameter(torch.randn(2, CHANNELS) * 0.02)
        self.cell = Cell()
        self.readout = nn.Conv1d(CHANNELS, 1, 1)
        self.auxiliary: object | None = None

    def _transition(self, source_bits: Tensor, modulus_bits: Tensor) -> Tensor:
        batch = source_bits.shape[0]
        source = self.bit_embedding(source_bits.long()).transpose(1, 2)
        modulus = self.bit_embedding(modulus_bits.long()).transpose(1, 2)
        source = source + self.source_role[None, :, None]
        modulus = modulus + self.modulus_role[None, :, None]
        state = source.new_zeros(batch, CHANNELS, LANES, WORKSPACE_BITS)
        state[:, :, 2] = source + self.work_role[None, :, None]
        mask = None
        if self.training:
            keep = 0.91
            mask = torch.empty(batch, CHANNELS, 1, 1, device=state.device).bernoulli_(keep) / keep
        for _ in range(UPDATES):
            visible = state.clone()
            visible[:, :, 0] = source
            visible[:, :, 1] = modulus
            visible[:, :, :, 0] += self.boundaries[0][None, :, None]
            visible[:, :, :, -1] += self.boundaries[1][None, :, None]
            state = self.cell(visible, mask)
        return self.readout(state[:, :, 2, :BITS]).squeeze(1)

    def _decimal_logits(self, bits: Tensor, length: int) -> Tensor:
        weights = torch.bitwise_left_shift(
            torch.ones(BITS, dtype=torch.long, device=bits.device),
            torch.arange(BITS, device=bits.device),
        )
        value = ((bits >= 0.5).long() * weights).sum(1)
        digits = []
        for place in (1, 10, 100, 1000):
            quotient = torch.div(value, place, rounding_mode="floor")
            digit = quotient - 10 * torch.div(quotient, 10, rounding_mode="floor")
            digits.append(digit + DIGIT_OFFSET)
        tokens = torch.stack(digits, 1)
        digit_logits = bits.new_full((bits.shape[0], 4, self.config.vocab_size), -20.0)
        digit_logits.scatter_(2, tokens[:, :, None], 20.0)
        places = (length - 1 - torch.arange(length, device=bits.device)).clamp(max=3)
        return digit_logits[:, places]

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> tuple[Tensor, object | None]:
        del attention_mask
        n, x, t = _parse(input_ids)
        modulus_bits = _bits(n)
        state = _bits(x)
        effective_t = torch.ones_like(t) if self.training and _progress() < T1_FRACTION else t
        final_binary_logits = state[:, :BITS].float() * 2.0 - 1.0
        for step in range(int(effective_t.max().item())):
            proposed_logits = self._transition(state, modulus_bits)
            proposed = _straight_through_bits(proposed_logits)
            active = (step < effective_t)[:, None]
            proposed_workspace = F.pad(proposed, (0, WORKSPACE_BITS - BITS))
            state = torch.where(active, proposed_workspace, state)
            final_binary_logits = torch.where(active, proposed_logits, final_binary_logits)
        self.auxiliary = {"binary_logits": final_binary_logits, "t": t}
        return self._decimal_logits(state[:, :BITS], input_ids.shape[1]), self.auxiliary


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    assert isinstance(batch.auxiliary, dict)
    target = torch.zeros(batch.labels.shape[0], dtype=torch.long, device=batch.labels.device)
    for position in range(batch.labels.shape[1]):
        selected = batch.labels[:, position] >= DIGIT_OFFSET
        digit = (batch.labels[:, position] - DIGIT_OFFSET).clamp(min=0, max=9)
        target = torch.where(selected, 10 * target + digit, target)
    target_bits = _bits(target, BITS).to(batch.auxiliary["binary_logits"].dtype)
    row = F.binary_cross_entropy_with_logits(
        batch.auxiliary["binary_logits"], target_bits, reduction="none"
    ).mean(1)
    t1 = (batch.auxiliary["t"] == 1).to(row.dtype)
    if _progress() < T1_FRACTION and bool((t1 > 0).any().item()):
        return (row * t1).sum() / t1.sum().clamp_min(1.0)
    weights = 1.0 + (T1_WEIGHT - 1.0) * t1
    return (row * weights).sum() / weights.sum()


def _zeropower(gradient: Tensor) -> Tensor:
    shape = gradient.shape
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
    return value.reshape(shape)


class FlattenedMuon(torch.optim.Optimizer):
    def __init__(self, params) -> None:
        super().__init__(params, {"lr": 0.006, "momentum": 0.95, "weight_decay": 0.1})

    @torch.no_grad()
    def step(self, closure=None) -> None:
        del closure
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
                rows, columns = parameter.shape[0], parameter.numel() // parameter.shape[0]
                scale = max(1.0, rows / columns) ** 0.5
                parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                parameter.add_(update, alpha=-group["lr"] * scale)


class CombinedOptimizer:
    def __init__(self, optimizers: list[torch.optim.Optimizer]) -> None:
        self.optimizers = optimizers
        self.param_groups = [group for optimizer in optimizers for group in optimizer.param_groups]

    def step(self, closure=None) -> None:
        del closure
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {"optimizers": [optimizer.state_dict() for optimizer in self.optimizers]}

    def load_state_dict(self, state_dict: dict) -> None:
        for optimizer, child in zip(self.optimizers, state_dict["optimizers"]):
            optimizer.load_state_dict(child)


class Warmup:
    def __init__(self, optimizer: CombinedOptimizer) -> None:
        self.optimizer, self.steps = optimizer, 0

    def step(self) -> None:
        self.steps += 1
        self.optimizer.param_groups[0]["lr"] = 0.006 * min(1.0, self.steps / 250.0)


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    global _started, _seconds
    _started, _seconds = time.monotonic(), max(float(spec.training_time_seconds), 1.0)
    matrix = [parameter for parameter in model.parameters() if parameter.ndim >= 2]
    scalar = [parameter for parameter in model.parameters() if parameter.ndim < 2]
    optimizer = CombinedOptimizer([
        FlattenedMuon(matrix),
        torch.optim.AdamW(
            scalar,
            lr=3e-4,
            betas=(0.9, 0.95),
            weight_decay=1e-5,
            capturable=spec.device_type == "cuda",
        ),
    ])
    return OptimizerBundle(optimizer, Warmup(optimizer))


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    token_training_loss=token_training_loss,
    batch_size=256,
    eval_batch_size=256,
)

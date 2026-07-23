"""Canonical implementation for bounded digit-column carry diagnostics.

This module is deliberately submission-shaped: ``freeze_submission.py`` copies
it verbatim and appends a selected ``CarryScanSettings`` value, yielding a
self-contained competition upload.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

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


@dataclass(frozen=True)
class CarryScanSettings:
    """The sole active mechanism switch for this diagnostic.

    ``num_prototypes=0`` is the continuous shared-state scan.  A positive
    value projects every state transition onto that many learned prototypes.
    """

    d_model: int = 32
    block_width: int = 3
    max_columns: int = 7
    flush_steps: int = 2
    num_prototypes: int = 0
    learning_rate: float = 3e-3
    weight_decay: float = 0.1
    batch_size: int = 256
    eval_batch_size: int = 512

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.block_width <= 0:
            raise ValueError("model and block widths must be positive")
        if self.max_columns <= 0 or self.flush_steps <= 0:
            raise ValueError("scan lengths must be positive")
        if self.num_prototypes < 0:
            raise ValueError("num_prototypes cannot be negative")


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


def _valid_token_mask(
    input_ids: Tensor,
    attention_mask: Tensor | None,
) -> Tensor:
    if attention_mask is None:
        return torch.ones_like(input_ids, dtype=torch.bool)
    if attention_mask.shape == input_ids.shape:
        return attention_mask.to(device=input_ids.device, dtype=torch.bool)
    if attention_mask.shape == (
        input_ids.shape[0],
        input_ids.shape[1],
        input_ids.shape[1],
    ):
        return attention_mask.to(
            device=input_ids.device,
            dtype=torch.bool,
        ).any(dim=1)
    raise ValueError("invalid attention_mask shape")


def _column_counts(valid_lengths: Tensor, settings: CarryScanSettings) -> Tensor:
    counts = torch.zeros_like(valid_lengths)
    recognized = torch.zeros_like(valid_lengths, dtype=torch.bool)
    for column_count in range(1, settings.max_columns + 1):
        expected_length = 1 + settings.block_width * column_count
        matches = valid_lengths == expected_length
        counts = torch.where(
            matches,
            torch.full_like(counts, column_count),
            counts,
        )
        recognized = recognized | matches
    if not bool(torch.all(recognized)):
        raise ValueError("prompt must be N followed by valid digit blocks")
    return counts


class CarryScanModel(nn.Module):
    def __init__(self, spec: ModelSpec, settings: CarryScanSettings) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.settings = settings
        self.digit_embedding = nn.Embedding(spec.vocab_size, settings.d_model)
        self.block_encoder = nn.Sequential(
            nn.Linear(settings.block_width * settings.d_model, settings.d_model),
            nn.GELU(),
            nn.Linear(settings.d_model, settings.d_model),
        )
        self.carry_cell = nn.GRUCell(settings.d_model, settings.d_model)
        self.initial_state = nn.Parameter(torch.zeros(settings.d_model))
        self.end_input = nn.Parameter(torch.empty(settings.d_model))
        self.output_head = nn.Linear(settings.d_model, spec.vocab_size, bias=False)
        self.output_head.weight = self.digit_embedding.weight
        if settings.num_prototypes:
            self.state_selector = nn.Linear(
                settings.d_model,
                settings.num_prototypes,
            )
            self.state_codebook = nn.Parameter(
                torch.empty(settings.num_prototypes, settings.d_model)
            )
            nn.init.normal_(self.state_codebook, std=0.02)
        nn.init.normal_(self.digit_embedding.weight, std=0.02)
        nn.init.normal_(self.end_input, std=0.02)

    def _transition(self, transition_input: Tensor, state: Tensor) -> Tensor:
        candidate_state = self.carry_cell(transition_input, state)
        if not self.settings.num_prototypes:
            return candidate_state
        soft_assignment = F.softmax(
            self.state_selector(candidate_state),
            dim=-1,
        )
        hard_assignment = F.one_hot(
            soft_assignment.argmax(dim=-1),
            num_classes=self.settings.num_prototypes,
        ).to(dtype=soft_assignment.dtype)
        straight_through = hard_assignment + (
            soft_assignment - soft_assignment.detach()
        )
        return straight_through @ self.state_codebook

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, length)")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("input exceeds configured maximum sequence length")

        settings = self.settings
        valid_lengths = _valid_token_mask(input_ids, attention_mask).long().sum(1)
        column_counts = _column_counts(valid_lengths, settings)
        padded_ids = F.pad(
            input_ids,
            (0, self.config.max_seq_len - input_ids.shape[1]),
            value=0,
        )
        batch_size = input_ids.shape[0]
        state = self.initial_state[None, :].expand(batch_size, -1)
        column_logits: list[Tensor] = []
        for column in range(settings.max_columns):
            block_start = 1 + settings.block_width * column
            block_ids = padded_ids[:, block_start : block_start + settings.block_width]
            block_input = self.block_encoder(
                self.digit_embedding(block_ids).reshape(
                    batch_size,
                    settings.block_width * settings.d_model,
                )
            )
            active = column_counts > column
            next_state = self._transition(block_input, state)
            state = torch.where(active[:, None], next_state, state)
            column_logits.append(self.output_head(state))

        flush_input = self.end_input[None, :].expand(batch_size, -1)
        flush_logits: list[Tensor] = []
        for _ in range(settings.flush_steps):
            state = self._transition(flush_input, state)
            flush_logits.append(self.output_head(state))

        emission_bank = torch.cat(
            (torch.stack(column_logits, 1), torch.stack(flush_logits, 1)),
            dim=1,
        )
        emission_slot = torch.arange(
            settings.max_columns + settings.flush_steps,
            device=input_ids.device,
        )[None, :]
        emission_valid = emission_slot < column_counts[:, None] + settings.flush_steps
        source_slot = torch.where(
            emission_slot < column_counts[:, None],
            emission_slot,
            settings.max_columns + emission_slot - column_counts[:, None],
        )
        source_slot = torch.where(
            emission_valid,
            source_slot,
            torch.zeros_like(source_slot),
        )
        selected_logits = emission_bank.gather(
            1,
            source_slot[:, :, None].expand(-1, -1, self.config.vocab_size),
        )
        target_positions = valid_lengths[:, None] - column_counts[:, None]
        target_positions = target_positions - settings.flush_steps + emission_slot
        target_positions = torch.where(
            emission_valid,
            target_positions,
            torch.zeros_like(target_positions),
        )
        logits = torch.zeros(
            batch_size,
            input_ids.shape[1],
            self.config.vocab_size,
            device=input_ids.device,
            dtype=selected_logits.dtype,
        )
        logits.scatter_add_(
            1,
            target_positions[:, :, None].expand(-1, -1, self.config.vocab_size),
            torch.where(
                emission_valid[:, :, None],
                selected_logits,
                torch.zeros_like(selected_logits),
            ),
        )
        return logits, None


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    spec: OptimizerSpec,
) -> torch.optim.lr_scheduler.LRScheduler:
    total_seconds = max(1.0, float(spec.training_time_seconds))
    started = time.monotonic()

    def factor(_step: int) -> float:
        progress = min(max((time.monotonic() - started) / total_seconds, 0.0), 1.0)
        if progress < 0.05:
            return 0.01 + 0.99 * progress / 0.05
        tail = (progress - 0.05) / 0.95
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * tail))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def build_submission(settings: CarryScanSettings) -> Submission:
    def build_model(spec: ModelSpec) -> CarryScanModel:
        model = CarryScanModel(spec, settings)
        assert_model_state(model, spec)
        return model

    def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=settings.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=settings.weight_decay,
            capturable=spec.device_type == "cuda",
        )
        return OptimizerBundle(optimizer, scheduler=_build_scheduler(optimizer, spec))

    return Submission(
        build_model=build_model,
        build_optimizer=build_optimizer,
        batch_size=settings.batch_size,
        eval_batch_size=settings.eval_batch_size,
    )

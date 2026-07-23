"""Shared learned carry-state scan for the bounded normalization diagnostic."""

from __future__ import annotations

import math

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


D_MODEL = 32
BLOCK_WIDTH = 3
MAX_COLUMNS = 7
FLUSH_STEPS = 2


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


def _column_counts(valid_lengths: Tensor) -> Tensor:
    counts = torch.zeros_like(valid_lengths)
    recognized = torch.zeros_like(valid_lengths, dtype=torch.bool)
    for column_count in range(1, MAX_COLUMNS + 1):
        expected_length = 1 + BLOCK_WIDTH * column_count
        matches = valid_lengths == expected_length
        counts = torch.where(
            matches,
            torch.full_like(counts, column_count),
            counts,
        )
        recognized = recognized | matches
    if not bool(torch.all(recognized)):
        raise ValueError("prompt must be N followed by one to seven 3-digit blocks")
    return counts


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.digit_embedding = nn.Embedding(spec.vocab_size, D_MODEL)
        self.block_encoder = nn.Sequential(
            nn.Linear(BLOCK_WIDTH * D_MODEL, D_MODEL),
            nn.GELU(),
            nn.Linear(D_MODEL, D_MODEL),
        )
        self.carry_cell = nn.GRUCell(D_MODEL, D_MODEL)
        self.initial_state = nn.Parameter(torch.zeros(D_MODEL))
        self.end_input = nn.Parameter(torch.empty(D_MODEL))
        self.output_head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.output_head.weight = self.digit_embedding.weight
        nn.init.normal_(self.digit_embedding.weight, std=0.02)
        nn.init.normal_(self.end_input, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, length)")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("input exceeds configured maximum sequence length")

        valid = _valid_token_mask(input_ids, attention_mask)
        valid_lengths = valid.to(torch.long).sum(dim=1)
        column_counts = _column_counts(valid_lengths)
        padded_ids = F.pad(
            input_ids,
            (0, self.config.max_seq_len - input_ids.shape[1]),
            value=0,
        )

        state = self.initial_state[None, :].expand(input_ids.shape[0], -1)
        column_logits: list[Tensor] = []
        for column in range(MAX_COLUMNS):
            block_start = 1 + BLOCK_WIDTH * column
            block_ids = padded_ids[
                :,
                block_start : block_start + BLOCK_WIDTH,
            ]
            block_embedding = self.digit_embedding(block_ids).reshape(
                input_ids.shape[0],
                BLOCK_WIDTH * D_MODEL,
            )
            block_input = self.block_encoder(block_embedding)
            next_state = self.carry_cell(block_input, state)
            active = column_counts > column
            state = torch.where(active[:, None], next_state, state)
            column_logits.append(self.output_head(state))

        flush_logits: list[Tensor] = []
        flush_input = self.end_input[None, :].expand(input_ids.shape[0], -1)
        for _ in range(FLUSH_STEPS):
            state = self.carry_cell(flush_input, state)
            flush_logits.append(self.output_head(state))

        emission_bank = torch.cat(
            (
                torch.stack(column_logits, dim=1),
                torch.stack(flush_logits, dim=1),
            ),
            dim=1,
        )
        emission_slot = torch.arange(
            MAX_COLUMNS + FLUSH_STEPS,
            device=input_ids.device,
        )[None, :]
        emission_valid = emission_slot < column_counts[:, None] + FLUSH_STEPS
        source_slot = torch.where(
            emission_slot < column_counts[:, None],
            emission_slot,
            MAX_COLUMNS + emission_slot - column_counts[:, None],
        )
        source_slot = torch.where(
            emission_valid,
            source_slot,
            torch.zeros_like(source_slot),
        )
        selected_logits = emission_bank.gather(
            1,
            source_slot[:, :, None].expand(
                -1,
                -1,
                emission_bank.shape[-1],
            ),
        )

        target_start = valid_lengths - column_counts - FLUSH_STEPS
        target_positions = target_start[:, None] + emission_slot
        target_positions = torch.where(
            emission_valid,
            target_positions,
            torch.zeros_like(target_positions),
        )
        logits = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            self.config.vocab_size,
            device=input_ids.device,
            dtype=selected_logits.dtype,
        )
        logits.scatter_add_(
            1,
            target_positions[:, :, None].expand(
                -1,
                -1,
                self.config.vocab_size,
            ),
            torch.where(
                emission_valid[:, :, None],
                selected_logits,
                torch.zeros_like(selected_logits),
            ),
        )
        return logits, None


WARMUP_FRACTION = 0.05
FINAL_LR_FRACTION = 0.01


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    spec: OptimizerSpec,
) -> torch.optim.lr_scheduler.LRScheduler:
    import time

    total_seconds = max(1.0, float(spec.training_time_seconds))
    started = time.monotonic()

    def factor(_step: int) -> float:
        progress = (time.monotonic() - started) / total_seconds
        progress = min(max(progress, 0.0), 1.0)
        if progress < WARMUP_FRACTION:
            return FINAL_LR_FRACTION + (1.0 - FINAL_LR_FRACTION) * (
                progress / WARMUP_FRACTION
            )
        tail = (progress - WARMUP_FRACTION) / (1.0 - WARMUP_FRACTION)
        cosine = 0.5 * (1.0 + math.cos(math.pi * tail))
        return FINAL_LR_FRACTION + (1.0 - FINAL_LR_FRACTION) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(
    model: nn.Module,
    spec: OptimizerSpec,
) -> OptimizerBundle:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-3,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        capturable=spec.device_type == "cuda",
    )
    return OptimizerBundle(
        optimizer,
        scheduler=_build_scheduler(optimizer, spec),
    )


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    batch_size=256,
    eval_batch_size=512,
)

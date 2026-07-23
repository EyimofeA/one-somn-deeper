"""Learned pair-table schoolbook executor for a local multiplication diagnostic.

The table starts random and stores a feature per digit-pair category, not a
hard-coded product. The carry transition and decimal decode remain learned.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from benchmark import (
    ModelSpec,
    OptimizerBundle,
    OptimizerSpec,
    Submission,
    assert_model_state,
)


D_MODEL = 64
NUM_DIGITS = 3
PRODUCT_COLUMNS = 2 * NUM_DIGITS - 1
TOTAL_STEPS = 8_000
WARMUP_STEPS = 400


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.pair_table = nn.Embedding(spec.vocab_size * spec.vocab_size, D_MODEL)
        self.carry_cell = nn.GRUCell(D_MODEL, D_MODEL)
        self.initial_state = nn.Parameter(torch.zeros(D_MODEL))
        self.flush_input = nn.Parameter(torch.empty(D_MODEL))
        self.output_head = nn.Linear(D_MODEL, spec.vocab_size)
        nn.init.normal_(self.pair_table.weight, std=0.02)
        nn.init.normal_(self.flush_input, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        del attention_mask
        if input_ids.ndim != 2 or input_ids.shape[1] != 10:
            raise ValueError("product-scan prompts must have shape (batch, 10)")
        columns = torch.zeros(
            input_ids.shape[0],
            PRODUCT_COLUMNS,
            D_MODEL,
            device=input_ids.device,
            dtype=self.pair_table.weight.dtype,
        )
        for left_index in range(NUM_DIGITS):
            for right_index in range(NUM_DIGITS):
                pair = (
                    input_ids[:, 1 + left_index] * self.config.vocab_size
                    + input_ids[:, 5 + right_index]
                )
                columns[:, left_index + right_index] += self.pair_table(pair)
        state = self.initial_state[None, :].expand(input_ids.shape[0], -1)
        emitted: list[Tensor] = []
        for column in columns.unbind(dim=1):
            state = self.carry_cell(column, state)
            emitted.append(self.output_head(state))
        state = self.carry_cell(
            self.flush_input[None, :].expand_as(state),
            state,
        )
        emitted.append(self.output_head(state))
        logits = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            self.config.vocab_size,
            device=input_ids.device,
            dtype=state.dtype,
        )
        logits[:, -len(emitted) :, :] = torch.stack(emitted, dim=1)
        return logits, None


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-3,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        capturable=spec.device_type == "cuda",
    )

    def factor(step: int) -> float:
        if step < WARMUP_STEPS:
            return (step + 1) / WARMUP_STEPS
        progress = min((step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS), 1.0)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return OptimizerBundle(
        optimizer,
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, factor),
    )


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    batch_size=256,
    eval_batch_size=512,
)

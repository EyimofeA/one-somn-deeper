"""Shared learned digit-squaring cell with an STE-discrete recurrent state."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from benchmark import ModelSpec, OptimizerBundle, OptimizerSpec, Submission, assert_model_state


DIGIT_OFFSET = 7
VOCAB_SIZE = 17
NUM_DIGITS = 4
D_MODEL = 64
TOTAL_STEPS = 10_000
WARMUP_STEPS = 500


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.pair_table = nn.Parameter(torch.empty(10, 10, D_MODEL))
        self.pair_fold = nn.GRUCell(D_MODEL, D_MODEL)
        self.pair_fold_initial = nn.Parameter(torch.zeros(D_MODEL))
        self.carry_cell = nn.GRUCell(D_MODEL, D_MODEL)
        self.carry_initial = nn.Parameter(torch.zeros(D_MODEL))
        self.flush_input = nn.Parameter(torch.empty(D_MODEL))
        self.digit_head = nn.Linear(D_MODEL, 10)
        nn.init.normal_(self.pair_table, std=0.02)
        nn.init.normal_(self.flush_input, std=0.02)

    def square_step(self, digits: Tensor) -> tuple[Tensor, Tensor]:
        terms: list[list[Tensor]] = [[] for _ in range(2 * NUM_DIGITS - 1)]
        for left_index in range(NUM_DIGITS):
            for right_index in range(NUM_DIGITS):
                feature = torch.einsum(
                    "bi,bj,ijd->bd",
                    digits[:, left_index],
                    digits[:, right_index],
                    self.pair_table,
                )
                terms[left_index + right_index].append(feature)
        columns: list[Tensor] = []
        for column_terms in terms:
            folded = self.pair_fold_initial[None, :].expand(digits.shape[0], -1)
            for term in column_terms:
                folded = self.pair_fold(term, folded)
            columns.append(folded)

        carry = self.carry_initial[None, :].expand(digits.shape[0], -1)
        emitted: list[Tensor] = []
        emitted_logits: list[Tensor] = []
        for column in columns:
            carry = self.carry_cell(column, carry)
            logits = self.digit_head(carry)
            emitted_logits.append(logits)
            emitted.append(F.softmax(logits, dim=-1))
        carry = self.carry_cell(self.flush_input[None, :].expand_as(carry), carry)
        logits = self.digit_head(carry)
        emitted_logits.append(logits)
        emitted.append(F.softmax(logits, dim=-1))
        return (
            torch.stack(emitted[:7], dim=1),
            torch.stack(emitted_logits[:7], dim=1),
        )

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> tuple[Tensor, None]:
        del attention_mask
        if input_ids.ndim != 2 or input_ids.shape[1] != 9:
            raise ValueError("auxiliary-supervision prompts must have shape (batch, 9)")
        initial = (input_ids[:, 1:5] - DIGIT_OFFSET).clamp(0, 9)
        digits = F.one_hot(initial, num_classes=10).to(self.pair_table.dtype)
        recurrence_count = (input_ids[:, 6] - DIGIT_OFFSET).clamp(0, 3)
        output_logits = torch.zeros(
            (digits.shape[0], 7, 10), device=digits.device, dtype=digits.dtype
        )
        for step in range(3):
            all_digits, step_logits = self.square_step(digits)
            soft_digits = all_digits[:, :NUM_DIGITS]
            hard_digits = F.one_hot(soft_digits.argmax(dim=-1), num_classes=10).to(
                soft_digits.dtype
            )
            updated = soft_digits + (hard_digits - soft_digits).detach()
            active = (recurrence_count > step)[:, None, None]
            digits = torch.where(active, updated, digits)
            output_logits = torch.where(active, step_logits, output_logits)
        logits = torch.full(
            (input_ids.shape[0], input_ids.shape[1], self.config.vocab_size),
            -20.0,
            device=input_ids.device,
            dtype=digits.dtype,
        )
        logits[:, 1:5, DIGIT_OFFSET : DIGIT_OFFSET + 10] = output_logits[:, 3:7]
        logits[:, -NUM_DIGITS:, DIGIT_OFFSET : DIGIT_OFFSET + 10] = output_logits[:, :4]
        if self.training:
            weights = torch.tensor(
                [0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0],
                device=digits.device,
                dtype=digits.dtype,
            ).repeat(input_ids.shape[0])
        else:
            weights = torch.ones(
                input_ids.shape[0] * NUM_DIGITS,
                device=digits.device,
                dtype=digits.dtype,
            )
        return logits, weights


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-3, betas=(0.9, 0.95), weight_decay=0.01,
        capturable=spec.device_type == "cuda",
    )

    def factor(step: int) -> float:
        if step < WARMUP_STEPS:
            return (step + 1) / WARMUP_STEPS
        progress = min((step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS), 1.0)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return OptimizerBundle(optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, factor))


def training_loss(logits: Tensor, labels: Tensor, weights: Tensor) -> Tensor:
    per_token = F.cross_entropy(logits, labels, reduction="none")
    return (per_token * weights).sum() / weights.sum()


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    training_loss=training_loss,
    batch_size=256,
    eval_batch_size=512,
)

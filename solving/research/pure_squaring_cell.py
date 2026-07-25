"""Multiply-and-carry cell ONLY, no reduction stage — pure squaring isolation.

Same validated mechanism as soft_digit_squaring_recurrence.py / the reduction
cell's square_raw (learned digit-pair table, GRU column-fold, GRU carry
scan), but emits all 2*NUM_DIGITS raw digits directly as the answer — no
truncation, no mod-N reduction. Tests whether the multiply mechanism's
97.8%-peak result (a) holds at full 8-digit width (not just the low 4) and
(b) is a STABLE floor, not a transient peak, via periodic eval.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from benchmark import ModelSpec, OptimizerBundle, OptimizerSpec, Submission, assert_model_state


DIGIT_OFFSET = 7
VOCAB_SIZE = 17
NUM_DIGITS = 4
NUM_N_DIGITS = 3
D_MODEL = 128
WARMUP_STEPS = 500
TOTAL_STEPS = 20_000


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

    def square_raw(self, digits: Tensor) -> Tensor:
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
        emitted_logits: list[Tensor] = []
        for column in columns:
            carry = self.carry_cell(column, carry)
            emitted_logits.append(self.digit_head(carry))
        carry = self.carry_cell(self.flush_input[None, :].expand_as(carry), carry)
        emitted_logits.append(self.digit_head(carry))
        return torch.stack(emitted_logits, dim=1)  # (batch, 2*NUM_DIGITS, 10), LSB->MSB

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> tuple[Tensor, None]:
        del attention_mask
        # layout: [N_marker, n0,n1,n2, X_marker, x0,x1,x2,x3, T_marker, t] (11 tokens)
        if input_ids.ndim != 2 or input_ids.shape[1] != 11:
            raise ValueError("pure-squaring prompts must have shape (batch, 11)")
        # N is present for token-format realism only; never read/used.
        x_raw = (input_ids[:, 5:9] - DIGIT_OFFSET).clamp(0, 9).flip(dims=[-1])
        x_digits = F.one_hot(x_raw, num_classes=10).to(self.pair_table.dtype)

        product_lsb_first = self.square_raw(x_digits)          # (batch, 8, 10), LSB->MSB
        product_msb_first = product_lsb_first.flip(dims=[1])   # match label convention (MSB first)

        logits = torch.full(
            (input_ids.shape[0], input_ids.shape[1], self.config.vocab_size),
            -20.0,
            device=input_ids.device,
            dtype=product_msb_first.dtype,
        )
        n_answer_digits = product_msb_first.shape[1]
        logits[:, -n_answer_digits:, DIGIT_OFFSET : DIGIT_OFFSET + 10] = product_msb_first
        return logits, None


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    del spec  # normal (step-count) schedule, matching the reduction-cell card
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-3, betas=(0.9, 0.95), weight_decay=0.01,
        capturable=True,
    )

    def factor(step: int) -> float:
        if step < WARMUP_STEPS:
            return (step + 1) / WARMUP_STEPS
        progress = min((step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS), 1.0)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return OptimizerBundle(optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, factor))


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    batch_size=250,  # 8,000 / 250 = 32 exact batches
    eval_batch_size=512,
)

"""Fixed-step neural arithmetic diagnostic for held-out decimal digit pairs.

Diagnostic only: the ordinal digit representation is an intentionally strong
numeric prior. This file is not a competition-submission candidate.
"""

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


DIGIT_OFFSET = 7
WIDTH = 32
WARMUP_STEPS = 200
TOTAL_STEPS = 4_000


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class NeuralArithmeticUnit(nn.Module):
    """Trainable additive/multiplicative paths with a learned input gate."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight_raw = nn.Parameter(torch.empty(2, width))
        self.weight_gate = nn.Parameter(torch.empty(2, width))
        self.gate = nn.Linear(2, width)
        nn.init.xavier_uniform_(self.weight_raw)
        nn.init.xavier_uniform_(self.weight_gate)

    def forward(self, values: Tensor) -> Tensor:
        weights = torch.tanh(self.weight_raw) * torch.sigmoid(self.weight_gate)
        additive = values @ weights
        multiplicative = torch.exp(torch.log(values.abs() + 1e-7) @ weights)
        mix = torch.sigmoid(self.gate(values))
        return mix * additive + (1.0 - mix) * multiplicative


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.vocab_size = spec.vocab_size
        numeric = torch.zeros(spec.vocab_size)
        numeric[DIGIT_OFFSET : DIGIT_OFFSET + 10] = torch.arange(10) / 9.0
        self.register_buffer("numeric_token_value", numeric)
        self.arithmetic = NeuralArithmeticUnit(WIDTH)
        self.decoder = nn.Sequential(
            nn.LayerNorm(WIDTH),
            nn.Linear(WIDTH, WIDTH),
            nn.GELU(),
            nn.Linear(WIDTH, 20),
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        del attention_mask
        if input_ids.ndim != 2 or input_ids.shape[1] != 6:
            raise ValueError("digit-product prompts must have shape (batch, 6)")
        operands = torch.stack(
            (
                self.numeric_token_value[input_ids[:, 1]],
                self.numeric_token_value[input_ids[:, 3]],
            ),
            dim=-1,
        )
        digit_logits = self.decoder(self.arithmetic(operands)).view(-1, 2, 10)
        token_logits = F.pad(digit_logits, (DIGIT_OFFSET, 0))
        logits = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            self.vocab_size,
            device=input_ids.device,
            dtype=token_logits.dtype,
        )
        logits[:, -2:, :] = token_logits
        return logits, None


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    del spec
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-3,
        betas=(0.9, 0.95),
        weight_decay=0.01,
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

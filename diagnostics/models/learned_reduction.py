"""Reimplementation of the prior learned digit-serial reduction-cell form.

This is a new diagnostic implementation, not a recovery of its historical
result. The transition is learned: no modulo, comparison, arithmetic table, or
handwritten quotient/subtraction operation appears in ``forward``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from data.tokens import VOCAB_SIZE


class LearnedReductionCell(nn.Module):
    """Eight digit-serial learned state updates for the fixed Task-B prompt."""

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_digit_embed = nn.Embedding(10, d_model)
        self.n_place_embed = nn.Embedding(4, d_model)
        self.u_digit_embed = nn.Embedding(10, d_model)
        self.remainder_initial = nn.Parameter(torch.zeros(d_model))
        self.shift_cell = nn.GRUCell(d_model, d_model)
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.quotient_head = nn.Linear(d_model, 10)
        self.quotient_digit_embed = nn.Embedding(10, d_model)
        self.subtract_cell = nn.GRUCell(d_model, d_model)
        self.remainder_digit_heads = nn.ModuleList([nn.Linear(d_model, 10) for _ in range(4)])

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        del attention_mask
        if input_ids.ndim != 2 or input_ids.shape[1] != 19:
            raise ValueError("fixed-N mod prompts must have shape (batch, 19)")
        # [<MOD>, <N>, n0..n3, <U>, u0..u7, <OUT> x4]; tokens 9..18 are digits.
        n_digits = (input_ids[:, 2:6] - 9).clamp(0, 9)
        u_digits = (input_ids[:, 7:15] - 9).clamp(0, 9)
        n = self.n_digit_embed(n_digits)
        n = n + self.n_place_embed(torch.arange(4, device=input_ids.device))[None, :, :]
        keys = self.key_proj(n)

        remainder = self.remainder_initial[None, :].expand(input_ids.shape[0], -1)
        for digit in u_digits.unbind(dim=1):
            remainder = self.shift_cell(self.u_digit_embed(digit), remainder)
            scores = torch.einsum("bd,bpd->bp", self.query_proj(remainder), keys) / math.sqrt(self.d_model)
            context = torch.einsum("bp,bpd->bd", F.softmax(scores, dim=-1), n)
            quotient = F.softmax(self.quotient_head(remainder + context), dim=-1)
            quotient_state = quotient @ self.quotient_digit_embed.weight
            remainder = self.subtract_cell(quotient_state + context, remainder)

        digits = torch.stack([head(remainder) for head in self.remainder_digit_heads], dim=1)
        logits = torch.full(
            (input_ids.shape[0], input_ids.shape[1], VOCAB_SIZE), -20.0,
            device=input_ids.device, dtype=digits.dtype,
        )
        logits[:, -4:, :10] = digits
        return logits

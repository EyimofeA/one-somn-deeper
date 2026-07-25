"""Reduction cell ONLY, no squaring — pure modular-reduction isolation.

Same reduce_mod mechanism as learned_reduction_cell.py (recurrent remainder +
attention over N's digits + learned quotient head + learned subtract), but P
is fed DIRECTLY from the input tokens (an arbitrary 8-digit integer, not a
squaring cell's output). Every P digit is a real, direct, fully-observed
input this time — no dependence on an upstream mechanism's reliability.
Answers whether the reduction mechanism itself can learn division, decoupled
from both squaring's unreliability and the composed test's weak (final-
remainder-only) supervision. Ban-list clean: no `%`, learned subtract only.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from benchmark import ModelSpec, OptimizerBundle, OptimizerSpec, Submission, assert_model_state


DIGIT_OFFSET = 7
VOCAB_SIZE = 17
NUM_N_DIGITS = 3
NUM_P_DIGITS = 8
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
        self.n_digit_embed = nn.Embedding(10, D_MODEL)
        self.n_pos_embed = nn.Embedding(NUM_N_DIGITS, D_MODEL)
        self.p_digit_embed = nn.Embedding(10, D_MODEL)
        self.remainder_initial = nn.Parameter(torch.zeros(D_MODEL))
        self.shift_cell = nn.GRUCell(D_MODEL, D_MODEL)
        self.query_proj = nn.Linear(D_MODEL, D_MODEL)
        self.key_proj = nn.Linear(D_MODEL, D_MODEL)
        self.quotient_head = nn.Linear(D_MODEL, 10)
        self.quotient_digit_embed = nn.Embedding(10, D_MODEL)
        self.subtract_cell = nn.GRUCell(D_MODEL, D_MODEL)
        self.remainder_digit_heads = nn.ModuleList(
            [nn.Linear(D_MODEL, 10) for _ in range(NUM_N_DIGITS)]
        )

    def reduce_mod(self, p_digits_msb_first: Tensor, n_digits: Tensor) -> Tensor:
        batch = p_digits_msb_first.shape[0]
        n_digit_vecs = torch.einsum("bpd,dk->bpk", n_digits, self.n_digit_embed.weight)
        n_positions = torch.arange(NUM_N_DIGITS, device=n_digits.device)
        n_digit_vecs = n_digit_vecs + self.n_pos_embed(n_positions)[None, :, :]
        keys = self.key_proj(n_digit_vecs)
        values = n_digit_vecs

        r = self.remainder_initial[None, :].expand(batch, -1)
        for position in range(p_digits_msb_first.shape[1]):
            p_digit = p_digits_msb_first[:, position]
            p_vec = torch.einsum("bd,dk->bk", p_digit, self.p_digit_embed.weight)
            r = self.shift_cell(p_vec, r)

            query = self.query_proj(r)[:, None, :]
            scores = torch.einsum("bqd,bpd->bqp", query, keys) / math.sqrt(D_MODEL)
            weights = F.softmax(scores, dim=-1)
            context = torch.einsum("bqp,bpd->bqd", weights, values).squeeze(1)

            quotient_soft = F.softmax(self.quotient_head(r + context), dim=-1)
            quotient_vec = torch.einsum("bd,dk->bk", quotient_soft, self.quotient_digit_embed.weight)
            r = self.subtract_cell(quotient_vec + context, r)
        return r

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> tuple[Tensor, None]:
        del attention_mask
        # layout: [N_marker, n0,n1,n2, X_marker, p0..p7, T_marker, t] (15 tokens)
        if input_ids.ndim != 2 or input_ids.shape[1] != 15:
            raise ValueError("pure-reduction prompts must have shape (batch, 15)")
        n_raw = (input_ids[:, 1 : 1 + NUM_N_DIGITS] - DIGIT_OFFSET).clamp(0, 9)
        n_digits = F.one_hot(n_raw, num_classes=10).to(self.n_digit_embed.weight.dtype)
        p_raw = (input_ids[:, 5 : 5 + NUM_P_DIGITS] - DIGIT_OFFSET).clamp(0, 9)
        p_digits = F.one_hot(p_raw, num_classes=10).to(self.n_digit_embed.weight.dtype)  # already MSB-first

        remainder_state = self.reduce_mod(p_digits, n_digits)
        output_logits = torch.stack(
            [head(remainder_state) for head in self.remainder_digit_heads], dim=1
        )

        logits = torch.full(
            (input_ids.shape[0], input_ids.shape[1], self.config.vocab_size),
            -20.0,
            device=input_ids.device,
            dtype=output_logits.dtype,
        )
        logits[:, -NUM_N_DIGITS:, DIGIT_OFFSET : DIGIT_OFFSET + 10] = output_logits
        return logits, None


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    del spec
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-3, betas=(0.9, 0.95), weight_decay=1.0,
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
    batch_size=250,
    eval_batch_size=512,
)

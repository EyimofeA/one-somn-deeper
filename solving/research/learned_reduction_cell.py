"""Learned multiply-and-carry cell, extended with a learned mod-N reduction.

Fixed-N rung (see solving/DESIGN_NEXT.md OPT 1): N=323 held constant across
every row for this first pass. N's digits are still a real model input (not a
hardcoded constant) so the interface is unchanged when N later varies.

Diagnostic only: not a competition-submission candidate as-is (no wall-clock
schedule yet, deferred per plan). Ban-list clean: the reduction below is a
learned recurrent subtract-and-compare, never a `%` or closed-form solve.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from benchmark import ModelSpec, OptimizerBundle, OptimizerSpec, Submission, assert_model_state


DIGIT_OFFSET = 7
VOCAB_SIZE = 17
NUM_DIGITS = 4       # digits of x (and of x^2's low/high halves before reduction)
NUM_N_DIGITS = 3     # digits of N=323, fixed for this rung
D_MODEL = 128        # already-validated width from the one-step gate
TOTAL_STEPS = 20_000
WARMUP_STEPS = 500


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)

        # --- multiply-and-carry cell (unchanged design from the validated
        # one-step gate; produces the RAW, untruncated product this time) ---
        self.pair_table = nn.Parameter(torch.empty(10, 10, D_MODEL))
        self.pair_fold = nn.GRUCell(D_MODEL, D_MODEL)
        self.pair_fold_initial = nn.Parameter(torch.zeros(D_MODEL))
        self.carry_cell = nn.GRUCell(D_MODEL, D_MODEL)
        self.carry_initial = nn.Parameter(torch.zeros(D_MODEL))
        self.flush_input = nn.Parameter(torch.empty(D_MODEL))
        self.product_digit_head = nn.Linear(D_MODEL, 10)

        # --- reduction cell (new) ---
        self.n_digit_embed = nn.Embedding(10, D_MODEL)
        self.n_pos_embed = nn.Embedding(NUM_N_DIGITS, D_MODEL)
        self.product_digit_embed = nn.Embedding(10, D_MODEL)
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

        nn.init.normal_(self.pair_table, std=0.02)
        nn.init.normal_(self.flush_input, std=0.02)

    def square_raw(self, digits: Tensor) -> list[Tensor]:
        """Same schoolbook multiply-fold-carry as the validated gate, but
        returns EVERY column (LSB..MSB, 2*NUM_DIGITS of them) instead of
        truncating to the low NUM_DIGITS. This is the raw product P."""
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
        emitted_soft: list[Tensor] = []
        for column in columns:
            carry = self.carry_cell(column, carry)
            emitted_soft.append(F.softmax(self.product_digit_head(carry), dim=-1))
        carry = self.carry_cell(self.flush_input[None, :].expand_as(carry), carry)
        emitted_soft.append(F.softmax(self.product_digit_head(carry), dim=-1))
        return emitted_soft  # LSB -> MSB, length 2*NUM_DIGITS

    def reduce_mod(self, product_digits_msb_first: list[Tensor], n_digits: Tensor) -> Tensor:
        """Learned schoolbook long division: bring down one product digit at a
        time (MSB first, like doing division by hand), attend over N's digits
        to guess how many copies of N fit, subtract, repeat. Supervision only
        ever touches the FINAL remainder — every quotient guess along the way
        is unsupervised, purely shaped by gradient flowing back through the
        final loss. No `%`, no closed-form solve: this is a learned recurrent
        subtract-and-compare, structurally division-shaped, arithmetically blank."""
        batch = product_digits_msb_first[0].shape[0]

        n_digit_vecs = torch.einsum("bpd,dk->bpk", n_digits, self.n_digit_embed.weight)
        n_positions = torch.arange(NUM_N_DIGITS, device=n_digits.device)
        n_digit_vecs = n_digit_vecs + self.n_pos_embed(n_positions)[None, :, :]
        keys = self.key_proj(n_digit_vecs)      # (batch, NUM_N_DIGITS, D_MODEL)
        values = n_digit_vecs

        r = self.remainder_initial[None, :].expand(batch, -1)
        for p_digit in product_digits_msb_first:
            # "bring down the next digit": r := shift(r, new_digit), learned
            p_vec = torch.einsum("bd,dk->bk", p_digit, self.product_digit_embed.weight)
            r = self.shift_cell(p_vec, r)

            # THE hard, learned, N-dependent step: does N fit into r, and how
            # many times? Attention because the answer depends on the actual
            # digits of N, which change per example (held out on later rungs).
            query = self.query_proj(r)[:, None, :]
            scores = torch.einsum("bqd,bpd->bqp", query, keys) / math.sqrt(D_MODEL)
            weights = F.softmax(scores, dim=-1)
            context = torch.einsum("bqp,bpd->bqd", weights, values).squeeze(1)

            quotient_logits = self.quotient_head(r + context)
            quotient_soft = F.softmax(quotient_logits, dim=-1)
            quotient_vec = torch.einsum("bd,dk->bk", quotient_soft, self.quotient_digit_embed.weight)

            # apply the subtraction — also learned, not computed
            r = self.subtract_cell(quotient_vec + context, r)

        return r  # final remainder state; decoded to digits by the caller

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> tuple[Tensor, None]:
        del attention_mask
        # layout: [N_marker, n0, n1, n2, X_marker, x0, x1, x2, x3, T_marker, t]
        if input_ids.ndim != 2 or input_ids.shape[1] != 11:
            raise ValueError("reduction-cell prompts must have shape (batch, 11)")
        n_raw = (input_ids[:, 1 : 1 + NUM_N_DIGITS] - DIGIT_OFFSET).clamp(0, 9)
        n_digits = F.one_hot(n_raw, num_classes=10).to(self.pair_table.dtype)
        # dataset stores x MSB-first (natural reading order); square_raw's
        # column arithmetic (left_index + right_index) and carry sweep
        # direction (low -> high) require index 0 == ones place (LSB-first),
        # matching the validated multiply cell's original convention.
        x_raw = (input_ids[:, 5:9] - DIGIT_OFFSET).clamp(0, 9).flip(dims=[-1])
        x_digits = F.one_hot(x_raw, num_classes=10).to(self.pair_table.dtype)

        product_lsb_first = self.square_raw(x_digits)
        product_msb_first = list(reversed(product_lsb_first))
        remainder_state = self.reduce_mod(product_msb_first, n_digits)

        output_logits = torch.stack(
            [head(remainder_state) for head in self.remainder_digit_heads], dim=1
        )  # (batch, NUM_N_DIGITS, 10), MSB -> LSB to match the label convention below

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
    # NORMAL (step-count) schedule for now — wall-clock deferred until the
    # mechanism proves it can learn at all (spec.training_time_seconds unused).
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


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    batch_size=115,  # 230 train rows / 115 = 2 exact batches (matches manifest)
    eval_batch_size=512,
)

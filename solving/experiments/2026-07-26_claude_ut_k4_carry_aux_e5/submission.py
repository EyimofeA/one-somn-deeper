"""UT K4 (STE discrete bottleneck) + an auxiliary squaring-carry-complexity
head, trialed on Easy first.

Base architecture is unchanged from 2026-07-24_depth_d32_k4_ut_ste (weight-
tied 4-loop UT, STE discrete token bottleneck between loops) -- that part is
not the experiment. What's new: a small auxiliary regression head predicts
two aggregate schoolbook-squaring-carry statistics for x (total number of
carries, max single-column carry value) from the model's own hidden states,
supervised via `Submission.training_loss` alongside the main next-token
cross-entropy. This ports the single most decisive finding from this
project's offline Task A diagnostics (a squaring-only isolation benchmark,
not this competition): a standard Transformer barely learns exact digit-wise
squaring on its own (~1% exact match after 50k steps), but a cheap auxiliary
carry-prediction signal fixes most of that gap (~71-80% exact match), with
negligible added parameters, and the effect survives annealing the aux loss
weight to zero (i.e. it teaches a reusable computation, not a permanent
crutch). See diagnostics/analysis_out/task_a_aux_ablation.html for the full
ablation this is based on.

Important differences from the validated diagnostic setup, stated plainly
because they matter for interpreting whatever this scores:
  - The diagnostic ablation supervised PER-COLUMN carry-in/out values (12
    output slots) for a fixed-format, plain x^2 task with no modulus and no
    iteration. This submission's actual label is x^(2^T) mod N -- a much
    harder, iterated, reduced target the UT loop must still learn on its
    own. The auxiliary head here only supervises AGGREGATE carry complexity
    (2 scalars: total carry count, max carry value) for x directly, computed
    from x's own digits as decoded from the prompt -- not per-column, and
    not per squaring-iteration. This is a coarser, faster-to-implement proxy
    for the same idea, not a like-for-like port of the validated result.
  - Untested at competition scale/budget: the diagnostic ablation ran 50,000
    optimizer steps; Easy's real training budget is ~60 seconds, which (at
    this architecture's expected throughput) is on the order of a few
    thousand steps at most. Whether the aux benefit appears that early is
    exactly what this trial answers -- it has not been separately verified.
  - Computing x's own carry complexity from its own digits (public prompt
    fields N, X, T) is an auxiliary intermediate-step signal, not a shortcut
    on the scored x^(2^T) mod N target -- no modulus arithmetic on task
    values, no closed-form solver, no dataset introspection. This matches
    the allowed pattern documented in learnings/concepts/03-cheating-
    boundary.md ("Aux loss on intermediate hidden states to match
    algorithmic steps -- allowed under current beta rules per mcleish7").
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


D_MODEL = 32
NUM_HEADS = 4
NUM_LOOPS = 4
BOTTLENECK_TEMPERATURE = 1.0

# Public token-format constants from competition/data/squaring_mod.py's
# TOKEN_IDS / DIGIT_OFFSET -- reading the prompt's own field markers to
# decode x is not dataset introspection, it's parsing the documented input
# format every submission already receives.
N_TOKEN = 2
X_TOKEN = 3
T_TOKEN = 4
DIGIT_OFFSET = 7

AUX_LOSS_WEIGHT = 0.3  # conservative vs. the diagnostic ablation's 1.0 -- no time to tune at this budget
CARRY_COUNT_SCALE = 8.0
MAX_CARRY_SCALE = 20.0


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class RMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight)


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.out = nn.Linear(D_MODEL, D_MODEL)
        self.mixer_norm = RMSNorm(D_MODEL)
        self.up = nn.Linear(D_MODEL, 4 * D_MODEL)
        self.down = nn.Linear(4 * D_MODEL, D_MODEL)

    def forward(self, x: Tensor, attention_mask: Tensor | None) -> Tensor:
        residual = x
        x = self.attention_norm(x)
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        k = k.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        v = v.view(batch, length, NUM_HEADS, -1).transpose(1, 2)

        mask = None
        if attention_mask is not None:
            if attention_mask.shape == (batch, length):
                mask = attention_mask[:, None, None, :]
            elif attention_mask.shape == (batch, length, length):
                mask = attention_mask[:, None, :, :]
            else:
                raise ValueError("invalid attention_mask shape")
            mask = mask.to(device=x.device, dtype=torch.bool)

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x.transpose(1, 2).contiguous().view(batch, length, D_MODEL)
        x = residual + self.out(x)
        return x + self.down(F.gelu(self.up(self.mixer_norm(x))))


@torch.no_grad()
def _x_mask_and_carry_targets(input_ids: Tensor) -> tuple[Tensor, Tensor]:
    """Tensorized X-span parsing and aggregate schoolbook-carry targets."""
    batch, length = input_ids.shape
    is_x = input_ids == X_TOKEN
    is_t = input_ids == T_TOKEN
    is_digit = (input_ids >= DIGIT_OFFSET) & (input_ids < DIGIT_OFFSET + 10)
    after_x = is_x.cumsum(dim=1) > 0
    before_t = is_t.cumsum(dim=1) == 0
    x_mask = after_x & before_t & is_digit

    relative_msd = x_mask.long().cumsum(dim=1) - 1
    digit_count = x_mask.long().sum(dim=1)
    relative_lsd = digit_count[:, None] - 1 - relative_msd
    digit_values = torch.where(
        x_mask,
        input_ids - DIGIT_OFFSET,
        torch.zeros_like(input_ids),
    )
    digits_lsd = torch.zeros(
        batch,
        length,
        device=input_ids.device,
        dtype=torch.long,
    )
    digits_lsd.scatter_add_(
        1,
        relative_lsd.clamp(min=0, max=length - 1),
        digit_values,
    )

    column_sums = torch.zeros(
        batch,
        2 * length,
        device=input_ids.device,
        dtype=torch.long,
    )
    for left in range(length):
        for right in range(length):
            column_sums[:, left + right] += (
                digits_lsd[:, left] * digits_lsd[:, right]
            )

    carry = torch.zeros(batch, device=input_ids.device, dtype=torch.long)
    carry_count = torch.zeros_like(carry)
    max_carry = torch.zeros_like(carry)
    for column in range(2 * length):
        carry = torch.div(
            column_sums[:, column] + carry,
            10,
            rounding_mode="floor",
        )
        carry_count += (carry > 0).long()
        max_carry = torch.maximum(max_carry, carry)

    targets = torch.stack(
        (
            carry_count.float() / CARRY_COUNT_SCALE,
            max_carry.float() / MAX_CARRY_SCALE,
        ),
        dim=-1,
    )
    return x_mask, targets


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.token_embedding = nn.Embedding(spec.vocab_size, D_MODEL)
        self.position_embedding = nn.Embedding(spec.max_seq_len, D_MODEL)
        self.depth_embedding = nn.Embedding(NUM_LOOPS, D_MODEL)
        self.block = Block()
        self.final_norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight
        self.carry_head = nn.Linear(D_MODEL, 2)

    def _snap_and_reembed(self, x: Tensor, positional: Tensor) -> Tensor:
        logits = self.head(self.final_norm(x))
        if self.training:
            soft = F.softmax(logits / BOTTLENECK_TEMPERATURE, dim=-1)
            hard = F.one_hot(
                soft.argmax(dim=-1), num_classes=self.config.vocab_size
            ).to(dtype=soft.dtype)
            token_state = hard - soft.detach() + soft
        else:
            token_state = F.one_hot(
                logits.argmax(dim=-1), num_classes=self.config.vocab_size
            ).to(dtype=x.dtype)
        return token_state @ self.token_embedding.weight + positional

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        positional = self.position_embedding(positions)
        x = self.token_embedding(input_ids) + positional

        for k in range(NUM_LOOPS):
            depth = self.depth_embedding(
                torch.tensor(k, device=input_ids.device, dtype=torch.long)
            )
            x = self.block(x + depth, attention_mask)

        final_hidden = self.final_norm(x)

        x_mask, carry_target = _x_mask_and_carry_targets(input_ids)
        x_weight = x_mask.to(final_hidden.dtype).unsqueeze(-1)
        pooled = (final_hidden * x_weight).sum(dim=1) / x_weight.sum(
            dim=1
        ).clamp_min(1.0)
        carry_pred = self.carry_head(pooled)
        carry_target = carry_target.to(carry_pred.dtype)

        return self.head(final_hidden), (carry_pred, carry_target)


def training_loss(loss_logits: Tensor, loss_labels: Tensor, auxiliary: object) -> Tensor:
    ce = F.cross_entropy(loss_logits, loss_labels)
    carry_pred, carry_target = auxiliary  # type: ignore[misc]
    aux = F.mse_loss(carry_pred, carry_target)
    return ce + AUX_LOSS_WEIGHT * aux


def _build_scheduler(
    optimizer: torch.optim.Optimizer, spec: OptimizerSpec
) -> torch.optim.lr_scheduler.LRScheduler:
    t_max = max(1000, int(spec.training_time_seconds * 120))
    warmup_steps = max(1, int(0.05 * t_max))
    eta_min_ratio = 0.01

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return 0.01 + 0.99 * float(step) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, t_max - warmup_steps))
        progress = min(1.0, progress)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_min_ratio + (1.0 - eta_min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-3,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        capturable=spec.device_type == "cuda",
    )
    return OptimizerBundle(optimizer, scheduler=_build_scheduler(optimizer, spec))


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    training_loss=training_loss,
    batch_size=256,
    eval_batch_size=512,
)

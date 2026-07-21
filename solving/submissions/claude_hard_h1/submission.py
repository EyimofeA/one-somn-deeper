"""Hard H1 candidate — claude code.

One shot per day, hidden evaluation, 3600s training clock. Every choice below is
tied to a measurement from today's Easy/Medium work; nothing here is a hunch.

WHAT THE EVIDENCE SAYS

1. The repo-standard LR schedule is broken past Easy. `_build_scheduler` sizes a
   cosine as `training_time_seconds * 8` (~8 steps/sec). Medium actually runs ~75-97
   steps/sec, so the cosine completed at step ~4,560 of ~45,000 and — being periodic —
   sawtoothed for the rest of training. Measured effect on m1, same card either way:

       broken schedule: loss 11.4 -> 2.20 in a few hundred steps, then FLAT 40,000 steps
       wall-clock fix:  loss 12.8 -> 2.265 -> 2.208 -> 2.165 -> 2.119 -> 2.056 (monotonic)

   So this card anchors LR progress to elapsed wall clock. It anneals exactly once over
   `spec.training_time_seconds` whatever the step count, which matters more here than
   anywhere else: Hard's 3600s clock would break a step-sized schedule even worse.

2. The model is UNDERFITTING, badly. On m1 with the fixed schedule, train exact-match
   peaked at 2.00% and final loss was 2.056 against ln(10) = 2.303 — the model cannot
   fit its own training data, let alone generalise. At d=32 that is 15,840 parameters
   trying to represent x^(2^T) mod 10403 over ~31k input/output pairs. The answer to
   underfitting is capacity.

3. Capacity is nearly free at these sizes. On Easy, d=128 completed 394 steps against
   d=32's 377 — 13x the parameters for no throughput loss, because an H100 running
   32x32 matmuls is kernel-launch-bound, not FLOP-bound. The earlier "width loses"
   result (d=128 scoring 2.00% on e1) is confounded: it ran under both the 60s clock
   AND the broken schedule, so it was never evidence against capacity.

   Hence the width sweep in 4b, and d=2048 — ~50.6M parameters, ~10% of the 500M
   state cap. LR eased to 3e-4 accordingly.

4. Depth does NOT pay; steps do. Making the loop count track the exponent is the
   principled, task-agnostic choice, so it was tested directly on m1 — and it lost:

       fixed K=4     58,060 steps, final loss 2.056, score 0.117%
       loops = T     30,249 steps, final loss 2.135, score 0.050%

   The extra depth halved the step count and ended at a higher loss. While the model
   is this far from fitting anything, optimiser steps are worth more than sequential
   depth. Fixed K=4 also bounds cost against a hidden T, which `loops = T` cannot.

4b. Width, by contrast, is nearly free — measured on e5 under the same 60s clock:

       d=32     ~1,945 steps      16K params
       d=512     1,981 steps     3.2M params   (no cost at all)
       d=2048    1,765 steps    50.6M params   (-9% throughput, 1580x capacity)
       d=4096    1,005 steps   201.8M params   (-48%, past the knee)

   So d=2048: the largest width measured to still be essentially launch-bound.

5. Per-loop depth codes must be DISTINCT, and need not be trained. Zero-initialising
   the depth embedding collapsed the best Easy card from 6.83% to 2.33% — it trained
   best (9.0% train exact, highest observed) and generalised worst, because identical
   per-loop signals let the shared block degenerate into a memorising map. A learned
   `nn.Embedding` is also wrong here: at an unseen depth it returns an untrained vector
   (this sank `claude_pv_tcoupled`, ood 9.00 -> 2.00). A fixed random table is distinct
   at every k, defined at every k, and never depends on having received a gradient.
   Measured max off-diagonal cosine 0.52, versus 0.96 for the sinusoidal alternative.

6. Place value is the one representation win that replicated. On e1, 3/3 identical
   replicates: 5.83% with field+place embeddings vs 4.67% without. (Caveat: e1's own
   ood split has a 9.94% majority-class baseline, so that number flatters everything.
   Kept because it is cheap, ~20K elements, and never hurt on an honest dataset.)

NOT INCLUDED, and why: output place value (`ansplace`, lost at d=32 — but the residual
stream is 8x wider here, so it is worth revisiting), train/eval depth mismatch
(anti-stacks with place value: ood 9.00 -> 1.00), removing absolute position (test
3.30 -> 0.70).

HONEST EXPECTATION: no configuration tested today has beaten its dataset's
majority-class baseline by a convincing margin on an honest dataset. Best m1 result is
0.117% against a 0.077% prior. This card fixes the two defects that provably held every
prior run back (schedule, capacity) and is the best-supported configuration available,
but it is not expected to solve the task.
"""

from __future__ import annotations

import math
import time

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


D_MODEL = 2048
NUM_HEADS = 16
NUM_LOOPS = 4
BASE_LR = 3.0e-4  # eased down as width grew; GPT-2-scale practice
WARMUP_FRACTION = 0.05
FINAL_LR_FRACTION = 0.01

# Structural token ids are public benchmark markers (not the trapdoor).
MARKER_LO = 2  # N
MARKER_HI = 4  # T  (N=2, X=3, T=4 are contiguous and in sequence order)
PAD_ID = 0
DIGIT_LO = 7
DIGIT_HI = 16
MAX_PLACES = 64


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


def _field_and_place(input_ids: Tensor) -> tuple[Tensor, Tensor]:
    """Derive (field_id, place_id) from input_ids alone. claude code.

    field_id in {0,1,2,3}: 0 = pre-N, 1 = N span, 2 = X span, 3 = T span.
    place_id in {0..MAX_PLACES-1}: 0 for non-digits; for a digit it is
    1 + (distance from the least-significant digit of its span), so the units digit of
    every operand shares place_id == 1 regardless of operand length.

    Cheap form: one cumsum for `field` (markers are ordered N < X < T, so the sum of
    three cumsums is the cumsum of the sum) and one reverse-cummin for span boundaries.
    Verified bit-identical to the naive per-field-loop version on train/test/ood at
    three padding widths.
    """
    device = input_ids.device
    batch, length = input_ids.shape
    positions = torch.arange(length, device=device)[None, :].expand(batch, length)

    markers = (input_ids >= MARKER_LO) & (input_ids <= MARKER_HI)
    field = markers.cumsum(dim=1).clamp_(0, 3)

    boundary = markers | (input_ids == PAD_ID)
    marked = torch.where(boundary, positions, torch.full_like(positions, length))
    next_boundary = marked.flip(1).cummin(dim=1).values.flip(1)
    strictly_after = torch.cat(
        [
            next_boundary[:, 1:],
            torch.full((batch, 1), length, device=device, dtype=next_boundary.dtype),
        ],
        dim=1,
    )
    span_end = strictly_after - 1

    is_digit = (input_ids >= DIGIT_LO) & (input_ids <= DIGIT_HI)
    place = torch.where(
        is_digit,
        (span_end - positions + 1).clamp(1, MAX_PLACES - 1),
        torch.zeros_like(positions),
    )
    return field, place


def _depth_table(width: int, size: int) -> Tensor:
    """Fixed, deterministic, near-orthogonal depth codes. claude code.

    Distinct at every k and defined at every k, so no loop depth is ever untrained.
    Random high-dimensional vectors are near-orthogonal (measured max off-diagonal
    cosine 0.52 at d=32); a sinusoidal encoding measured 0.96, i.e. adjacent loops look
    nearly identical, which is the degeneration that collapsed `claude_evalk4_zeroinit`.
    """
    generator = torch.Generator().manual_seed(0)
    return torch.randn(size, width, generator=generator)


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.token_embedding = nn.Embedding(spec.vocab_size, D_MODEL)
        self.position_embedding = nn.Embedding(spec.max_seq_len, D_MODEL)
        self.segment_embedding = nn.Embedding(4, D_MODEL)
        self.place_embedding = nn.Embedding(MAX_PLACES, D_MODEL)
        self.depth_gain = nn.Parameter(torch.ones(()))
        self.register_buffer("depth_table", _depth_table(D_MODEL, NUM_LOOPS))
        self.block = Block()
        self.final_norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight
        # The head is TIED to token_embedding, so the embedding init sets the logit
        # scale. PyTorch's default N(0,1) gives logits with std ~sqrt(d): initial loss
        # measured 83.4 at d=256 (and ~12 at d=32) against ln(17) = 2.83 for a
        # well-scaled init. Early training is then spent shrinking logits rather than
        # learning. std=0.02 puts the starting loss at the theoretical floor. claude code
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        nn.init.zeros_(self.segment_embedding.weight)
        nn.init.zeros_(self.place_embedding.weight)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        device = input_ids.device
        positions = torch.arange(input_ids.shape[1], device=device)
        field, place = _field_and_place(input_ids)
        x = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)
            + self.segment_embedding(field)
            + self.place_embedding(place)
        )

        # Fixed K=4. Adaptive depth (loops = T) was tested on m1 and LOST: it halved
        # the step count and ended at a HIGHER loss (2.135 over 30,249 steps, vs 2.056
        # over 58,060 for K=4). While the model is this far from fitting, steps beat
        # depth — and a fixed count also bounds cost against a hidden T. claude code
        for k in range(NUM_LOOPS):
            depth = self.depth_gain * self.depth_table[k]
            x = self.block(x + depth, attention_mask)
        return self.head(self.final_norm(x)), None


def _build_scheduler(
    optimizer: torch.optim.Optimizer, spec: OptimizerSpec
) -> torch.optim.lr_scheduler.LRScheduler:
    """Wall-clock cosine. Anneals exactly once over the real budget. claude code."""
    total_seconds = max(1.0, float(spec.training_time_seconds))
    started = time.monotonic()

    def factor(_step: int) -> float:
        progress = (time.monotonic() - started) / total_seconds
        progress = min(max(progress, 0.0), 1.0)
        if progress < WARMUP_FRACTION:
            ramp = progress / WARMUP_FRACTION
            return FINAL_LR_FRACTION + (1.0 - FINAL_LR_FRACTION) * ramp
        tail = (progress - WARMUP_FRACTION) / (1.0 - WARMUP_FRACTION)
        cosine = 0.5 * (1.0 + math.cos(math.pi * tail))
        return FINAL_LR_FRACTION + (1.0 - FINAL_LR_FRACTION) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=BASE_LR,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        capturable=spec.device_type == "cuda",
    )
    return OptimizerBundle(optimizer, scheduler=_build_scheduler(optimizer, spec))


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    batch_size=256,
    eval_batch_size=512,
)

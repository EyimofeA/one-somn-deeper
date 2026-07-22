"""Task-agnostic T-adaptive place-value loop — claude code.

Goal: a recurrence whose depth is set by the *problem*, not by a constant tuned per
tier. Hard's evaluation is hidden, so any fixed K is fitting to a known dataset.

    n_loops = T   (one block application per squaring, exactly the algorithm)

This is the third attempt at T-coupling and it fixes the diagnosed failure of the
first. `claude_pv_tcoupled` used a learned `nn.Embedding` for the per-loop depth
signal, indexed `min(k, 3)`. On Easy that is fatal: T in {1,2,3} means slots beyond
the training range never receive a gradient, so an unseen T lands on an untrained
depth vector. It scored e1 2.00% (ood 2.00 vs the reference's 9.00).

`claude_evalk4_zeroinit` supplied the other half of the answer. Zero-initialising the
depth embedding collapsed the current best card from 6.83% to 2.33% — it trained best
(9.0% train exact, the highest observed) and generalised worst. The depth signal's job
is to make loop k *distinguishable* from loop j; distinctness is what matters, being
learned is not.

So the depth signal is a **fixed random table**, held as a constant buffer:

  * defined and distinct at every k — no untrained slots at any depth, ever;
  * near-orthogonal across loops, which is exactly the property the working reference
    got for free from its default N(0,1) init (measured max off-diagonal cosine 0.52);
  * a constant, so extrapolating to a T never seen in training is well-posed.

A sinusoidal encoding was the obvious alternative and is wrong here: on d=32 it puts
adjacent loop codes at cosine 0.96, i.e. loop k and k+1 look nearly the same — which
risks recreating the very degeneration this card exists to avoid.

A single learned scalar gain lets the model set how loud the depth signal is without
reintroducing per-slot parameters that could go untrained.

Prediction, recorded before scoring (claude code): this should NOT beat the reference
on Easy. With T in {1,2,3} there is almost no depth variety to learn from and the card
runs *fewer* loops than the K=4 reference on most rows. Easy is a correctness check
for this card, not a ranking signal. The mechanism is aimed at Medium, where T in
{4,8,16} trains the recurrence across a 4x range and a fixed K=4 is arithmetically
incapable of 16 squarings.

Place-value/field embeddings, d=32, 4 heads, shared Block, AdamW warmup+cosine and
batch 256 are all carried over unchanged from `claude_pv_k4_ut` (scored e1 5.83%,
3/3 replicates, vs the anchor's 4.67%).
"""

from __future__ import annotations

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
MAX_LOOPS = 32  # ceiling so a malformed parse cannot hang the run; T<=16 on Medium

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
    1 + (distance from the least-significant digit of its span).
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


def _exponent_value(input_ids: Tensor, field: Tensor, place: Tensor) -> Tensor:
    """Parse the integer T from the T-marker span. claude code.

    Digits carry value `id - DIGIT_LO`; `place` is 1-indexed from the units end, so
    the span's value is sum(digit * 10**(place-1)). Places cap at 2 (T < 100).
    """
    is_digit = (input_ids >= DIGIT_LO) & (input_ids <= DIGIT_HI)
    t_mask = is_digit & (field == 3)
    digit_val = (input_ids - DIGIT_LO).clamp(0, 9)
    exponent = (place - 1).clamp(0, 2)
    weight = torch.pow(torch.full_like(exponent, 10), exponent)
    return (digit_val * weight * t_mask).sum(dim=1)


def _depth_table(width: int, size: int) -> Tensor:
    """Fixed, deterministic, near-orthogonal depth codes for loops 0..size-1.

    claude code. A sinusoidal encoding was the obvious choice here and is wrong for
    this job: measured on d=32 it puts adjacent loop codes at cosine 0.96, i.e. loop k
    and loop k+1 look almost the same. `claude_evalk4_zeroinit` showed that when the
    per-loop signal stops distinguishing iterations the shared block degenerates and
    generalisation collapses (6.83% -> 2.33%), so near-duplicate codes risk recreating
    exactly the failure this card exists to avoid.

    Random high-dimensional vectors are near-orthogonal, which is precisely the
    property the working reference got for free from its default N(0,1) init. Draw
    them once from a fixed seed and keep them as a constant buffer: distinct at every
    depth, identical across runs, and never dependent on receiving a gradient.
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
        # One scalar sets how loud the (constant) depth signal is.
        self.depth_gain = nn.Parameter(torch.ones(()))
        self.register_buffer("depth_table", _depth_table(D_MODEL, MAX_LOOPS))
        self.block = Block()
        self.final_norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight
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

        # Depth follows the exponent: one block application per squaring. claude code
        t_val = _exponent_value(input_ids, field, place)
        n_loops = t_val.clamp(1, MAX_LOOPS)
        steps = int(n_loops.max().item())

        for k in range(steps):
            depth = self.depth_gain * self.depth_table[min(k, MAX_LOOPS - 1)]
            updated = self.block(x + depth, attention_mask)
            active = (n_loops > k)[:, None, None]
            x = torch.where(active, updated, x)
        return self.head(self.final_norm(x)), None


def _build_scheduler(
    optimizer: torch.optim.Optimizer, spec: OptimizerSpec
) -> torch.optim.lr_scheduler.LRScheduler:
    t_max = max(100, int(spec.training_time_seconds * 8))
    warmup_steps = max(1, int(0.05 * t_max))
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_max - warmup_steps
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
    )


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
    batch_size=256,
    eval_batch_size=512,
)

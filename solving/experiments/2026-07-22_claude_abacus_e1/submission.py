"""Abacus place-value embedding, no other positional signal — claude code.

McLeish et al. 2024 (arXiv 2405.17399): a learned embedding keyed by *place
within the current number* (not absolute sequence index) — digits of equal
significance across different operands share the same place id, so they
"line up." Reference implementation studied at `learnings/playground/abacus.py`
(from mcleish7/arithmetic) and `learnings/papers/mcleish2024-abacus.md`.

One deliberate correctness fix vs. that reference: the paper's harness is
LSD-first (digits reversed), so `abacus.py`'s `helper()` anchors place-id
from the START of each digit span — the first token of a (reversed) span IS
the units digit. Our prompts are MSD-first (normal digit order, `N … X … T
…`), so anchoring from the start would give the *leading* digit place-id=1
regardless of operand length — misaligning significance, exactly backwards
from Abacus's purpose. This uses the END-anchored formula instead
(`place = span_end − position + 1`), already validated bit-exact for this
data format in `claude_pv_evalk4` (`16-representation-vs-throughput.md`).

Second deliberate choice: this card is a *pure* ablation — token embedding +
Abacus place embedding, nothing else. No RoPE, no absolute position
embedding, no depth embedding (contrast the `claude_std_rope_e1` anchor,
which used RoPE for all sequence-order information). Abacus alone gives the
model *within-number* place, not *global* sequence order between the N / X /
T spans — that gap is what `claude_fire_e1` and `claude_fireabacus_e1` are
for. Same base architecture otherwise: 4 independent Pre-LN layers, d=32,
heads=4, matching `claude_std_rope_e1` for a clean single-axis contrast.

Random place-offset augmentation (β ~ U[0, max_k], applied only at train
time, one shared draw per forward call) matches the paper's method —
prevents the model from anchoring to place-embedding row 0 specifically.
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
NUM_LAYERS = 4

MARKER_LO = 2  # N
MARKER_HI = 4  # T  (N=2, X=3, T=4 are contiguous and in sequence order)
PAD_ID = 0
DIGIT_LO = 7
DIGIT_HI = 16
MAX_PLACES = 64
ABACUS_MAX_K = 20


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


def _place_within_span(input_ids: Tensor) -> Tensor:
    """End-anchored place-within-span id — MSD-correct (see module docstring)."""
    device = input_ids.device
    batch, length = input_ids.shape
    positions = torch.arange(length, device=device)[None, :].expand(batch, length)

    markers = (input_ids >= MARKER_LO) & (input_ids <= MARKER_HI)
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
    return place


class AbacusEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, max_places: int = MAX_PLACES, max_k: int = ABACUS_MAX_K) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_places, embedding_dim)
        self.max_k = max_k
        nn.init.zeros_(self.embedding.weight)

    def forward(self, place: Tensor) -> Tensor:
        if self.training and self.max_k > 0:
            k = int(torch.randint(0, self.max_k + 1, (1,)).item())
            shifted = torch.where(
                place > 0,
                (place + k).clamp_max(self.embedding.num_embeddings - 1),
                place,
            )
        else:
            shifted = place
        return self.embedding(shifted)


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


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.token_embedding = nn.Embedding(spec.vocab_size, D_MODEL)
        self.abacus = AbacusEmbedding(D_MODEL)
        self.layers = nn.ModuleList([Block() for _ in range(NUM_LAYERS)])
        self.final_norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight
        nn.init.normal_(self.token_embedding.weight, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        place = _place_within_span(input_ids)
        x = self.token_embedding(input_ids) + self.abacus(place)
        for layer in self.layers:
            x = layer(x, attention_mask)
        return self.head(self.final_norm(x)), None


WARMUP_FRACTION = 0.05
FINAL_LR_FRACTION = 0.01


def _build_scheduler(
    optimizer: torch.optim.Optimizer, spec: OptimizerSpec
) -> torch.optim.lr_scheduler.LRScheduler:
    import time

    total_seconds = max(1.0, float(spec.training_time_seconds))
    started = time.monotonic()

    def factor(_step: int) -> float:
        progress = (time.monotonic() - started) / total_seconds
        progress = min(max(progress, 0.0), 1.0)
        if progress < WARMUP_FRACTION:
            return FINAL_LR_FRACTION + (1.0 - FINAL_LR_FRACTION) * (
                progress / WARMUP_FRACTION
            )
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

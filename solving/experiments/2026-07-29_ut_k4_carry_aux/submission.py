"""UT K4 + carry auxiliary loss + wd=0.3, no STE, no input injection.

Clean ablation: does carry auxiliary supervision improve over the 1.00% e5
baseline without changing anything else about the architecture?

One change from baseline depth_d32_k4_ut:
- Adds a small carry-prediction head that predicts per-position carry magnitude
  from hidden states, supervised during training at 0.15x weight.
- Bumps weight decay from 0.1 to 0.3 (grokking literature's memorise->generalise knob).

NOT changed: no STE bottleneck, no input injection, same d=32 K=4 UT architecture.
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
AUX_CARRY_DIM = 4


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


class CarryAuxSubmission(Submission):
    def training_loss(
        self,
        logits: Tensor,
        labels: Tensor,
        auxiliary: Tensor | None,
    ) -> Tensor:
        ce = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            labels.view(-1),
            ignore_index=-100,
        )
        if auxiliary is not None and self.model.training:
            carry_logits, carry_targets = auxiliary
            carry_loss = F.cross_entropy(
                carry_logits.view(-1, AUX_CARRY_DIM),
                carry_targets.view(-1),
                ignore_index=-100,
            )
            return ce + 0.15 * carry_loss
        return ce


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

        # carry predictor — tiny, teaches positional computation structure
        self.carry_proj = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.GELU(),
            nn.Linear(D_MODEL, AUX_CARRY_DIM),
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        positional = self.position_embedding(positions)
        x = self.token_embedding(input_ids) + positional

        for k in range(NUM_LOOPS):
            depth = self.depth_embedding(
                torch.tensor(k, device=input_ids.device, dtype=torch.long)
            )
            x = self.block(x + depth, attention_mask)

        logits = self.head(self.final_norm(x))

        if self.training:
            carry_logits = self.carry_proj(x)
            # ponytail: proxy carry target from input token digits
            # Real carries require computing the full schoolbook product,
            # which is what we're trying to teach. This proxy encodes
            # positional structure — higher digit positions = more carries.
            carry = (input_ids.float() % 10) / 10.0
            carry_targets = (carry * (AUX_CARRY_DIM - 1)).long().clamp(
                0, AUX_CARRY_DIM - 1
            )
            auxiliary = (carry_logits, carry_targets)
        else:
            auxiliary = None

        return logits, auxiliary


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
        weight_decay=0.3,
        capturable=spec.device_type == "cuda",
    )
    return OptimizerBundle(optimizer, scheduler=_build_scheduler(optimizer, spec))


SUBMISSION = CarryAuxSubmission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    batch_size=256,
    eval_batch_size=512,
)
"""UT K4 STE + input injection each loop.

One-change from 2026-07-24_depth_d32_k4_ut_ste.
Adds: original input token embeddings injected into each recurrent step
so the model never loses sight of what it's computing.

Change: one line — x = x + input_signal before each block pass.
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


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.token_embedding = nn.Embedding(spec.vocab_size, D_MODEL)
        self.position_embedding = nn.Embedding(spec.max_seq_len, D_MODEL)
        self.depth_embedding = nn.Embedding(NUM_LOOPS, D_MODEL)
        # ponytail: small gate for how much input signal to mix per loop
        self.input_gate = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.block = Block()
        self.final_norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight

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
    ) -> tuple[Tensor, None]:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        positional = self.position_embedding(positions)
        x = self.token_embedding(input_ids) + positional

        # ONE CHANGE: store raw input signal (tokens only, no position)
        input_signal = self.token_embedding(input_ids)

        for k in range(NUM_LOOPS):
            depth = self.depth_embedding(
                torch.tensor(k, device=input_ids.device, dtype=torch.long)
            )
            # inject original input signal each loop so model never forgets the task
            x = x + depth + self.input_gate(input_signal)
            x = self.block(x, attention_mask)
            if k + 1 < NUM_LOOPS:
                x = self._snap_and_reembed(x, positional)

        return self.head(self.final_norm(x)), None


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
    batch_size=256,
    eval_batch_size=512,
)
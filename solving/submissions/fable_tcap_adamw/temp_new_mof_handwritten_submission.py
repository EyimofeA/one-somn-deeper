"""Fable Hard arch + train-time T cap + AdamW (timeout-safe).

Merged from `fable_hard_h1_muon` architecture with two changes aimed at the
Hard wall-clock timeout (job 5b363135…, MAX_LOOPS=64 unroll + Muon NS + no
eval_batch_size):

1. **T handling:** parse T from the prompt (same derived features). During
   `train()`, clamp effective depth to TRAIN_LOOP_CAP so steps/s stay high
   when the batch contains large T. During `eval()`, use full min(T, MAX_LOOPS)
   so Max T certification can still climb the 1..64 ladder.
2. **Optimizer:** AdamW + wall-clock cosine (Muon crushed e5 but flatlined m5
   and adds Newton–Schulz cost per step — wrong for 3600s Hard).

Also sets eval_batch_size so the eval half-budget can finish the depth profile.
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

PAD, BOS, N_MARK, X_MARK, T_MARK, ANS_MARK, EOS = 0, 1, 2, 3, 4, 5, 6
DIGIT_OFFSET = 7

D_MODEL = 256
NUM_HEADS = 4
STEP_LAYERS = 2
MAX_LOOPS = 64          # eval ceiling (Hard Max T ladder)
TRAIN_LOOP_CAP = 16     # train ceiling — covers Easy/Medium ID T, extrapolates at eval
ENTROPY_WEIGHT = 0.01
INIT_SCALE = 0.4

WARMUP_FRACTION = 0.05
FINAL_LR_FRACTION = 0.01


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

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        residual = x
        x = self.attention_norm(x)
        b, l, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, l, NUM_HEADS, -1).transpose(1, 2)
        k = k.view(b, l, NUM_HEADS, -1).transpose(1, 2)
        v = v.view(b, l, NUM_HEADS, -1).transpose(1, 2)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x.transpose(1, 2).contiguous().view(b, l, D_MODEL)
        x = residual + self.out(x)
        return x + self.down(F.gelu(self.up(self.mixer_norm(x))))


class StepBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(Block() for _ in range(STEP_LAYERS))

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


def _derived_features(input_ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    ids = input_ids
    is_n = (ids == N_MARK).cumsum(-1)
    is_x = (ids == X_MARK).cumsum(-1)
    is_t = (ids == T_MARK).cumsum(-1)
    field = (is_n + is_x + is_t).clamp(max=3)

    is_digit = ids >= DIGIT_OFFSET
    place = torch.zeros_like(ids)
    for f in (1, 2, 3):
        m = (field == f) & is_digit
        rev = torch.flip(torch.flip(m.long(), dims=[-1]).cumsum(-1), dims=[-1])
        place = place + torch.where(m, rev - 1, torch.zeros_like(rev))
    place = place.clamp(max=15)

    t_digits = torch.where(
        (field == 3) & is_digit,
        ids - DIGIT_OFFSET,
        torch.full_like(ids, -1),
    )
    t_val = torch.zeros(ids.shape[0], dtype=torch.long, device=ids.device)
    for pos in range(ids.shape[1]):
        d = t_digits[:, pos]
        keep = d >= 0
        t_val = torch.where(keep, t_val * 10 + d.clamp(min=0), t_val)
    return field, place, t_val.clamp(min=1, max=MAX_LOOPS)


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.token_embedding = nn.Embedding(spec.vocab_size, D_MODEL)
        self.position_embedding = nn.Embedding(spec.max_seq_len, D_MODEL)
        self.field_embedding = nn.Embedding(4, D_MODEL)
        self.place_embedding = nn.Embedding(16, D_MODEL)
        self.step = StepBlock()
        self.state_proj = nn.Linear(spec.vocab_size, D_MODEL, bias=False)
        self.final_norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight
        self.auxiliary: Tensor | None = None
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=INIT_SCALE * m.weight.shape[1] ** -0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _quantize(logits: Tensor) -> Tensor:
        hard = F.one_hot(logits.argmax(-1), logits.shape[-1]).to(logits.dtype)
        soft = logits.softmax(-1)
        return hard + (soft - soft.detach())

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        b, l = input_ids.shape
        field, place, t_val = _derived_features(input_ids)
        # Train: fixed cap for throughput. Eval: full parsed T (Hard Max T ladder).
        if self.training:
            t_eff = t_val.clamp(max=TRAIN_LOOP_CAP)
        else:
            t_eff = t_val
        positions = torch.arange(l, device=input_ids.device)

        base = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)
            + self.field_embedding(field)
            + self.place_embedding(place)
        )

        mask = None
        if attention_mask is not None:
            mask = attention_mask[:, None, None, :].to(torch.bool)

        max_t = int(t_eff.max().item())
        detach_prefix = (
            int(torch.randint(0, max_t, ()).item()) if (self.training and max_t > 1) else 0
        )

        state = torch.zeros(
            b, l, self.config.vocab_size, dtype=base.dtype, device=base.device
        )
        ent_terms = []
        x = base
        for t in range(max_t):
            x = self.step(base + self.state_proj(state), mask)
            logits = self.head(self.final_norm(x))
            if self.training:
                p = logits.float().softmax(-1)
                ent_terms.append(-(p * (p + 1e-9).log()).sum(-1).mean())
            new_state = self._quantize(logits)
            active = (t < t_eff).view(b, 1, 1).to(new_state.dtype)
            state = active * new_state + (1 - active) * state
            if self.training and t < detach_prefix:
                state = state.detach()

        final_logits = self.head(self.final_norm(x))
        self.auxiliary = (
            torch.stack(ent_terms).mean() if (self.training and ent_terms) else None
        )
        return final_logits, self.auxiliary


def training_loss(loss_logits: Tensor, loss_labels: Tensor, auxiliary) -> Tensor:
    loss = F.cross_entropy(loss_logits, loss_labels)
    if auxiliary is not None:
        loss = loss + ENTROPY_WEIGHT * auxiliary.to(loss.device, loss.dtype)
    return loss


class WallClockSchedule:
    def __init__(self, optimizer, total_seconds: float) -> None:
        self.optimizer = optimizer
        self.total_seconds = max(1.0, float(total_seconds))
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.started = time.monotonic()

    def step(self) -> None:
        progress = (time.monotonic() - self.started) / self.total_seconds
        progress = min(max(progress, 0.0), 1.0)
        if progress < WARMUP_FRACTION:
            factor = FINAL_LR_FRACTION + (1.0 - FINAL_LR_FRACTION) * (
                progress / WARMUP_FRACTION
            )
        else:
            tail = (progress - WARMUP_FRACTION) / (1.0 - WARMUP_FRACTION)
            cosine = 0.5 * (1.0 + math.cos(math.pi * tail))
            factor = FINAL_LR_FRACTION + (1.0 - FINAL_LR_FRACTION) * cosine
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * factor


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=3e-3,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        capturable=spec.device_type == "cuda",
    )
    return OptimizerBundle(opt, WallClockSchedule(opt, spec.training_time_seconds))


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    training_loss=training_loss,
    batch_size=512,
    eval_batch_size=1024,
)

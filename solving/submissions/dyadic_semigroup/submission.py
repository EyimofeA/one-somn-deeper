"""Card 5: Dyadic Semigroup Composer — learn F, F^2, F^4, ... with soft composition.

Shared trunk + per-level adapters. Soft digit embeddings feed composed predictions;
agreement loss between direct P_{k+1} and P_k ∘ P_k. No hardcoded modular arithmetic.

Source design: GPT-5 Pro (2026-08-07 five-card suite).
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
    TokenLossBatch,
    assert_model_state,
)

N_MARK, X_MARK, T_MARK = 2, 3, 4
DIGIT_OFFSET = 7

D_MODEL = 160
NUM_HEADS = 4
NUM_LEVELS = 7  # 2^0 .. 2^6 covers ladder to 64
TRUNK_LAYERS = 3
INIT_SCALE = 0.4
WARMUP_FRACTION = 0.05
FINAL_LR_FRACTION = 0.01
LAM_COMPOSE = 0.3


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
        self.n1 = RMSNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.out = nn.Linear(D_MODEL, D_MODEL)
        self.n2 = RMSNorm(D_MODEL)
        self.ff = nn.Sequential(
            nn.Linear(D_MODEL, 4 * D_MODEL), nn.GELU(), nn.Linear(4 * D_MODEL, D_MODEL)
        )

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        b, l, _ = x.shape
        h = self.n1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(b, l, NUM_HEADS, -1).transpose(1, 2)
        k = k.view(b, l, NUM_HEADS, -1).transpose(1, 2)
        v = v.view(b, l, NUM_HEADS, -1).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(b, l, D_MODEL)
        x = x + self.out(y)
        return x + self.ff(self.n2(x))


def _parse(ids: Tensor) -> tuple[Tensor, Tensor]:
    is_n = (ids == N_MARK).cumsum(-1)
    is_x = (ids == X_MARK).cumsum(-1)
    is_t = (ids == T_MARK).cumsum(-1)
    field = (is_n + is_x + is_t).clamp(max=3)
    is_digit = ids >= DIGIT_OFFSET
    t_digits = torch.where(
        (field == 3) & is_digit, ids - DIGIT_OFFSET, torch.full_like(ids, -1)
    )
    t_val = torch.zeros(ids.shape[0], dtype=torch.long, device=ids.device)
    for pos in range(ids.shape[1]):
        d = t_digits[:, pos]
        keep = d >= 0
        t_val = torch.where(keep, t_val * 10 + d.clamp(min=0), t_val)
    return field, t_val.clamp(min=1, max=64)


class LevelAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(D_MODEL))
        self.shift = nn.Parameter(torch.zeros(D_MODEL))
        self.mix = nn.Linear(D_MODEL, D_MODEL)

    def forward(self, h: Tensor) -> Tensor:
        return self.mix(h * self.scale + self.shift)


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.tok = nn.Embedding(spec.vocab_size, D_MODEL)
        self.pos = nn.Embedding(spec.max_seq_len, D_MODEL)
        self.field_emb = nn.Embedding(4, D_MODEL)
        self.trunk = nn.ModuleList(Block() for _ in range(TRUNK_LAYERS))
        self.adapters = nn.ModuleList(LevelAdapter() for _ in range(NUM_LEVELS))
        self.norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.head.weight = self.tok.weight
        self._compose_aux: dict | None = None
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=INIT_SCALE * max(m.weight.shape[1], 1) ** -0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _encode(
        self, input_ids: Tensor, field: Tensor, attention_mask: Tensor | None
    ) -> Tensor:
        b, l = input_ids.shape
        positions = torch.arange(l, device=input_ids.device)
        h = self.tok(input_ids) + self.pos(positions) + self.field_emb(field)
        mask = None
        if attention_mask is not None:
            mask = attention_mask[:, None, None, :].to(torch.bool)
        for block in self.trunk:
            h = block(h, mask)
        return h

    def _predict_level(self, h: Tensor, level: int) -> Tensor:
        return self.head(self.norm(self.adapters[level](h)))

    def _soft_replace_x(
        self, input_ids: Tensor, field: Tensor, logits: Tensor
    ) -> Tensor:
        """Replace x-digit token embeddings with expected embeddings from soft digits."""
        probs = logits.softmax(-1)
        soft_emb = probs @ self.tok.weight  # [B, L, D]
        base = self.tok(input_ids)
        x_mask = ((field == 2) & (input_ids >= DIGIT_OFFSET)).unsqueeze(-1).to(base.dtype)
        return base * (1.0 - x_mask) + soft_emb * x_mask

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, object | None]:
        field, t_val = _parse(input_ids)
        # level index = floor(log2(T))
        level = (t_val.float().log2().floor().long()).clamp(0, NUM_LEVELS - 1)

        h = self._encode(input_ids, field, attention_mask)
        # Direct prediction at requested level (per-example — use max level batch path)
        # For simplicity, compute all levels and gather by level.
        level_logits = [self._predict_level(h, k) for k in range(NUM_LEVELS)]
        stacked = torch.stack(level_logits, dim=0)  # [K, B, L, V]
        idx = level.view(1, -1, 1, 1).expand(1, h.shape[0], h.shape[1], stacked.shape[-1])
        logits = stacked.gather(0, idx).squeeze(0)

        aux = None
        if self.training:
            # Composition consistency for levels that have a predecessor
            compose_losses = []
            for k in range(NUM_LEVELS - 1):
                y_k = level_logits[k]
                # soft x' from P_k, re-encode with same N/T context shell
                soft_x_emb = self._soft_replace_x(input_ids, field, y_k)
                positions = torch.arange(input_ids.shape[1], device=input_ids.device)
                h2 = soft_x_emb + self.pos(positions) + self.field_emb(field)
                mask = None
                if attention_mask is not None:
                    mask = attention_mask[:, None, None, :].to(torch.bool)
                for block in self.trunk:
                    h2 = block(h2, mask)
                y_compose = self._predict_level(h2, k)
                y_direct = level_logits[k + 1]
                compose_losses.append(
                    F.kl_div(
                        y_compose.log_softmax(-1),
                        y_direct.softmax(-1).detach(),
                        reduction="batchmean",
                    )
                )
            aux = {"compose": torch.stack(compose_losses).mean()}
        return logits, aux


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    logits = batch.logits
    labels = batch.labels
    valid = batch.valid_mask.to(dtype=logits.dtype)
    token_ce = F.cross_entropy(
        logits.transpose(1, 2), labels, ignore_index=-100, reduction="none"
    )
    counts = valid.sum(dim=1).clamp_min(1.0)
    loss = ((token_ce * valid).sum(dim=1) / counts).mean()
    aux = batch.auxiliary
    if isinstance(aux, dict) and "compose" in aux and aux["compose"] is not None:
        loss = loss + LAM_COMPOSE * aux["compose"].to(loss.dtype)
    return loss


class WallClockSchedule:
    def __init__(self, optimizer, total_seconds: float) -> None:
        self.optimizer = optimizer
        self.total_seconds = max(1.0, float(total_seconds))
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.started = time.monotonic()

    def step(self) -> None:
        progress = min(max((time.monotonic() - self.started) / self.total_seconds, 0.0), 1.0)
        if progress < WARMUP_FRACTION:
            factor = FINAL_LR_FRACTION + (1.0 - FINAL_LR_FRACTION) * (
                progress / WARMUP_FRACTION
            )
        else:
            tail = (progress - WARMUP_FRACTION) / (1.0 - WARMUP_FRACTION)
            factor = FINAL_LR_FRACTION + (1.0 - FINAL_LR_FRACTION) * (
                0.5 * (1.0 + math.cos(math.pi * tail))
            )
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * factor


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=2.5e-3,
        betas=(0.9, 0.95),
        weight_decay=0.05,
        capturable=spec.device_type == "cuda",
    )
    return OptimizerBundle(opt, WallClockSchedule(opt, spec.training_time_seconds))


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    token_training_loss=token_training_loss,
    batch_size=128,
    eval_batch_size=256,
)

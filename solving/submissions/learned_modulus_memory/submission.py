"""Card 4: Learned Modulus Memory — seen-N memory path + generic trunk + gate.

All memory values random-init and learned online. No hard-coded answer tables.

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
    assert_model_state,
)

N_MARK, X_MARK, T_MARK = 2, 3, 4
DIGIT_OFFSET = 7

D_MODEL = 160
NUM_HEADS = 4
MEM_SIDE = 45  # factorized address; slots = 45^2 = 2025
MEM_SLOTS = MEM_SIDE * MEM_SIDE
KEY_DIM = 64
VAL_DIM = 128
DEPTH_BUCKETS = 8
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


def _parse(ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
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
    return field, is_digit, t_val.clamp(min=1, max=64)


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


class ProductKeyMemory(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.key_a = nn.Parameter(torch.randn(MEM_SIDE, KEY_DIM // 2) * 0.02)
        self.key_b = nn.Parameter(torch.randn(MEM_SIDE, KEY_DIM // 2) * 0.02)
        self.values = nn.Parameter(torch.randn(MEM_SLOTS, VAL_DIM) * 0.02)
        self.query = nn.Linear(D_MODEL * 2 + 8, KEY_DIM)
        self.depth_emb = nn.Embedding(DEPTH_BUCKETS, 8)
        self.n_proj = nn.Linear(D_MODEL, D_MODEL)
        self.x_proj = nn.Linear(D_MODEL, D_MODEL)
        self.out = nn.Linear(VAL_DIM, D_MODEL)

    def forward(self, n_vec: Tensor, x_vec: Tensor, t_val: Tensor) -> tuple[Tensor, Tensor]:
        depth = (t_val.float().log2().floor().long()).clamp(0, DEPTH_BUCKETS - 1)
        q = self.query(
            torch.cat(
                [self.n_proj(n_vec), self.x_proj(x_vec), self.depth_emb(depth)], dim=-1
            )
        )
        qa, qb = q.chunk(2, dim=-1)
        # factorized top scores
        sa = qa @ self.key_a.T  # [B, half]
        sb = qb @ self.key_b.T
        # soft address over product grid via outer sum approximation: pick top
        wa = sa.softmax(-1)
        wb = sb.softmax(-1)
        # reconstruct soft slot weights [B, half, half] -> flatten
        w = torch.einsum("bi,bj->bij", wa, wb).reshape(n_vec.shape[0], -1)
        retrieved = w @ self.values
        conf = w.max(dim=-1).values
        return self.out(retrieved), conf


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.tok = nn.Embedding(spec.vocab_size, D_MODEL)
        self.pos = nn.Embedding(spec.max_seq_len, D_MODEL)
        self.blocks = nn.ModuleList(Block() for _ in range(4))
        self.memory = ProductKeyMemory()
        self.gate = nn.Sequential(nn.Linear(D_MODEL + 1, D_MODEL), nn.GELU(), nn.Linear(D_MODEL, 1))
        self.norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.mem_to_seq = nn.Linear(D_MODEL, D_MODEL)
        self.head.weight = self.tok.weight
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=INIT_SCALE * max(m.weight.shape[1], 1) ** -0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _pool(self, h: Tensor, mask: Tensor) -> Tensor:
        m = mask.to(h.dtype).unsqueeze(-1)
        return (h * m).sum(1) / m.sum(1).clamp_min(1.0)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        b, l = input_ids.shape
        field, is_digit, t_val = _parse(input_ids)
        positions = torch.arange(l, device=input_ids.device)
        h = self.tok(input_ids) + self.pos(positions)
        mask = None
        if attention_mask is not None:
            mask = attention_mask[:, None, None, :].to(torch.bool)
        for block in self.blocks:
            h = block(h, mask)

        n_mask = (field == 1) & is_digit
        x_mask = (field == 2) & is_digit
        n_vec = self._pool(h, n_mask)
        x_vec = self._pool(h, x_mask)
        mem_vec, conf = self.memory(n_vec, x_vec, t_val)
        gate = torch.sigmoid(self.gate(torch.cat([mem_vec, conf.unsqueeze(-1)], dim=-1)))
        boost = self.mem_to_seq(mem_vec).unsqueeze(1) * gate.unsqueeze(-1)
        logits = self.head(self.norm(h + boost))
        return logits, None


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
        lr=3e-3,
        betas=(0.9, 0.95),
        weight_decay=0.05,
        capturable=spec.device_type == "cuda",
    )
    return OptimizerBundle(opt, WallClockSchedule(opt, spec.training_time_seconds))


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    batch_size=256,
    eval_batch_size=512,
)

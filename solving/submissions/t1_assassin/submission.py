"""Card 2: T=1 Assassin — one-step specialist with weak multi-T fallback.

Learned pairwise x-digit interactions cross-attend to N; iterative refinement;
T-conditioned router between T=1 specialist and generic head. No explicit
digit-product arithmetic.

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

D_MODEL = 192
NUM_HEADS = 4
PAIR_DIM = 64
REFINE_ROUNDS = 6
MAX_DIGITS = 16
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


def _field_place(ids: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    is_n = (ids == N_MARK).cumsum(-1)
    is_x = (ids == X_MARK).cumsum(-1)
    is_t = (ids == T_MARK).cumsum(-1)
    field = (is_n + is_x + is_t).clamp(max=3)
    is_digit = ids >= DIGIT_OFFSET
    place_lsd = torch.zeros_like(ids)
    place_msd = torch.zeros_like(ids)
    for f in (1, 2, 3):
        m = (field == f) & is_digit
        rev = torch.flip(torch.flip(m.long(), dims=[-1]).cumsum(-1), dims=[-1])
        place_lsd = place_lsd + torch.where(m, rev - 1, torch.zeros_like(rev))
        fwd = m.long().cumsum(-1)
        place_msd = place_msd + torch.where(m, fwd - 1, torch.zeros_like(fwd))
    place_lsd = place_lsd.clamp(0, MAX_DIGITS - 1)
    place_msd = place_msd.clamp(0, MAX_DIGITS - 1)
    t_digits = torch.where(
        (field == 3) & is_digit, ids - DIGIT_OFFSET, torch.full_like(ids, -1)
    )
    t_val = torch.zeros(ids.shape[0], dtype=torch.long, device=ids.device)
    for pos in range(ids.shape[1]):
        d = t_digits[:, pos]
        keep = d >= 0
        t_val = torch.where(keep, t_val * 10 + d.clamp(min=0), t_val)
    return field, place_lsd, place_msd, is_digit, t_val.clamp(min=1, max=64)


class PairGrid(nn.Module):
    """Learned bilinear interactions between x-digit slots (no hardcoded mul)."""

    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Linear(D_MODEL, PAIR_DIM)
        self.right = nn.Linear(D_MODEL, PAIR_DIM)
        self.mix = nn.Linear(PAIR_DIM, D_MODEL)
        self.n_cross = nn.MultiheadAttention(D_MODEL, NUM_HEADS, batch_first=True)
        self.norm = RMSNorm(D_MODEL)

    def forward(self, x_tok: Tensor, n_tok: Tensor, x_mask: Tensor, n_mask: Tensor) -> Tensor:
        # x_tok: [B, Dx, D]
        b, dx, _ = x_tok.shape
        left = self.left(x_tok)  # [B, Dx, P]
        right = self.right(x_tok)
        # pairwise: [B, Dx, Dx, P] via outer product in pair dim
        grid = left.unsqueeze(2) * right.unsqueeze(1)
        flat = grid.mean(dim=2)  # compress partner axis
        h = self.mix(flat)
        h = self.norm(h + x_tok)
        # cross-attend to N digits
        key_pad = ~n_mask
        attn_out, _ = self.n_cross(h, n_tok, n_tok, key_padding_mask=key_pad)
        return self.norm(h + attn_out)


class RefineBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = RMSNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.out = nn.Linear(D_MODEL, D_MODEL)
        self.ffn = nn.Sequential(
            nn.Linear(D_MODEL, 4 * D_MODEL),
            nn.GELU(),
            nn.Linear(4 * D_MODEL, D_MODEL),
        )

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        b, l, _ = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(b, l, NUM_HEADS, -1).transpose(1, 2)
        k = k.view(b, l, NUM_HEADS, -1).transpose(1, 2)
        v = v.view(b, l, NUM_HEADS, -1).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(b, l, D_MODEL)
        x = x + self.out(y)
        return x + self.ffn(self.norm(x))


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.tok = nn.Embedding(spec.vocab_size, D_MODEL)
        self.pos = nn.Embedding(spec.max_seq_len, D_MODEL)
        self.place_lsd = nn.Embedding(MAX_DIGITS, D_MODEL)
        self.place_msd = nn.Embedding(MAX_DIGITS, D_MODEL)
        self.field_emb = nn.Embedding(4, D_MODEL)
        self.pair = PairGrid()
        self.refine = nn.ModuleList(RefineBlock() for _ in range(REFINE_ROUNDS))
        self.pool = nn.Linear(D_MODEL, D_MODEL)
        self.head_t1 = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.head_gen = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.t_router = nn.Sequential(
            nn.Linear(D_MODEL + 8, D_MODEL),
            nn.GELU(),
            nn.Linear(D_MODEL, 1),
        )
        self.t_embed = nn.Embedding(65, 8)
        self.norm = RMSNorm(D_MODEL)
        self.head_t1.weight = self.tok.weight
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=INIT_SCALE * max(m.weight.shape[1], 1) ** -0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _gather_field(
        self, h: Tensor, field: Tensor, is_digit: Tensor, which: int
    ) -> tuple[Tensor, Tensor]:
        b, l, d = h.shape
        mask = (field == which) & is_digit
        # pack up to MAX_DIGITS slots (LSD order via place already in emb)
        out = h.new_zeros(b, MAX_DIGITS, d)
        out_mask = torch.zeros(b, MAX_DIGITS, dtype=torch.bool, device=h.device)
        for i in range(b):
            idx = mask[i].nonzero(as_tuple=False).flatten()
            n = min(int(idx.numel()), MAX_DIGITS)
            if n:
                out[i, :n] = h[i, idx[:n]]
                out_mask[i, :n] = True
        return out, out_mask

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        b, l = input_ids.shape
        field, place_lsd, place_msd, is_digit, t_val = _field_place(input_ids)
        positions = torch.arange(l, device=input_ids.device)
        h = (
            self.tok(input_ids)
            + self.pos(positions)
            + self.field_emb(field)
            + self.place_lsd(place_lsd)
            + self.place_msd(place_msd)
        )
        mask = None
        if attention_mask is not None:
            mask = attention_mask[:, None, None, :].to(torch.bool)

        n_tok, n_mask = self._gather_field(h, field, is_digit, 1)
        x_tok, x_mask = self._gather_field(h, field, is_digit, 2)
        grid = self.pair(x_tok, n_tok, x_mask, n_mask)

        # Scatter grid summary back onto sequence via mean + broadcast add on x digits
        summary = self.pool(grid.mean(dim=1, keepdim=True))
        h = h + summary

        for block in self.refine:
            h = block(h, mask)

        h = self.norm(h)
        logits_t1 = self.head_t1(h)
        logits_gen = self.head_gen(h)
        t_feat = self.t_embed(t_val.clamp(0, 64))
        pooled = h.mean(dim=1)
        gate = torch.sigmoid(self.t_router(torch.cat([pooled, t_feat], dim=-1)))
        # Prefer specialist when T==1
        t1 = (t_val == 1).to(h.dtype).view(b, 1, 1)
        gate = gate.view(b, 1, 1)
        mix = t1 * gate + (1.0 - t1) * (1.0 - gate) * 0.5
        logits = mix * logits_t1 + (1.0 - mix) * logits_gen
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

"""Card 3: Gated Arithmetic Tape — Neural-GPU-style local recurrent workspace.

Tape rows for N/x/workspace vectors; tied local conv + gated mixer + control
tokens; random train microsteps, deeper eval. Final-label only.

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

D_MODEL = 128
TAPE_ROWS = 6  # N, x, 4 workspaces (packed along channel groups)
WORK_CHANNELS = D_MODEL
MAX_CELLS = 24
TRAIN_STEPS_MIN = 12
TRAIN_STEPS_MAX = 32
EVAL_STEPS = 64
NUM_HEADS = 4
INIT_SCALE = 0.35
WARMUP_FRACTION = 0.05
FINAL_LR_FRACTION = 0.01
NOISE_STD = 0.02
DAMP_INIT = 0.1


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
    place = torch.zeros_like(ids)
    for f in (1, 2):
        m = (field == f) & is_digit
        rev = torch.flip(torch.flip(m.long(), dims=[-1]).cumsum(-1), dims=[-1])
        place = place + torch.where(m, rev - 1, torch.zeros_like(rev))
    place = place.clamp(0, MAX_CELLS - 1)
    t_digits = torch.where(
        (field == 3) & is_digit, ids - DIGIT_OFFSET, torch.full_like(ids, -1)
    )
    t_val = torch.zeros(ids.shape[0], dtype=torch.long, device=ids.device)
    for pos in range(ids.shape[1]):
        d = t_digits[:, pos]
        keep = d >= 0
        t_val = torch.where(keep, t_val * 10 + d.clamp(min=0), t_val)
    return field, place, t_val.clamp(min=1, max=64)


class TapeTransition(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv3 = nn.Conv1d(WORK_CHANNELS, WORK_CHANNELS, 3, padding=1, groups=4)
        self.conv5 = nn.Conv1d(WORK_CHANNELS, WORK_CHANNELS, 5, padding=2, groups=4)
        self.gate = nn.Linear(WORK_CHANNELS, WORK_CHANNELS * 2)
        self.local_attn = nn.MultiheadAttention(WORK_CHANNELS, NUM_HEADS, batch_first=True)
        self.ctrl_proj = nn.Linear(WORK_CHANNELS * 2, WORK_CHANNELS)
        self.norm = RMSNorm(WORK_CHANNELS)
        self.alpha = nn.Parameter(torch.tensor(DAMP_INIT))

    def forward(self, tape: Tensor, ctrl: Tensor) -> Tensor:
        # tape: [B, C, Lcells]
        b, c, l = tape.shape
        local = self.conv3(tape) + self.conv5(tape)
        h = tape.transpose(1, 2)  # [B, L, C]
        attn, _ = self.local_attn(h, h, h)
        g, u = self.gate(self.norm(h + attn)).chunk(2, dim=-1)
        upd = torch.tanh(u) * torch.sigmoid(g)
        ctrl_b = self.ctrl_proj(ctrl).unsqueeze(1)
        upd = upd + 0.1 * ctrl_b
        damp = torch.sigmoid(self.alpha)
        new_h = h + damp * upd
        # mix conv residual in channel space
        new_h = new_h + 0.25 * local.transpose(1, 2)
        return new_h.transpose(1, 2)


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.tok = nn.Embedding(spec.vocab_size, D_MODEL)
        self.place = nn.Embedding(MAX_CELLS, D_MODEL)
        self.row_emb = nn.Embedding(TAPE_ROWS, D_MODEL)
        self.read_in = nn.Linear(D_MODEL, WORK_CHANNELS)
        self.transition = TapeTransition()
        self.ctrl_n = nn.Linear(D_MODEL, WORK_CHANNELS)
        self.ctrl_t = nn.Embedding(65, WORK_CHANNELS)
        self.decode = nn.Linear(WORK_CHANNELS, D_MODEL)
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.seq_enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=D_MODEL,
                nhead=NUM_HEADS,
                dim_feedforward=4 * D_MODEL,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            ),
            num_layers=2,
        )
        self.head.weight = self.tok.weight
        self._late_logits: list[Tensor] = []
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.normal_(m.weight, std=INIT_SCALE * max(m.weight.shape[1], 1) ** -0.5)
                if getattr(m, "bias", None) is not None and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _build_tape(self, ids: Tensor, field: Tensor, place: Tensor, h: Tensor) -> Tensor:
        b = ids.shape[0]
        tape = h.new_zeros(b, WORK_CHANNELS, MAX_CELLS)
        # row0 N digits, row1 x digits — average into cells by place
        for row, f in ((0, 1), (1, 2)):
            m = (field == f) & (ids >= DIGIT_OFFSET)
            for i in range(b):
                idx = m[i].nonzero(as_tuple=False).flatten()
                for j in idx.tolist():
                    p = int(place[i, j].item())
                    tape[i, :, p] = tape[i, :, p] + self.read_in(
                        h[i, j] + self.row_emb.weight[row]
                    )
        # workspace rows: small learned bias
        for row in range(2, TAPE_ROWS):
            tape = tape + 0.01 * self.read_in(self.row_emb.weight[row]).view(1, -1, 1)
        return tape

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, object | None]:
        b, l = input_ids.shape
        field, place, t_val = _parse(input_ids)
        h = self.tok(input_ids) + self.place(place.clamp(0, MAX_CELLS - 1))
        pad_mask = None
        if attention_mask is not None:
            pad_mask = ~attention_mask.to(torch.bool)
        h = self.seq_enc(h, src_key_padding_mask=pad_mask)

        tape = self._build_tape(input_ids, field, place, h)
        n_pool = h.new_zeros(b, D_MODEL)
        n_m = (field == 1) & (input_ids >= DIGIT_OFFSET)
        for i in range(b):
            idx = n_m[i].nonzero(as_tuple=False).flatten()
            if idx.numel():
                n_pool[i] = h[i, idx].mean(0)
        ctrl = torch.cat(
            [self.ctrl_n(n_pool), self.ctrl_t(t_val.clamp(0, 64))], dim=-1
        )

        if self.training:
            steps = int(torch.randint(TRAIN_STEPS_MIN, TRAIN_STEPS_MAX + 1, ()).item())
        else:
            steps = EVAL_STEPS

        self._late_logits = []
        for s in range(steps):
            if self.training and NOISE_STD > 0:
                tape = tape + NOISE_STD * torch.randn_like(tape)
            tape = self.transition(tape, ctrl)
            if s >= steps - 3:
                # decode tape cells onto sequence positions by place of x digits
                cell = tape.mean(dim=2)  # [B, C]
                boost = self.decode(cell).unsqueeze(1)
                logits_s = self.head(h + boost)
                self._late_logits.append(logits_s)

        cell = tape.mean(dim=2)
        logits = self.head(h + self.decode(cell).unsqueeze(1))
        aux = {"late": self._late_logits} if self.training else None
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
    if isinstance(aux, dict) and aux.get("late"):
        for late in aux["late"]:
            if late.shape[:2] != logits.shape[:2]:
                continue
            late_ce = F.cross_entropy(
                late.transpose(1, 2), labels, ignore_index=-100, reduction="none"
            )
            loss = loss + 0.25 * ((late_ce * valid).sum(dim=1) / counts).mean()
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
        lr=2e-3,
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

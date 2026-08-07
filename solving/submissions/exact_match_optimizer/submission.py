"""Card 1: Exact-Match Optimizer — Fable shell + sequence-exact losses + 2-pass SAM.

CHANGE vs fable_tcap_adamw: dual opposite-orientation heads, token_training_loss
(sequence CE + worst-digit softmax + margin + dual agreement), evaluator-owned
2-pass SAM perturbation + high-loss batch reuse. Architecture loop unchanged.

Source design: GPT-5 Pro (2026-08-07 five-card suite).
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from benchmark import (
    BatchReuseContext,
    BackwardPassContext,
    ModelSpec,
    OptimizerBundle,
    OptimizerSpec,
    Submission,
    TokenLossBatch,
    assert_model_state,
)

PAD, BOS, N_MARK, X_MARK, T_MARK, ANS_MARK, EOS = 0, 1, 2, 3, 4, 5, 6
DIGIT_OFFSET = 7

D_MODEL = 256
NUM_HEADS = 4
STEP_LAYERS = 2
MAX_LOOPS = 64
TRAIN_LOOP_CAP = 16
INIT_SCALE = 0.4
WARMUP_FRACTION = 0.05
FINAL_LR_FRACTION = 0.01

LAM_WORST = 0.5
LAM_MARGIN = 0.1
LAM_AGREE = 0.2
MARGIN_TARGET = 0.5
SAM_RHO = 0.05
REUSE_LOSS_THRESHOLD = 2.0
MAX_REUSES = 2


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
        (field == 3) & is_digit, ids - DIGIT_OFFSET, torch.full_like(ids, -1)
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
        self.head_fwd = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.head_rev = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.fuse_gate = nn.Linear(D_MODEL, 1)
        self.head_fwd.weight = self.token_embedding.weight
        self.auxiliary: object | None = None
        self._sam_backup: dict[int, Tensor] = {}
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.head_fwd:
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
    ) -> tuple[Tensor, object | None]:
        b, l = input_ids.shape
        field, place, t_val = _derived_features(input_ids)
        t_eff = t_val.clamp(max=TRAIN_LOOP_CAP) if self.training else t_val
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
        state = torch.zeros(b, l, self.config.vocab_size, dtype=base.dtype, device=base.device)
        x = base
        for t in range(max_t):
            x = self.step(base + self.state_proj(state), mask)
            logits = self.head_fwd(self.final_norm(x))
            new_state = self._quantize(logits)
            active = (t < t_eff).view(b, 1, 1).to(new_state.dtype)
            state = active * new_state + (1 - active) * state
            if self.training and t < detach_prefix:
                state = state.detach()

        h = self.final_norm(x)
        logits_fwd = self.head_fwd(h)
        # Opposite positional orientation: reverse sequence features into rev head.
        logits_rev = self.head_rev(torch.flip(h, dims=[1]))
        logits_rev = torch.flip(logits_rev, dims=[1])
        gate = torch.sigmoid(self.fuse_gate(h))
        logits = gate * logits_fwd + (1.0 - gate) * logits_rev
        self.auxiliary = {"logits_fwd": logits_fwd, "logits_rev": logits_rev}
        return logits, self.auxiliary


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    logits = batch.logits
    labels = batch.labels
    valid = batch.valid_mask.to(dtype=logits.dtype)
    # Per-token CE, keep [B, L]
    token_ce = F.cross_entropy(
        logits.transpose(1, 2),
        labels,
        ignore_index=-100,
        reduction="none",
    )
    counts = valid.sum(dim=1).clamp_min(1.0)
    seq_ce = (token_ce * valid).sum(dim=1) / counts
    primary = seq_ce.mean()

    # Softmax over digit losses (worst-digit emphasis)
    masked_ce = token_ce + (1.0 - valid) * (-1e4)
    worst = (masked_ce.softmax(dim=1) * token_ce * valid).sum(dim=1)
    worst = (worst / counts).mean()

    # Margin: correct logit vs strongest incorrect
    flat_logits = logits
    flat_labels = labels.clamp(min=0)
    gather = flat_logits.gather(-1, flat_labels.unsqueeze(-1)).squeeze(-1)
    # mask correct class
    neg = flat_logits.clone()
    neg.scatter_(-1, flat_labels.unsqueeze(-1), -1e9)
    hard_neg, _ = neg.max(dim=-1)
    margin = F.relu(MARGIN_TARGET - (gather - hard_neg))
    margin = ((margin * valid).sum(dim=1) / counts).mean()

    agree = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    aux = batch.auxiliary
    if isinstance(aux, dict) and "logits_fwd" in aux and "logits_rev" in aux:
        # Align aux to target length if needed (evaluator may slice logits)
        lf, lr = aux["logits_fwd"], aux["logits_rev"]
        if lf.shape[:2] == logits.shape[:2]:
            agree = F.kl_div(
                lf.log_softmax(-1),
                lr.softmax(-1).detach(),
                reduction="none",
            ).sum(-1)
            agree = ((agree * valid).sum(dim=1) / counts).mean()

    return primary + LAM_WORST * worst + LAM_MARGIN * margin + LAM_AGREE * agree


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


class RestoringAdamW(torch.optim.AdamW):
    """AdamW that undoes SAM ascent before applying the update."""

    def __init__(self, params, model: Model, **kwargs) -> None:
        super().__init__(params, **kwargs)
        self._model = model

    @torch.no_grad()
    def step(self, closure=None):
        backup = self._model._sam_backup
        if backup:
            for p in self._model.parameters():
                if id(p) in backup:
                    p.copy_(backup[id(p)])
            backup.clear()
        return super().step(closure)


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    assert isinstance(model, Model)
    opt = RestoringAdamW(
        model.parameters(),
        model,
        lr=3e-3,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        capturable=spec.device_type == "cuda",
    )

    def between_backward_passes(ctx: BackwardPassContext) -> None:
        # First-order SAM ascent after pass 1; restore happens in optimizer.step.
        if ctx.pass_index != 1:
            return
        grads = []
        params = []
        for p in model.parameters():
            if p.grad is None:
                continue
            params.append(p)
            grads.append(p.grad)
        if not grads:
            return
        flat = torch.cat([g.reshape(-1) for g in grads])
        norm = flat.norm().clamp_min(1e-12)
        scale = SAM_RHO / norm
        model._sam_backup.clear()
        for p, g in zip(params, grads):
            model._sam_backup[id(p)] = p.detach().clone()
            p.add_(g, alpha=float(scale))

    def should_reuse_batch(ctx: BatchReuseContext) -> bool:
        return ctx.current_batch_uses < MAX_REUSES and ctx.loss > REUSE_LOSS_THRESHOLD

    return OptimizerBundle(
        opt,
        WallClockSchedule(opt, spec.training_time_seconds),
        backward_passes_per_step=2,
        between_backward_passes=between_backward_passes,
        should_reuse_batch=should_reuse_batch,
    )


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    token_training_loss=token_training_loss,
    batch_size=512,
    eval_batch_size=1024,
)

"""One Layer Deeper — Hard submission (Fable architecture, Muon + AdamW hybrid).

Same architecture as fable_hard_h1 (design thesis unchanged — see that
file's docstring). Optimizer changed from the original flat-lr=3e-4
WarmupSchedule (loss stuck ~2.1-2.2 across both 60s and 600s local runs —
see fable_hard_h1's NOTE.md) to the standard Muon+AdamW hybrid split:

  - Muon (Jordan et al., orthogonalized momentum via Newton-Schulz
    iteration) for every 2D hidden weight matrix inside the step block
    (qkv/out/up/down) and state_proj — the matrices that do the actual
    transformation work each loop.
  - AdamW for everything else: embeddings (including the tied head),
    RMSNorm scales, Linear biases — 1D/lookup parameters Muon isn't
    designed for.

Relevant here specifically because the recurrence is applied many times
(read from T, up to MAX_LOOPS) — Muon's orthogonalized updates keep each
step's weight matrix closer to well-conditioned, which bears directly on
the error-compounding argument in learnings/concepts/17-recurrence-
generalisation.md (state = digit slots, held at the TAIL positions,
requantised every loop — the quantization step already bounds per-step
error; Muon is the complementary lever on the *weights* doing the
compounding, not the state).

validate_optimizer (benchmark/validation.py) explicitly supports a
multi-optimizer wrapper via an `.optimizers` list attribute — that's what
CombinedOptimizer below is for, not a hack around the contract.
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

# ---- fixed token layout (data/squaring_mod.py TOKEN_IDS) -------------------
PAD, BOS, N_MARK, X_MARK, T_MARK, ANS_MARK, EOS = 0, 1, 2, 3, 4, 5, 6
DIGIT_OFFSET = 7

D_MODEL = 256
NUM_HEADS = 4
STEP_LAYERS = 2          # depth of the tied step block
MAX_LOOPS = 64           # hard cap on outer iterations (covers T<=64)
ENTROPY_WEIGHT = 0.01
INIT_SCALE = 0.4         # shrunken init: bias toward the generalizing basin


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
    """One learned recurrence step: STEP_LAYERS tied transformer layers."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(Block() for _ in range(STEP_LAYERS))

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


def _derived_features(input_ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Field id, place-within-field (from the field's end), and parsed T.

    Pure functions of input_ids under the fixed serialization
    N <digits> X <digits> T <digits> — derived features, not a tokenizer.
    """
    ids = input_ids
    is_n = (ids == N_MARK).cumsum(-1)
    is_x = (ids == X_MARK).cumsum(-1)
    is_t = (ids == T_MARK).cumsum(-1)
    field = is_n + is_x + is_t          # 1 in N-field, 2 in X, 3 in T-tail
    field = field.clamp(max=3)

    is_digit = ids >= DIGIT_OFFSET
    # place value = count of digits AFTER this one within the same field
    place = torch.zeros_like(ids)
    for f in (1, 2, 3):
        m = (field == f) & is_digit
        # reversed cumulative count within field f
        rev = torch.flip(torch.flip(m.long(), dims=[-1]).cumsum(-1), dims=[-1])
        place = place + torch.where(m, rev - 1, torch.zeros_like(rev))
    place = place.clamp(max=15)

    # parse the integer value of the T field per row
    t_digits = torch.where((field == 3) & is_digit, ids - DIGIT_OFFSET,
                           torch.full_like(ids, -1))
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

        max_t = int(t_val.max().item())
        detach_prefix = (
            int(torch.randint(0, max_t, ()).item()) if (self.training and max_t > 1) else 0
        )

        state = torch.zeros(b, l, self.config.vocab_size,
                            dtype=base.dtype, device=base.device)
        ent_terms = []
        x = base
        for t in range(max_t):
            x = self.step(base + self.state_proj(state), mask)  # operand re-injection
            logits = self.head(self.final_norm(x))
            if self.training:
                p = logits.float().softmax(-1)
                ent_terms.append(-(p * (p + 1e-9).log()).sum(-1).mean())
            new_state = self._quantize(logits)
            active = (t < t_val).view(b, 1, 1).to(new_state.dtype)
            state = active * new_state + (1 - active) * state  # per-row stop at its own T
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


# ---- Muon (Jordan et al.) — standard reference implementation --------------


def _zeropower_via_newtonschulz5(g: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    assert g.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    x = x / (x.norm() + eps)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    for _ in range(steps):
        aa = x @ x.T
        bb = b * aa + c * aa @ aa
        x = a * x + bb @ x
    if transposed:
        x = x.T
    return x


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> None:
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf
                g = _zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                if group["weight_decay"] != 0:
                    p.data.mul_(1 - lr * group["weight_decay"])
                scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                p.data.add_(g, alpha=-lr * scale)


class CombinedOptimizer:
    """Wraps Muon + AdamW behind one optimizer-shaped object.

    validate_optimizer (benchmark/validation.py) reads `.optimizers` off the
    bundle's optimizer if present and validates each child's state
    separately — this shape is explicitly supported by the harness.
    """

    def __init__(self, optimizers: list) -> None:
        self.optimizers = optimizers
        self.param_groups = [g for opt in optimizers for g in opt.param_groups]

    def step(self, closure=None) -> None:
        for opt in self.optimizers:
            opt.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict: dict) -> None:
        for opt, sd in zip(self.optimizers, state_dict["optimizers"]):
            opt.load_state_dict(sd)


WARMUP_FRACTION = 0.05
FINAL_LR_FRACTION = 0.01


class WallClockSchedule:
    """Validated wall-clock warmup+cosine, learnings/concepts/15-lr-schedules-wallclock.md.

    Scales every param group by the same fractional factor while preserving
    each group's own base lr — correct for a multi-optimizer combo where
    Muon and AdamW run at different absolute learning rates.
    """

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
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        is_hidden_matrix = (
            p.ndim == 2
            and (
                ".qkv.weight" in name
                or ".out.weight" in name
                or ".up.weight" in name
                or ".down.weight" in name
                or name == "state_proj.weight"
            )
        )
        (muon_params if is_hidden_matrix else adamw_params).append(p)

    muon = Muon(muon_params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5)
    adamw = torch.optim.AdamW(
        adamw_params,
        lr=3e-3,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        capturable=spec.device_type == "cuda",
    )
    combined = CombinedOptimizer([muon, adamw])
    return OptimizerBundle(combined, WallClockSchedule(combined, spec.training_time_seconds))


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    training_loss=training_loss,
    batch_size=256,
)

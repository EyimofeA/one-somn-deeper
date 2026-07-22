"""One Layer Deeper — Hard submission.

Design thesis: exact-match at held-out depth is a fixed-point problem.
Structure = (1) ONE weight-tied step block looped a number of times read
from the T field of the input (adaptive computation, README Rule 3);
(2) straight-through digit quantization of the answer-position state
between loops, so per-step error resets to zero instead of compounding;
(3) operand re-injection every loop (prompt embeddings re-added, so task
identity never has to survive the recurrence); (4) detached-prefix
training (train-only) so the block learns to be correct on its OWN
quantized outputs, the distribution it lives on at eval T; (5) entropy
auxiliary consumed by a custom training_loss, keeping pre-quantization
margins wide.  The step function itself is fully learned — nothing here
assumes squaring, only "iterate a step T times", which is the problem
statement.  Derived features (field id, place-within-field) are computed
from input_ids at runtime — no tokenizer changes.
"""
from __future__ import annotations

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


class WarmupSchedule:
    def __init__(self, optimizer, warmup_steps: int = 300, lr: float = 3e-4) -> None:
        self.optimizer, self.warmup, self.lr, self.n = optimizer, warmup_steps, lr, 0

    def step(self) -> None:
        self.n += 1
        scale = min(1.0, self.n / self.warmup)
        for g in self.optimizer.param_groups:
            g["lr"] = self.lr * scale


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    opt = torch.optim.AdamW(
        model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.5,
        capturable=spec.device_type == "cuda",
    )
    return OptimizerBundle(opt, WarmupSchedule(opt))


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    training_loss=training_loss,
    batch_size=256,
)

"""New candidate: T-conditioned tied digit-recurrent Transformer.

The same transition is reused, giving the model an explicit path from a learned
T=1 map to repeated application. A depthwise local mixer supplies a short path
for decimal carry/reduction information; attention supplies global N/x access.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from benchmark import ModelSpec, OptimizerBundle, OptimizerSpec, Submission, TokenLossBatch, assert_model_state

D = 160
H = 5
LOOPS = 6


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size, self.max_seq_len = vocab_size, max_seq_len


class Transition(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.n1 = nn.RMSNorm(D)
        self.attn = nn.MultiheadAttention(D, H, batch_first=True)
        self.n2 = nn.RMSNorm(D)
        self.local = nn.Conv1d(D, D, 3, padding=1, groups=D)
        self.mix = nn.Linear(2 * D, 2 * D)
        self.gate = nn.Linear(D, D)

    def forward(self, x: Tensor, pad: Tensor | None, control: Tensor) -> Tensor:
        q = self.n1(x + control[:, None])
        a, _ = self.attn(q, q, q, key_padding_mask=pad, need_weights=False)
        h = self.n2(x + a)
        local = self.local(h.transpose(1, 2)).transpose(1, 2)
        update, candidate = self.mix(torch.cat((h, local), -1)).chunk(2, -1)
        g = torch.sigmoid(self.gate(control))[:, None]
        return x + g * torch.sigmoid(update) * F.silu(candidate)


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.tok = nn.Embedding(spec.vocab_size, D)
        self.pos = nn.Embedding(spec.max_seq_len, D)
        self.phase = nn.Embedding(LOOPS, D)
        self.transition = Transition()
        self.norm = nn.RMSNorm(D)
        self.head = nn.Linear(D, spec.vocab_size, bias=False)
        self.head.weight = self.tok.weight
        self._stages: list[Tensor] = []

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None):
        length = input_ids.shape[1]
        p = torch.arange(length, device=input_ids.device)
        x = self.tok(input_ids) + self.pos(p)
        pad = None
        if attention_mask is not None and attention_mask.ndim == 2:
            pad = ~attention_mask.bool()
        pooled = x.mean(1)
        self._stages = []
        for i in range(LOOPS):
            x = self.transition(x, pad, pooled + self.phase.weight[i])
            if i >= LOOPS - 3:
                self._stages.append(self.head(self.norm(x)))
        return self._stages[-1], {"stages": self._stages[:-1]}


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    def ce(logits: Tensor) -> Tensor:
        raw = F.cross_entropy(logits.transpose(1, 2), batch.labels, ignore_index=-100, reduction="none")
        valid = batch.valid_mask.to(raw.dtype)
        return ((raw * valid).sum(1) / valid.sum(1).clamp_min(1)).mean()
    loss = ce(batch.logits)
    if isinstance(batch.auxiliary, dict):
        for logits in batch.auxiliary.get("stages", []):
            loss = loss + 0.2 * ce(logits)
    return loss


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    return OptimizerBundle(torch.optim.AdamW(model.parameters(), lr=1.5e-3, betas=(0.9, 0.95), weight_decay=0.05, capturable=spec.device_type == "cuda"))


SUBMISSION = Submission(build_model=build_model, build_optimizer=build_optimizer, token_training_loss=token_training_loss, batch_size=64, eval_batch_size=128)

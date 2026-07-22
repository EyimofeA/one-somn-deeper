"""Bidirectional GRU baseline with tied LM head."""

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


D_MODEL = 128
RNN_HIDDEN = 64
NUM_RNN_LAYERS = 1


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


class Model(nn.Module):
    num_loops = 1

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.token_embedding = nn.Embedding(spec.vocab_size, D_MODEL)
        self.position_embedding = nn.Embedding(spec.max_seq_len, D_MODEL)
        self.rnn = nn.GRU(
            D_MODEL,
            RNN_HIDDEN,
            num_layers=NUM_RNN_LAYERS,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(2 * RNN_HIDDEN, D_MODEL)
        self.final_norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).to(torch.int64).clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.rnn(packed)
            x, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
            if x.shape[1] < input_ids.shape[1]:
                pad = torch.zeros(
                    x.shape[0],
                    input_ids.shape[1] - x.shape[1],
                    x.shape[2],
                    device=x.device,
                    dtype=x.dtype,
                )
                x = torch.cat([x, pad], dim=1)
        else:
            x, _ = self.rnn(x)

        x = self.proj(x)
        x = self.final_norm(x)

        if attention_mask is not None:
            pad_mask = attention_mask.unsqueeze(-1).to(dtype=x.dtype)
            x = x * pad_mask

        return self.head(x), None


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    return OptimizerBundle(
        torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            capturable=spec.device_type == "cuda",
        )
    )


SUBMISSION = Submission(build_model=build_model, build_optimizer=build_optimizer)

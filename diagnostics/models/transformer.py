"""Baseline 1: standard bidirectional Transformer, output-slot cross-entropy.

A control, not the preferred architecture (see recurrent_workspace.py). Plain
learned token + absolute position embeddings, bidirectional (non-causal)
self-attention, and a shared vocab head applied at every position — loss and
exact-match are computed only where labels != IGNORE_INDEX (the appended
<OUT> slots).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from data.tokens import VOCAB_SIZE


class StandardTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        max_seq_len: int = 32,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.0,
        digit_vocab: int = 10,
    ) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, digit_vocab)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        batch, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device)
        h = self.token_embed(input_ids) + self.pos_embed(positions)[None, :, :]
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask  # True = ignore, per nn.Transformer convention
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        h = self.norm(h)
        return self.head(h)  # (batch, seq_len, digit_vocab) — a 10-way logit per position

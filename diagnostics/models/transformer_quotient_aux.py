"""Standard Transformer with an optional four-digit auxiliary readout."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from data.tokens import NUM_MOD_DIGITS, VOCAB_SIZE


class QuotientAuxTransformer(nn.Module):
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
        self.aux_head = nn.Linear(d_model, digit_vocab)

    def hidden(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        _, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device)
        h = self.token_embed(input_ids) + self.pos_embed(positions)[None, :, :]
        key_padding_mask = ~attention_mask if attention_mask is not None else None
        return self.norm(self.encoder(h, src_key_padding_mask=key_padding_mask))

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        return self.head(self.hidden(input_ids, attention_mask))

    def forward_with_aux(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        h = self.hidden(input_ids, attention_mask)
        return self.head(h), self.aux_head(h[:, -NUM_MOD_DIGITS:])

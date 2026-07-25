"""Same StandardTransformer backbone (embeddings + TransformerEncoder + norm +
digit head, identical hyperparameters), with two OPTIONAL small auxiliary
regression heads bolted onto the same final hidden states used for the main
digit prediction:

  - carry head:    Linear(d_model, 2) -> (carry_in, carry_out), normalized
  - diagonal head: Linear(d_model, 1) -> diagonal sum, normalized

Heads are only constructed if requested, so the "baseline" condition (no aux)
has exactly the same parameter count as models/transformer.py's
StandardTransformer -- and conditions with aux heads add only ~2-3 x d_model
extra parameters, negligible next to the ~800k-parameter backbone. This is a
deliberately separate module from models/transformer.py so the ablation can't
accidentally change behavior for any of the other (non-ablation) training
runs that import StandardTransformer directly.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from data.tokens import VOCAB_SIZE


class StandardTransformerAux(nn.Module):
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
        use_carry_aux: bool = False,
        use_diag_aux: bool = False,
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

        self.use_carry_aux = use_carry_aux
        self.use_diag_aux = use_diag_aux
        self.carry_head = nn.Linear(d_model, 2) if use_carry_aux else None
        self.diag_head = nn.Linear(d_model, 1) if use_diag_aux else None

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None):
        batch, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device)
        h = self.token_embed(input_ids) + self.pos_embed(positions)[None, :, :]
        key_padding_mask = ~attention_mask if attention_mask is not None else None
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        h = self.norm(h)
        logits = self.head(h)
        carry_pred = self.carry_head(h) if self.carry_head is not None else None
        diag_pred = self.diag_head(h) if self.diag_head is not None else None
        return logits, carry_pred, diag_pred

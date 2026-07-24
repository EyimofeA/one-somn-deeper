"""Baseline 2 (main architecture): arithmetic recurrent workspace model.

Separates an immutable input context `c` from a mutable workspace `w`.
A single weight-tied transition block F_theta is applied `num_loops` times:
workspace tokens self-attend, cross-attend to the frozen context, and get a
learned per-iteration embedding added before each application. The answer is
decoded only from a fixed subset of workspace tokens ("output registers"),
never from the context directly.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from data.tokens import VOCAB_SIZE


class ContextEncoder(nn.Module):
    """Encoder(N, x, task) -> c. Two bidirectional self-attention layers over
    the raw input tokens (OUT-slot tokens included; the model can learn to
    ignore them, they carry no target-relevant information)."""

    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int, n_heads: int, d_ff: int, n_layers: int = 2):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        batch, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device)
        h = self.token_embed(input_ids) + self.pos_embed(positions)[None, :, :]
        key_padding_mask = ~attention_mask if attention_mask is not None else None
        c = self.encoder(h, src_key_padding_mask=key_padding_mask)
        return c, key_padding_mask


class WorkspaceTransition(nn.Module):
    """F_theta: one weight-tied step. Workspace self-attends, then
    cross-attends to the immutable context, then an FFN, each with a
    pre-norm residual — a standard decoder-style block but reused in a loop
    rather than stacked."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, w: Tensor, c: Tensor, context_key_padding_mask: Tensor | None) -> Tensor:
        h = self.norm1(w)
        attn_out, _ = self.self_attn(h, h, h, need_weights=False)
        w = w + attn_out

        h = self.norm2(w)
        attn_out, _ = self.cross_attn(h, c, c, key_padding_mask=context_key_padding_mask, need_weights=False)
        w = w + attn_out

        h = self.norm3(w)
        w = w + self.ffn(h)
        return w


class RecurrentWorkspaceModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        max_seq_len: int = 32,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 512,
        context_layers: int = 2,
        workspace_size: int = 8,
        num_output_slots: int = 8,
        num_loops: int = 8,
        digit_vocab: int = 10,
    ) -> None:
        super().__init__()
        if num_output_slots > workspace_size:
            raise ValueError("num_output_slots must not exceed workspace_size")
        self.workspace_size = workspace_size
        self.num_output_slots = num_output_slots
        self.num_loops = num_loops

        self.context_encoder = ContextEncoder(vocab_size, max_seq_len, d_model, n_heads, d_ff, context_layers)
        self.workspace_init = nn.Parameter(torch.randn(workspace_size, d_model) * 0.02)
        self.iter_embed = nn.Embedding(num_loops, d_model)
        self.transition = WorkspaceTransition(d_model, n_heads, d_ff)
        self.register_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, digit_vocab)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None, override_loops: int | None = None) -> Tensor:
        batch = input_ids.shape[0]
        c, ctx_mask = self.context_encoder(input_ids, attention_mask)
        w = self.workspace_init[None, :, :].expand(batch, -1, -1).contiguous()
        loops = self.num_loops if override_loops is None else override_loops
        for t in range(loops):
            w = w + self.iter_embed(torch.tensor(t, device=input_ids.device))[None, None, :]
            w = self.transition(w, c, ctx_mask)
        registers = self.register_norm(w[:, : self.num_output_slots, :])
        return self.output_head(registers)  # (batch, num_output_slots, digit_vocab)

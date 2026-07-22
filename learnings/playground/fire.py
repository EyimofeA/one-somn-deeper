"""FIRE — Functional Interpolation for Relative Positional Encoding (Li et al., 2023).

Paper: https://arxiv.org/abs/2310.04418  (Eq. 4)

  b(i, j) = f_θ( ψ(i − j) / ψ(max(L, i)) )
  ψ(x) = log(c · x + 1)

Added as a bias to attention logits (per head). Progressive interpolation keeps the
MLP input in [0, 1] at any sequence length — that is the length-generalization trick.

This file is for learning only. Not wired into competition submissions.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FIRE(nn.Module):
    """Per-head additive attention bias from relative positions."""

    def __init__(self, num_heads: int, mlp_hidden: int = 32):
        super().__init__()
        self.num_heads = num_heads
        # log(c): softplus keeps c > 0
        self.log_c = nn.Parameter(torch.zeros(1))
        # threshold L > 0 (softplus)
        self.raw_L = nn.Parameter(torch.zeros(1))
        # f_θ: R → R^H  (tiny MLP, paper uses hidden 32)
        self.mlp = nn.Sequential(
            nn.Linear(1, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, num_heads),
        )

    def _psi(self, x: torch.Tensor) -> torch.Tensor:
        c = F.softplus(self.log_c) + 1e-6
        return torch.log(c * x + 1.0)

    def bias(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return (1, H, T, T) causal relative bias for positions 0..T-1."""
        # positions are 1-indexed in the paper (i, j ≥ 1); use 1..T
        pos = torch.arange(1, seq_len + 1, device=device, dtype=dtype)
        i = pos.view(-1, 1)  # query
        j = pos.view(1, -1)  # key
        dist = (i - j).clamp(min=0)  # causal: i ≥ j
        L = F.softplus(self.raw_L) + 1.0
        normalizer = torch.maximum(L, i)
        x = self._psi(dist) / self._psi(normalizer)
        # (T, T, 1) → MLP → (T, T, H) → (H, T, T)
        out = self.mlp(x.unsqueeze(-1))
        return out.permute(2, 0, 1).unsqueeze(0)

    def forward(self, attn_logits: torch.Tensor) -> torch.Tensor:
        """attn_logits: (B, H, T, T) → same + FIRE bias."""
        b, h, t, _ = attn_logits.shape
        assert h == self.num_heads
        return attn_logits + self.bias(t, attn_logits.device, attn_logits.dtype)


class TinyCausalSelfAttentionFIRE(nn.Module):
    """Minimal causal MHA + FIRE — playground only."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.fire = FIRE(num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scale = 1.0 / math.sqrt(self.head_dim)
        logits = torch.matmul(q, k.transpose(-2, -1)) * scale
        # causal mask
        causal = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        logits = logits.masked_fill(causal, float("-inf"))
        logits = self.fire(logits)
        attn = torch.softmax(logits, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, t, d)
        return self.proj(out)


if __name__ == "__main__":
    torch.manual_seed(0)
    m = TinyCausalSelfAttentionFIRE(d_model=32, num_heads=4)
    x = torch.randn(2, 16, 32)
    y = m(x)
    print("FIRE attn out", tuple(y.shape), "ok")
    # show that longer seq still produces finite biases in [domain]
    fire = FIRE(4)
    b16 = fire.bias(16, x.device, x.dtype)
    b64 = fire.bias(64, x.device, x.dtype)
    print("bias16", tuple(b16.shape), "bias64", tuple(b64.shape))

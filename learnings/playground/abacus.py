"""Official Abacus Embeddings (McLeish et al., 2024) — learning copy.

Source: https://github.com/mcleish7/arithmetic/blob/main/abacus.py
Paper: https://arxiv.org/abs/2405.17399

Integers must be reversed (LSD first) for place-value alignment to match the paper.
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn


class Abacus(nn.Module):
    """Learned place embeddings reused for each digit span.

    digit_tokens: token ids for '0'..'9'
    max_k: random offset U[0, max_k] added to place indices during training
    """

    def __init__(
        self,
        digit_tokens: list[int],
        embedding_dim: int,
        max_seq_length: int = 1024,
        max_k: int = 99,
    ):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_length, embedding_dim)
        self.register_buffer("digits", torch.tensor(digit_tokens), persistent=False)
        self.max_k = max_k

    def helper(self, mask: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Binary digit mask → 1-based place index within each contiguous digit span."""
        mask_shape = mask.shape
        shifted_mask = torch.cat(
            [torch.zeros((mask_shape[0], 1), device=device, dtype=mask.dtype), mask[:, :-1]],
            dim=1,
        )
        starts = (shifted_mask != mask) & mask
        segment_ids = torch.cumsum(starts, dim=1)
        index = torch.arange(mask.size(1), device=device).repeat(mask.size(0), 1)
        reset_index = torch.zeros_like(mask).long()
        second_term = index * starts.long()
        reset_index = reset_index.scatter_add(1, segment_ids, second_term)
        positions = index - reset_index.gather(1, segment_ids) + 1
        return positions * mask

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = torch.isin(input_ids, self.digits)
        output = self.helper(mask, input_ids.device)
        if self.training:
            k = random.randint(0, self.max_k)
            output = output.clone()
            output[output > 0] += k
        return self.embedding(output)

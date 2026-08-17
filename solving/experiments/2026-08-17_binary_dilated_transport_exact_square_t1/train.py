"""Exact-square reducer with tied weights and cyclic horizontal dilation."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


BASE_PATH = (
    Path(__file__).resolve().parents[1]
    / "2026-08-16_binary_workstate_matched"
    / "train.py"
)
_spec = importlib.util.spec_from_file_location("binary_matched_base", BASE_PATH)
assert _spec is not None and _spec.loader is not None
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)


class SharedDilatedConv(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        reference = nn.Conv2d(channels, channels, 3, padding=1)
        self.weight = reference.weight
        self.bias = reference.bias

    def forward(self, state: torch.Tensor, dilation: int) -> torch.Tensor:
        return F.conv2d(
            state,
            self.weight,
            self.bias,
            padding=(1, dilation),
            dilation=(1, dilation),
        )


class DilatedCell(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.update = SharedDilatedConv(channels)
        self.reset = SharedDilatedConv(channels)
        self.candidate = SharedDilatedConv(channels)

    def forward(
        self,
        state: torch.Tensor,
        dropout_mask: torch.Tensor | None,
        dilation: int,
    ) -> torch.Tensor:
        update = torch.sigmoid(self.update(state, dilation))
        reset = torch.sigmoid(self.reset(state, dilation))
        candidate = torch.tanh(self.candidate(reset * state, dilation))
        if dropout_mask is not None:
            candidate = candidate * dropout_mask
        return (1.0 - update) * state + update * candidate


class DilatedBinaryWorkState(base.BinaryWorkState):
    def __init__(self, channels: int, updates: int, dropout: float) -> None:
        super().__init__(channels, updates, dropout)
        self.cell = DilatedCell(channels)

    def forward(
        self, source_bits: torch.Tensor, modulus_bits: torch.Tensor
    ) -> torch.Tensor:
        batch = source_bits.shape[0]
        source = self.bit_embedding(source_bits).transpose(1, 2)
        modulus = self.bit_embedding(modulus_bits).transpose(1, 2)
        source = source + self.source_role[None, :, None]
        modulus = modulus + self.modulus_role[None, :, None]
        state = source.new_zeros(
            batch, self.channels, base.LANES, base.WORKSPACE_BITS
        )
        state[:, :, 2] = source + self.work_role[None, :, None]
        dropout_mask = None
        if self.training and self.dropout:
            keep = 1.0 - self.dropout
            dropout_mask = torch.empty(
                batch, self.channels, 1, 1, device=state.device
            ).bernoulli_(keep) / keep
        dilations = (1, 2, 4, 8)
        for step in range(self.updates):
            visible = state.clone()
            visible[:, :, 0] = source
            visible[:, :, 1] = modulus
            visible[:, :, :, 0] += self.boundaries[0][None, :, None]
            visible[:, :, :, -1] += self.boundaries[1][None, :, None]
            state = self.cell(visible, dropout_mask, dilations[step % 4])
        return self.readout(state[:, :, 2, : base.OPERAND_BITS]).squeeze(1)


base.BinaryWorkState = DilatedBinaryWorkState


if __name__ == "__main__":
    base.main()

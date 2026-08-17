"""Local exact-square reducer with messages routed to a scratch lane."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch import nn


BASE_PATH = (
    Path(__file__).resolve().parents[1]
    / "2026-08-16_binary_workstate_matched"
    / "train.py"
)
_spec = importlib.util.spec_from_file_location("binary_matched_base", BASE_PATH)
assert _spec is not None and _spec.loader is not None
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

LANES = base.LANES
WORKSPACE_BITS = base.WORKSPACE_BITS
OPERAND_BITS = base.OPERAND_BITS


def shifted_pair(state: torch.Tensor, distance: int) -> torch.Tensor:
    left = torch.zeros_like(state)
    right = torch.zeros_like(state)
    left[..., distance:] = state[..., :-distance]
    right[..., :-distance] = state[..., distance:]
    return left + right


class LocalCell(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.update = nn.Conv2d(channels, channels, 3, padding=1)
        self.reset = nn.Conv2d(channels, channels, 3, padding=1)
        self.candidate = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(
        self, state: torch.Tensor, dropout_mask: torch.Tensor | None
    ) -> torch.Tensor:
        update = torch.sigmoid(self.update(state))
        reset = torch.sigmoid(self.reset(state))
        candidate = torch.tanh(self.candidate(reset * state))
        if dropout_mask is not None:
            candidate = candidate * dropout_mask
        return (1.0 - update) * state + update * candidate


class ScratchMessageBinaryWorkState(base.BinaryWorkState):
    def __init__(self, channels: int, updates: int, dropout: float) -> None:
        super().__init__(channels, updates, dropout)
        self.cell = LocalCell(channels)
        self.fast_message_scales = nn.Parameter(torch.zeros(3))

    def forward(
        self, source_bits: torch.Tensor, modulus_bits: torch.Tensor
    ) -> torch.Tensor:
        batch = source_bits.shape[0]
        source = self.bit_embedding(source_bits).transpose(1, 2)
        modulus = self.bit_embedding(modulus_bits).transpose(1, 2)
        source = source + self.source_role[None, :, None]
        modulus = modulus + self.modulus_role[None, :, None]
        state = source.new_zeros(batch, self.channels, LANES, WORKSPACE_BITS)
        state[:, :, 2] = source + self.work_role[None, :, None]
        dropout_mask = None
        if self.training and self.dropout:
            keep = 1.0 - self.dropout
            dropout_mask = torch.empty(
                batch, self.channels, 1, 1, device=state.device
            ).bernoulli_(keep) / keep

        distances = (2, 4, 8)
        message_index = 0
        for step in range(self.updates):
            visible = state.clone()
            visible[:, :, 0] = source
            visible[:, :, 1] = modulus
            visible[:, :, :, 0] += self.boundaries[0][None, :, None]
            visible[:, :, :, -1] += self.boundaries[1][None, :, None]
            if step % 4 == 3:
                gate_index = message_index % len(distances)
                visible[:, :, 3] = visible[:, :, 3] + self.fast_message_scales[
                    gate_index
                ] * shifted_pair(visible[:, :, 2], distances[gate_index])
                message_index += 1
            state = self.cell(visible, dropout_mask)
        return self.readout(state[:, :, 2, :OPERAND_BITS]).squeeze(1)


base.BinaryWorkState = ScratchMessageBinaryWorkState


if __name__ == "__main__":
    base.main()

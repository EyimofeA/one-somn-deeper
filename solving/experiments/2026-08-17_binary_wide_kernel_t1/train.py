"""Wide-kernel variant of the deterministic binary T=1 harness."""
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


class WideKernelCell(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        kernel = (3, 7)
        padding = (1, 3)
        self.update = nn.Conv2d(channels, channels, kernel, padding=padding)
        self.reset = nn.Conv2d(channels, channels, kernel, padding=padding)
        self.candidate = nn.Conv2d(channels, channels, kernel, padding=padding)

    def forward(
        self, state: torch.Tensor, dropout_mask: torch.Tensor | None
    ) -> torch.Tensor:
        update = torch.sigmoid(self.update(state))
        reset = torch.sigmoid(self.reset(state))
        candidate = torch.tanh(self.candidate(reset * state))
        if dropout_mask is not None:
            candidate = candidate * dropout_mask
        return (1.0 - update) * state + update * candidate


class WideKernelBinaryWorkState(base.BinaryWorkState):
    def __init__(self, channels: int, updates: int, dropout: float) -> None:
        super().__init__(channels, updates, dropout)
        self.cell = WideKernelCell(channels)


base.BinaryWorkState = WideKernelBinaryWorkState

if __name__ == "__main__":
    base.main()

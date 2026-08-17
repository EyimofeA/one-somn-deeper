"""Final-label-only streaming reducer over an externally supplied square tape.

Research diagnostic only: the source tape is x*x, so this is not a submission
model.  No quotient, intermediate remainder, comparison, subtraction, carry,
or trace target is supplied to the learned cell or loss.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from torch import nn


BASE_PATH = (
    Path(__file__).resolve().parents[1]
    / "2026-08-17_binary_local_kernel_updates22_t1"
    / "full"
    / "source.py"
)
_spec = importlib.util.spec_from_file_location("binary_t1_base", BASE_PATH)
assert _spec is not None and _spec.loader is not None
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)


class StreamingExactSquareReducer(nn.Module):
    """Consume square bits MSB-first with one generic tied local cell."""

    def __init__(self, channels: int, updates: int, dropout: float) -> None:
        super().__init__()
        self.channels = channels
        self.microsteps_per_bit = updates
        self.dropout = dropout

        self.bit_embedding = nn.Embedding(2, channels)
        self.source_bit_embedding = nn.Embedding(2, channels)
        self.modulus_role = nn.Parameter(torch.randn(channels) * 0.02)
        self.work_role = nn.Parameter(torch.randn(channels) * 0.02)
        self.scratch_role = nn.Parameter(torch.randn(channels) * 0.02)
        self.source_role = nn.Parameter(torch.randn(channels) * 0.02)
        self.boundaries = nn.Parameter(torch.randn(2, channels) * 0.02)

        self.cell = base.Cell(channels)
        self.readout = nn.Conv1d(channels, 1, 1)

    def forward(
        self, source_bits: torch.Tensor, modulus_bits: torch.Tensor
    ) -> torch.Tensor:
        batch = source_bits.shape[0]
        positions = base.OPERAND_BITS

        modulus = self.bit_embedding(modulus_bits[:, :positions]).transpose(1, 2)
        modulus = modulus + self.modulus_role[None, :, None]

        state = modulus.new_zeros(batch, self.channels, base.LANES, positions)
        state[:, :, 1] = self.work_role[None, :, None]
        state[:, :, 2] = self.scratch_role[None, :, None]

        dropout_mask = None
        if self.training and self.dropout:
            keep = 1.0 - self.dropout
            dropout_mask = torch.empty(
                batch, self.channels, 1, 1, device=state.device
            ).bernoulli_(keep) / keep

        # tensor_batch emits source bits LSD-first; consume the same tensor in
        # reverse order.  The loop only routes input tokens.  Every learned
        # state transition uses the exact same randomly initialized cell.
        for source_bit in source_bits.flip(1).unbind(1):
            bit_token = (
                self.source_bit_embedding(source_bit)
                + self.source_role[None, :]
            )
            for _ in range(self.microsteps_per_bit):
                visible = state.clone()
                visible[:, :, 0] = modulus
                visible[:, :, 3] = 0.0
                visible[:, :, 3, 0] = bit_token
                visible[:, :, :, 0] += self.boundaries[0][None, :, None]
                visible[:, :, :, -1] += self.boundaries[1][None, :, None]
                state = self.cell(visible, dropout_mask)

        return self.readout(state[:, :, 1]).squeeze(1)


def _output_path() -> Path | None:
    try:
        return Path(sys.argv[sys.argv.index("--out") + 1])
    except (ValueError, IndexError):
        return None


if __name__ == "__main__":
    base.BinaryWorkState = StreamingExactSquareReducer
    base.main()
    output_path = _output_path()
    if output_path is not None:
        (output_path / "streaming_source.py").write_text(
            Path(__file__).read_text()
        )

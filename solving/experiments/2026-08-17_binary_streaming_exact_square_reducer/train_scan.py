"""Streaming exact-square reducer with a tied bidirectional digit scan.

Research diagnostic only.  The exact-square source is externally supplied,
while the learned model receives no arithmetic trace or intermediate target.
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


class StreamingScanReducer(nn.Module):
    """Apply one shared learned whole-tape transition per source bit."""

    def __init__(self, channels: int, updates: int, dropout: float) -> None:
        super().__init__()
        self.channels = channels
        self.refinements_per_bit = updates
        self.dropout = dropout

        self.bit_embedding = nn.Embedding(2, channels)
        self.source_bit_embedding = nn.Embedding(2, channels)
        self.modulus_role = nn.Parameter(torch.randn(channels) * 0.02)
        self.boundaries = nn.Parameter(torch.randn(2, channels) * 0.02)

        self.scan = nn.GRU(
            input_size=channels,
            hidden_size=channels,
            batch_first=True,
            bidirectional=True,
        )
        self.candidate = nn.Linear(2 * channels, channels)
        self.update = nn.Linear(2 * channels, channels)
        self.readout = nn.Linear(channels, 1)

    def forward(
        self, source_bits: torch.Tensor, modulus_bits: torch.Tensor
    ) -> torch.Tensor:
        batch = source_bits.shape[0]
        positions = base.OPERAND_BITS
        modulus = self.bit_embedding(modulus_bits[:, :positions])
        modulus = modulus + self.modulus_role[None, None, :]
        state = modulus.new_zeros(batch, positions, self.channels)

        dropout_mask = None
        if self.training and self.dropout:
            keep = 1.0 - self.dropout
            dropout_mask = torch.empty(
                batch, 1, self.channels, device=state.device
            ).bernoulli_(keep) / keep

        for source_bit in source_bits.flip(1).unbind(1):
            bit_token = self.source_bit_embedding(source_bit)
            for _ in range(self.refinements_per_bit):
                visible = state + modulus
                visible = visible.clone()
                visible[:, 0] += bit_token + self.boundaries[0][None, :]
                visible[:, -1] += self.boundaries[1][None, :]
                scanned, _ = self.scan(visible)
                candidate = torch.tanh(self.candidate(scanned))
                if dropout_mask is not None:
                    candidate = candidate * dropout_mask
                gate = torch.sigmoid(
                    self.update(torch.cat((visible, candidate), dim=-1))
                )
                state = (1.0 - gate) * state + gate * candidate

        return self.readout(state).squeeze(-1)


def _output_path() -> Path | None:
    try:
        return Path(sys.argv[sys.argv.index("--out") + 1])
    except (ValueError, IndexError):
        return None


if __name__ == "__main__":
    base.BinaryWorkState = StreamingScanReducer
    base.main()
    output_path = _output_path()
    if output_path is not None:
        (output_path / "streaming_scan_source.py").write_text(
            Path(__file__).read_text()
        )

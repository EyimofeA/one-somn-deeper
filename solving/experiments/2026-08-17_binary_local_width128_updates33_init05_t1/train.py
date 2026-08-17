"""Half-scale initialization ablation on the promoted local T=1 machine."""
from __future__ import annotations

import importlib.util
from pathlib import Path


BASE_PATH = (
    Path(__file__).resolve().parents[1]
    / "2026-08-16_binary_workstate_matched"
    / "train.py"
)
_spec = importlib.util.spec_from_file_location("binary_matched_base", BASE_PATH)
assert _spec is not None and _spec.loader is not None
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)


class HalfScaleBinaryWorkState(base.BinaryWorkState):
    """Use the identical model after scaling its random initialization by 0.5."""

    def __init__(self, channels: int, updates: int, dropout: float) -> None:
        super().__init__(channels, updates, dropout)
        with base.torch.no_grad():
            for parameter in self.parameters():
                parameter.mul_(0.5)


base.BinaryWorkState = HalfScaleBinaryWorkState


if __name__ == "__main__":
    base.main()

"""Lesson 2: see how local recurrent updates move information across a tape."""

import torch
from torch import nn


class ShiftRight(nn.Module):
    """A fixed teaching convolution that moves a signal one cell right."""

    def __init__(self) -> None:
        super().__init__()
        self.shift = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        with torch.no_grad():
            self.shift.weight.zero_()
            # PyTorch cross-correlation: output[i] reads input[i - 1].
            self.shift.weight[0, 0, 0] = 1.0
        self.shift.weight.requires_grad_(False)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Input/output shape: [batch, positions, channels=1].
        return self.shift(state.transpose(1, 2)).transpose(1, 2)


def show(step: int, state: torch.Tensor) -> None:
    values = [int(value) for value in state[0, :, 0].tolist()]
    print(f"step {step}: {values}")


def main() -> None:
    torch.set_default_device("cpu")

    state = torch.zeros(1, 8, 1)
    state[0, 4, 0] = 1.0
    transition = ShiftRight()

    print("One signal starts at position 1.")
    print("The same radius-one transition moves it one position per step.\n")
    show(0, state)

    for step in range(1, 7):
        state = transition(state)
        show(step, state)

    print("\nAfter K steps, a radius-one update can move information at most K positions.")
    print("The signal falls off the finite tape after reaching its right boundary.")


if __name__ == "__main__":
    main()

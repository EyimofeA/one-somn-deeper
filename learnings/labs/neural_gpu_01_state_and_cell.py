"""Lesson 1: inspect a tiny Neural-GPU-style recurrent cell on CPU."""

import torch
from torch import nn


class ConvGRUCell1D(nn.Module):
    """A GRU-like update shared across every position and recurrent step."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        # Kernel width 3 means each position reads: left, self, right.
        self.gates = nn.Conv1d(channels, 2 * channels, kernel_size=3, padding=1)
        self.candidate = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Friendly layout: [batch, positions, channels].
        # Conv1d expects: [batch, channels, positions].
        x = state.transpose(1, 2)

        update_logits, reset_logits = self.gates(x).chunk(2, dim=1)
        update = torch.sigmoid(update_logits)
        reset = torch.sigmoid(reset_logits)

        candidate = torch.tanh(self.candidate(reset * x))
        new_x = update * x + (1.0 - update) * candidate
        return new_x.transpose(1, 2)


def main() -> None:
    torch.manual_seed(0)
    torch.set_default_device("cpu")

    batch, positions, channels, steps = 2, 8, 16, 4
    state = torch.randn(batch, positions, channels, requires_grad=True)
    cell = ConvGRUCell1D(channels)

    print("initial state shape:", tuple(state.shape))
    print("cell object id:      ", id(cell))
    print("trainable parameters:", sum(p.numel() for p in cell.parameters()))

    recurrent_state = state
    for step in range(steps):
        recurrent_state = cell(recurrent_state)
        print(
            f"step {step + 1}: shape={tuple(recurrent_state.shape)} "
            f"mean={recurrent_state.mean().item():+.4f} "
            f"std={recurrent_state.std().item():.4f} "
            f"same_cell_id={id(cell)}"
        )

    # A deliberately simple loss: make the final state small.
    loss = recurrent_state.square().mean()
    loss.backward()

    print("loss:               ", round(loss.item(), 6))
    print("input gradient shape:", tuple(state.grad.shape))
    print("input gradient norm: ", round(state.grad.norm().item(), 6))
    print("device:              ", recurrent_state.device)


if __name__ == "__main__":
    main()

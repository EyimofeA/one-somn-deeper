"""Lesson 3: train a local convolution to discover a one-cell shift."""

import torch
from torch import nn


class LearnedLocalTransition(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.local = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Friendly shape: [batch, positions, channels=1].
        x = state.transpose(1, 2)
        return self.local(x).transpose(1, 2)


def desired_shift_right(state: torch.Tensor) -> torch.Tensor:
    """Training labels only: output position i copies input position i-1."""
    target = torch.zeros_like(state)
    target[:, 1:, :] = state[:, :-1, :]
    return target


def kernel(model: LearnedLocalTransition) -> list[float]:
    return [round(v, 4) for v in model.local.weight.detach().flatten().tolist()]


def row(tensor: torch.Tensor) -> list[float]:
    return [round(v, 3) for v in tensor[0, :, 0].detach().tolist()]


def main() -> None:
    torch.manual_seed(7)
    torch.set_default_device("cpu")

    model = LearnedLocalTransition()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.MSELoss()

    probe = torch.tensor([[[0.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]])

    print("Kernel order is [left neighbor, self, right neighbor].")
    print("A perfect shift-right kernel is approximately [1, 0, 0].\n")
    print("before training kernel:", kernel(model))
    print("probe input:           ", row(probe))
    print("before prediction:     ", row(model(probe)))
    print("desired output:        ", row(desired_shift_right(probe)))

    for step in range(301):
        # Fresh random tapes prevent memorizing one example.
        inputs = torch.randn(64, 8, 1)
        targets = desired_shift_right(inputs)

        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step in (0, 1, 10, 50, 100, 300):
            print(f"step {step:3d} loss={loss.item():.8f} kernel={kernel(model)}")

    print("\nafter prediction:      ", row(model(probe)))
    print("final kernel:          ", kernel(model))

    unseen = torch.randn(256, 8, 1)
    unseen_error = loss_fn(model(unseen), desired_shift_right(unseen)).item()
    print("unseen random-tape MSE:", f"{unseen_error:.10f}")


if __name__ == "__main__":
    main()

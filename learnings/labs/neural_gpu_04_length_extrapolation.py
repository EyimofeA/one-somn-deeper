"""Lesson 4: distinguish architectural possibility from measured evidence."""

import torch
from torch import nn


class LearnedLocalTransition(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.local = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.local(state.transpose(1, 2)).transpose(1, 2)


def shift_right(state: torch.Tensor) -> torch.Tensor:
    target = torch.zeros_like(state)
    target[:, 1:] = state[:, :-1]
    return target


def evaluate(model: nn.Module, length: int) -> tuple[float, float]:
    inputs = torch.randn(512, length, 1)
    targets = shift_right(inputs)
    predictions = model(inputs)
    mse = (predictions - targets).square().mean().item()
    max_error = (predictions - targets).abs().max().item()
    return mse, max_error


def main() -> None:
    torch.manual_seed(11)
    torch.set_default_device("cpu")

    training_length = 8
    model = LearnedLocalTransition()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    for _ in range(301):
        inputs = torch.randn(64, training_length, 1)
        loss = (model(inputs) - shift_right(inputs)).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    weights = model.local.weight.detach().flatten().tolist()
    print("trained only on length:", training_length)
    print("learned kernel:       ", [round(v, 6) for v in weights])
    print("\nEvaluation on fresh tapes:")
    for length in (8, 16, 64, 256):
        mse, max_error = evaluate(model, length)
        print(f"length {length:3d}: MSE={mse:.12f} max_error={max_error:.9f}")

    print("\nInterpretation:")
    print("The local rule has no parameter tied to absolute tape length.")
    print("That makes length transfer possible; these evaluations make it evidence.")


if __name__ == "__main__":
    main()

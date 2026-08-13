"""Lesson 11: weight decay prefers small parameters, not intended arithmetic."""

import torch
from torch import nn


class UnequalPaths(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.message_weight = nn.Parameter(torch.randn(()))
        self.shortcut_weight = nn.Parameter(torch.randn(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The shortcut has ten times more leverage per unit of parameter.
        return self.message_weight * x + 10.0 * self.shortcut_weight * x


def train(weight_decay: float, seed: int = 0) -> tuple[float, float, float, float]:
    torch.manual_seed(seed)
    model = UnequalPaths()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=weight_decay)
    x = torch.linspace(-1.0, 1.0, 201)
    target = 2.0 * x

    for _ in range(3000):
        loss = (model(x) - target).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        mse = (model(x) - target).square().mean().item()
        message = model.message_weight.item()
        shortcut = model.shortcut_weight.item()
        effective_sum = message + 10.0 * shortcut
    return mse, message, shortcut, effective_sum


def main() -> None:
    torch.set_default_device("cpu")

    print("Target: y = 2x")
    print("Model:  y = message_weight*x + 10*shortcut_weight*x")
    print("Intended message semantics would use message_weight near 1.\n")
    print("weight decay   final MSE    message weight   shortcut weight   effective sum")
    print("------------  -----------  ---------------  ----------------  -------------")

    for decay in (0.0, 0.001, 0.01, 0.1):
        mse, message, shortcut, effective_sum = train(decay)
        print(
            f"{decay:12.3g}  {mse:11.8f}  {message:15.6f}"
            f"  {shortcut:16.6f}  {effective_sum:13.6f}"
        )

    print("\nWeight decay rewards small raw parameters. Because the shortcut is scaled")
    print("by 10, a small shortcut parameter can explain most of the final function.")
    print("Regularization has no knowledge that the message weight 'should' mean 1.")
    print("Inductive bias depends on parameterization, architecture, data, and loss.")


if __name__ == "__main__":
    main()

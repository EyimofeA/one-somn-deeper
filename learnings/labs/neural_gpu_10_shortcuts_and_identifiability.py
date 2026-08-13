"""Lesson 10: perfect final answers do not identify the intended mechanism."""

import torch
from torch import nn


class TwoPathModel(nn.Module):
    """Both a message path and a shortcut path can explain the final target."""

    def __init__(self) -> None:
        super().__init__()
        self.message_writer = nn.Linear(1, 1, bias=False)
        self.shortcut = nn.Linear(1, 1, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        message = self.message_writer(x)
        shortcut_value = self.shortcut(x)
        answer = message + shortcut_value
        return answer, message, shortcut_value


def train(seed: int) -> tuple[float, float, float, float]:
    torch.manual_seed(seed)
    model = TwoPathModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    # The final function is y = 2x. We would like the message path to learn x,
    # but final labels never say how the two paths should divide the work.
    x = torch.linspace(-1.0, 1.0, 101).unsqueeze(1)
    y = 2.0 * x

    for _ in range(500):
        prediction, _, _ = model(x)
        loss = (prediction - y).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        prediction, message, shortcut = model(x)
        final_mse = (prediction - y).square().mean().item()
        message_mse = (message - x).square().mean().item()
        message_weight = model.message_writer.weight.item()
        shortcut_weight = model.shortcut.weight.item()
    return final_mse, message_mse, message_weight, shortcut_weight


def main() -> None:
    torch.set_default_device("cpu")

    print("Target: y = 2x")
    print("Desired internal story: message = x, shortcut = x")
    print("Training supervision: final y only\n")
    print("seed   final MSE    message MSE   message weight   shortcut weight   sum")
    print("----  -----------  ------------  ---------------  ----------------  -----")

    for seed in range(5):
        final_mse, message_mse, message_weight, shortcut_weight = train(seed)
        print(
            f" {seed:2d}   {final_mse:11.9f}  {message_mse:12.6f}"
            f"   {message_weight:15.6f}  {shortcut_weight:16.6f}"
            f"  {message_weight + shortcut_weight:5.2f}"
        )

    print("\nEvery model learns the final function because the two weights sum to 2.")
    print("Final labels do not require either internal path to have weight 1.")
    print("Different seeds therefore reach different zero-loss decompositions.")
    print("This is underidentification: correct behavior on training data does not")
    print("uniquely determine the internal algorithm or its unseen extension.")


if __name__ == "__main__":
    main()

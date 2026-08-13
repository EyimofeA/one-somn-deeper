"""Lesson 12: an easy training shortcut can fail under distribution shift."""

import torch
from torch import nn


class TwoFeatureModel(nn.Module):
    def __init__(self, allow_shortcut: bool) -> None:
        super().__init__()
        self.algorithm_weight = nn.Parameter(torch.randn(()))
        self.allow_shortcut = allow_shortcut
        if allow_shortcut:
            self.shortcut_weight = nn.Parameter(torch.randn(()))

    def forward(self, algorithm_feature: torch.Tensor, shortcut_feature: torch.Tensor):
        answer = self.algorithm_weight * algorithm_feature
        if self.allow_shortcut:
            answer = answer + self.shortcut_weight * shortcut_feature
        return answer


def fit(seed: int, allow_shortcut: bool):
    torch.manual_seed(seed)
    model = TwoFeatureModel(allow_shortcut)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    # Training shortcut is perfectly correlated with the true algorithm feature.
    train_x = torch.linspace(-1.0, 1.0, 201)
    train_shortcut = train_x
    train_target = train_x

    for _ in range(500):
        prediction = model(train_x, train_shortcut)
        loss = (prediction - train_target).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # OOD: the true rule is unchanged, but the shortcut correlation reverses.
    test_x = torch.linspace(-2.0, 2.0, 401)
    test_shortcut = -test_x
    test_target = test_x

    with torch.no_grad():
        train_mse = (model(train_x, train_shortcut) - train_target).square().mean().item()
        test_mse = (model(test_x, test_shortcut) - test_target).square().mean().item()
        algorithm_weight = model.algorithm_weight.item()
        shortcut_weight = model.shortcut_weight.item() if allow_shortcut else 0.0
    return train_mse, test_mse, algorithm_weight, shortcut_weight


def main() -> None:
    torch.set_default_device("cpu")

    print("True rule: answer = algorithm_feature")
    print("Training: shortcut_feature = algorithm_feature")
    print("OOD test: shortcut_feature = -algorithm_feature\n")
    print("model       seed   train MSE    OOD MSE      algorithm w   shortcut w")
    print("----------  ----  -----------  -----------  ------------  ----------")

    for allow_shortcut, name in ((True, "two paths"), (False, "forced alg")):
        for seed in range(5):
            train_mse, test_mse, algorithm, shortcut = fit(seed, allow_shortcut)
            print(
                f"{name:10s}   {seed:2d}   {train_mse:11.8f}  {test_mse:11.6f}"
                f"  {algorithm:12.6f}  {shortcut:10.6f}"
            )

    print("\nBoth architectures fit training. The two-path model can divide work")
    print("arbitrarily because both features agree there. When correlation flips,")
    print("shortcut reliance becomes error. Forcing the invariant feature trains")
    print("cleanly across every seed and transfers to OOD data.")


if __name__ == "__main__":
    main()

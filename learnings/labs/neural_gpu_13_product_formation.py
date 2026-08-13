"""Lesson 13: multiplication needs interactions, not only additive mixing."""

import torch
from torch import nn


class AffineProduct(nn.Module):
    """Can learn w1*a + w2*b + bias, but has no a*b interaction."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(2, 1)

    def forward(self, digits: torch.Tensor) -> torch.Tensor:
        return self.layer(digits)


class BilinearProduct(nn.Module):
    """Adds one generic multiplicative feature to the same two inputs."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(3, 1)

    def forward(self, digits: torch.Tensor) -> torch.Tensor:
        a, b = digits[:, :1], digits[:, 1:]
        features = torch.cat((a, b, a * b), dim=1)
        return self.layer(features)


def digit_data() -> tuple[torch.Tensor, torch.Tensor]:
    rows = [(float(a), float(b)) for a in range(10) for b in range(10)]
    inputs = torch.tensor(rows)
    targets = (inputs[:, :1] * inputs[:, 1:])
    return inputs, targets


def train(model: nn.Module) -> tuple[float, float]:
    inputs, targets = digit_data()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

    for _ in range(3000):
        predictions = model(inputs)
        loss = (predictions - targets).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        errors = (model(inputs) - targets).abs()
        return errors.mean().item(), errors.max().item()


def explain_square(x: int) -> None:
    tens, units = divmod(x, 10)
    columns = {
        0: [(units, units)],
        1: [(tens, units), (units, tens)],
        2: [(tens, tens)],
    }

    print(f"Raw product columns for {x}² before carrying:")
    for column, pairs in columns.items():
        terms = " + ".join(f"{a}×{b}" for a, b in pairs)
        total = sum(a * b for a, b in pairs)
        print(f"  column {column}: {terms:<11s} = {total}")
    print()


def main() -> None:
    torch.manual_seed(23)
    torch.set_default_device("cpu")

    explain_square(38)

    affine = AffineProduct()
    bilinear = BilinearProduct()
    affine_mean, affine_max = train(affine)
    bilinear_mean, bilinear_max = train(bilinear)

    print("Learning all 100 decimal digit products:")
    print("model       mean absolute error   maximum absolute error")
    print("----------  -------------------   ----------------------")
    print(f"affine      {affine_mean:19.6f}   {affine_max:22.6f}")
    print(f"bilinear    {bilinear_mean:19.6f}   {bilinear_max:22.6f}")

    print("\nLearned affine weights:", [round(v, 4) for v in affine.layer.weight.detach().flatten().tolist()])
    print("Learned bilinear weights [a, b, a×b]:", [round(v, 4) for v in bilinear.layer.weight.detach().flatten().tolist()])
    print()
    print("An affine cell can combine digits additively but cannot represent their")
    print("product exactly. The bilinear feature makes the interaction directly")
    print("available; training learns to select it and ignore the additive terms.")
    print("This still does not solve routing, accumulation, carry, or reduction.")


if __name__ == "__main__":
    main()

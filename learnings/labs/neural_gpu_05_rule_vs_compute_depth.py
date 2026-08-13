"""Lesson 5: separate transition learning from sufficient compute depth."""

import torch
from torch import nn


class LocalTransition(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.local = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.local(state.transpose(1, 2)).transpose(1, 2)


def shift_right(state: torch.Tensor) -> torch.Tensor:
    target = torch.zeros_like(state)
    target[:, 1:] = state[:, :-1]
    return target


def train_one_step_rule() -> LocalTransition:
    model = LocalTransition()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(301):
        inputs = torch.randn(64, 8, 1)
        loss = (model(inputs) - shift_right(inputs)).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def rollout(model: nn.Module, length: int, steps: int) -> torch.Tensor:
    state = torch.zeros(1, length, 1)
    state[0, 0, 0] = 1.0
    for _ in range(steps):
        state = model(state)
    return state


def show(length: int, steps: int, state: torch.Tensor) -> None:
    peak_position = int(state[0, :, 0].abs().argmax())
    final_value = state[0, -1, 0].item()
    print(
        f"length={length:2d} steps={steps:2d} "
        f"peak_position={peak_position:2d} final_cell={final_value:.6f}"
    )


def main() -> None:
    torch.manual_seed(17)
    torch.set_default_device("cpu")
    model = train_one_step_rule()

    print("The one-step local rule was trained only on length-eight tapes.")
    print("Learned kernel:", [round(v, 6) for v in model.local.weight.detach().flatten().tolist()])
    print()

    show(8, 7, rollout(model, length=8, steps=7))
    show(16, 7, rollout(model, length=16, steps=7))
    show(16, 15, rollout(model, length=16, steps=15))
    show(32, 15, rollout(model, length=32, steps=15))
    show(32, 31, rollout(model, length=32, steps=31))

    print("\nThe rule transfers to every length, but the signal reaches the end only")
    print("when recurrent compute depth is at least length minus one.")


if __name__ == "__main__":
    main()

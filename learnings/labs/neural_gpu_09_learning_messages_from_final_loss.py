"""Lesson 9: can a later answer loss teach an unlabeled earlier message?"""

import torch
from torch import nn


class MessageWriter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.writer = nn.Linear(1, 1)

    def forward(self, units_total: torch.Tensor) -> torch.Tensor:
        return self.writer(units_total)


def train(causal: bool) -> tuple[MessageWriter, list[tuple[int, float, float]]]:
    torch.manual_seed(3)
    model = MessageWriter()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

    units_total = torch.tensor([[64.0]])
    base_tens = torch.tensor([[48.0]])
    correct_tens_total = torch.tensor([[54.0]])
    history = []

    for step in range(201):
        message = model(units_total)

        if causal:
            later_content = base_tens + message
        else:
            # Keep a zero-strength connection so backward is valid while the
            # answer remains behaviorally independent of the message.
            later_content = base_tens + 0.0 * message

        loss = (later_content - correct_tens_total).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step in (0, 1, 10, 50, 100, 200):
            history.append((step, message.item(), loss.item()))

    return model, history


def show(name: str, history: list[tuple[int, float, float]]) -> None:
    print(name)
    print(" step   written message       final loss")
    for step, message, loss in history:
        print(f" {step:4d}   {message:15.6f}   {loss:14.8f}")
    print()


def main() -> None:
    torch.set_default_device("cpu")

    _, terminal_history = train(causal=False)
    _, causal_history = train(causal=True)

    print("Training target: later column total must be 54.")
    print("No target ever says that the message itself should equal 6.\n")
    show("TERMINAL / DISCONNECTED WRITER", terminal_history)
    show("CAUSALLY CONSUMED WRITER", causal_history)

    print("The causal writer discovers a message near 6 because that value fixes")
    print("the later answer. The disconnected writer receives no useful credit.")
    print("Real modular arithmetic is harder: the final remainder hides many such")
    print("messages behind many recurrent steps and admits shortcut solutions.")


if __name__ == "__main__":
    main()

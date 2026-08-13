"""Lesson 6: repeated transitions can erase or explode state and gradients."""

import torch


def run(scale: float, steps: int) -> tuple[float, float]:
    initial = torch.tensor(1.0, requires_grad=True)
    state = initial
    for _ in range(steps):
        state = scale * state

    # The final state itself is the loss so its gradient is easy to interpret.
    state.backward()
    return state.item(), initial.grad.item()


def main() -> None:
    torch.set_default_device("cpu")

    print("Repeated scalar transition: state_next = scale × state\n")
    print(" scale  steps      final state    gradient to initial")
    print("------  -----  ---------------  ---------------------")

    for scale in (0.90, 0.99, 1.00, 1.01, 1.10):
        for steps in (10, 100):
            final_state, gradient = run(scale, steps)
            print(f" {scale:4.2f}   {steps:4d}  {final_state:15.8g}  {gradient:21.8g}")

    print("\nFor this simple transition, both values equal scale raised to steps.")
    print("Below 1 repeatedly shrinks information and credit; above 1 amplifies both.")
    print("Real recurrent networks use matrices and nonlinear gates, but products of")
    print("many local Jacobians create the same stability problem.")


if __name__ == "__main__":
    main()

"""Lesson 8: compare terminal-only and causally consumed message paths."""

import torch


def terminal_only(base_tens: torch.Tensor, message: torch.Tensor):
    """Message is saved for the final readout but not used to form content."""
    next_content = base_tens
    shadow = message
    return next_content, shadow


def causal_message(base_tens: torch.Tensor, message: torch.Tensor):
    """Message enters the next content state before later computation."""
    next_content = base_tens + message
    shadow = message
    return next_content, shadow


def digit_and_carry(column_total: torch.Tensor):
    """Teaching-only arithmetic readout; not a neural competition forward."""
    digit = torch.remainder(column_total, 10.0)
    carry = torch.floor(column_total / 10.0)
    return digit, carry


def gradient_to_message(mode: str) -> float:
    # Use an incorrect candidate message so a connected path has nonzero loss
    # and therefore a visible corrective gradient toward the true value 6.
    message = torch.tensor(5.0, requires_grad=True)
    # Give the loss some valid differentiable path in both arms. The question is
    # whether it additionally has a path to `message`.
    base_tens = torch.tensor(48.0, requires_grad=True)

    if mode == "terminal":
        content, _shadow = terminal_only(base_tens, message)
    else:
        content, _shadow = causal_message(base_tens, message)

    # Later content should equal the correct column total, 54.
    loss = (content - 54.0).square()
    gradient = torch.autograd.grad(loss, message, allow_unused=True)[0]
    return 0.0 if gradient is None else gradient.item()


def main() -> None:
    torch.set_default_device("cpu")

    base_tens = torch.tensor(48.0)  # 3×8 + 8×3
    units_carry = torch.tensor(6.0)  # from 8×8 = 64

    terminal_content, terminal_shadow = terminal_only(base_tens, units_carry)
    causal_content, causal_shadow = causal_message(base_tens, units_carry)

    terminal_digit, terminal_next_carry = digit_and_carry(terminal_content)
    causal_digit, causal_next_carry = digit_and_carry(causal_content)

    print("base tens products:       ", base_tens.item())
    print("message from units:       ", units_carry.item())
    print()
    print("TERMINAL-ONLY")
    print("  content after update:   ", terminal_content.item())
    print("  shadow stores message:  ", terminal_shadow.item())
    print("  tens digit formed:      ", terminal_digit.item())
    print("  next carry formed:      ", terminal_next_carry.item())
    print("  gradient from content loss to message:", gradient_to_message("terminal"))
    print()
    print("CAUSAL")
    print("  content after update:   ", causal_content.item())
    print("  shadow stores message:  ", causal_shadow.item())
    print("  tens digit formed:      ", causal_digit.item())
    print("  next carry formed:      ", causal_next_carry.item())
    print("  gradient from content loss to message:", gradient_to_message("causal"))
    print()
    print("Both arms store the message. Only the causal arm consumes it while")
    print("forming the next content state, digit, and downstream carry.")


if __name__ == "__main__":
    main()

"""Lesson 7: a hidden variable can be perfectly decodable but behaviorally unused."""

import torch


def main() -> None:
    torch.set_default_device("cpu")

    # Think of content as the path that currently produces an answer.
    content = torch.tensor([2.0, 4.0, 6.0, 8.0])

    # Think of shadow_carry as a separate hidden lane. It stores carry perfectly.
    shadow_carry = torch.tensor([0.0, 1.0, 3.0, 6.0])

    # A perfect probe: reading the shadow lane recovers every carry exactly.
    probe_prediction = shadow_carry.clone()
    probe_accuracy = (probe_prediction == shadow_carry).float().mean().item()

    # But the model's answer path ignores the shadow lane completely.
    answer_before = 3.0 * content + 1.0

    # Causal intervention: erase all stored carries.
    ablated_shadow_carry = torch.zeros_like(shadow_carry)
    answer_after = 3.0 * content + 1.0

    print("stored carry:       ", shadow_carry.tolist())
    print("probe prediction:   ", probe_prediction.tolist())
    print("probe accuracy:     ", f"{100 * probe_accuracy:.1f}%")
    print("answer before erase:", answer_before.tolist())
    print("answer after erase: ", answer_after.tolist())
    print("answer changed:     ", bool((answer_before != answer_after).any()))
    print()
    print("The carry is represented perfectly, but deleting it changes nothing.")
    print("Therefore probe accuracy proves presence, not causal use.")

    # Keep this variable visible so the intervention itself is explicit.
    assert torch.equal(ablated_shadow_carry, torch.zeros_like(shadow_carry))


if __name__ == "__main__":
    main()

"""Lesson 14: product accumulation and carry propagation are separate failures."""


def normalize_columns(raw_columns: list[int]) -> tuple[list[int], list[int]]:
    """Teaching oracle: normalize LSD-first raw sums into decimal digits."""
    digits = []
    carries = []
    carry = 0

    for raw in raw_columns:
        total = raw + carry
        digits.append(total % 10)
        carry = total // 10
        carries.append(carry)

    while carry:
        digits.append(carry % 10)
        carry //= 10
        carries.append(carry)
    return digits, carries


def as_number(lsd_first_digits: list[int]) -> int:
    return sum(digit * (10**position) for position, digit in enumerate(lsd_first_digits))


def main() -> None:
    # Correct two-digit square for 38², before decimal normalization.
    correct_raw = [8 * 8, 3 * 8 + 8 * 3, 3 * 3]

    # Failure A: form only diagonal products; cross-products never accumulate.
    missing_cross_products = [8 * 8, 0, 3 * 3]

    # Failure B: form every product correctly but never pass carry between columns.
    no_carry_digits = [raw % 10 for raw in correct_raw]

    correct_digits, correct_carries = normalize_columns(correct_raw)
    missing_digits, missing_carries = normalize_columns(missing_cross_products)

    print("All lists are least-significant position first.\n")
    print("Correct raw column sums:             ", correct_raw)
    print("Correct outgoing carries:            ", correct_carries)
    print("Correct normalized digits:           ", correct_digits)
    print("Correct number:                      ", as_number(correct_digits))
    print()
    print("A. Missing cross-product accumulation")
    print("Raw column sums:                     ", missing_cross_products)
    print("Carries still propagated correctly:  ", missing_carries)
    print("Normalized digits:                   ", missing_digits)
    print("Wrong number:                        ", as_number(missing_digits))
    print()
    print("B. Correct products, missing carry")
    print("Raw column sums remain correct:      ", correct_raw)
    print("Digits formed independently:         ", no_carry_digits)
    print("Wrong number:                        ", as_number(no_carry_digits))
    print()
    print("Both failures damage middle digits, but they require different fixes:")
    print("A needs product formation/routing/accumulation; B needs temporal messages.")


if __name__ == "__main__":
    main()

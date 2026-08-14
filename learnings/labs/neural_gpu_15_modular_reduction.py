"""Lesson 15: modular reduction depends on N and on an efficient schedule."""


def repeated_subtraction(value: int, modulus: int) -> tuple[int, int]:
    """Correct but O(quotient): subtract N until the value is below N."""
    steps = 0
    while value >= modulus:
        value -= modulus
        steps += 1
    return value, steps


def shifted_subtraction(value: int, modulus: int) -> tuple[int, int, list[tuple[int, int]]]:
    """Teaching long-division schedule using powers of ten."""
    steps = 0
    trace = []

    shift = 1
    while modulus * shift * 10 <= value:
        shift *= 10

    while shift >= 1:
        count = 0
        aligned = modulus * shift
        while value >= aligned:
            value -= aligned
            count += 1
            steps += 1
        trace.append((shift, count))
        shift //= 10

    return value, steps, trace


def main() -> None:
    square = 38 * 38

    print(f"Hold x² fixed: 38² = {square}\n")
    print(" N    quotient   remainder")
    print("---   --------   ---------")
    for modulus in (73, 77, 78, 91, 97):
        quotient, remainder = divmod(square, modulus)
        print(f"{modulus:3d}   {quotient:8d}   {remainder:9d}")

    print("\nNearby x values with fixed N=77:")
    print(" x      x²   quotient   remainder")
    print("---   -----  --------   ---------")
    for x in range(36, 42):
        quotient, remainder = divmod(x * x, 77)
        print(f"{x:3d}   {x*x:5d}  {quotient:8d}   {remainder:9d}")

    print("\nTwo exact reduction schedules for 1444 mod 77:")
    repeated_remainder, repeated_steps = repeated_subtraction(square, 77)
    shifted_remainder, shifted_steps, trace = shifted_subtraction(square, 77)
    print(f"repeated subtraction: remainder={repeated_remainder}, subtractions={repeated_steps}")
    print(f"shifted subtraction:  remainder={shifted_remainder}, subtractions={shifted_steps}")
    print("shift trace (decimal shift, times subtracted):", trace)

    large_value = 9_999_999
    modulus = 97
    _, repeated_large_steps = repeated_subtraction(large_value, modulus)
    _, shifted_large_steps, _ = shifted_subtraction(large_value, modulus)
    print(f"\nFor {large_value} mod {modulus}:")
    print(f"repeated subtraction needs {repeated_large_steps:,} learned subtract steps")
    print(f"shifted subtraction needs  {shifted_large_steps:,} subtract actions")

    print("\nA neural reducer must learn N-conditioned comparison and subtraction.")
    print("It must also learn or induce an efficient schedule; correctness alone is")
    print("not enough when the quotient becomes large and runtime is fixed.")


if __name__ == "__main__":
    main()

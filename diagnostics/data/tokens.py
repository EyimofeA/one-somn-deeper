"""Shared vocabulary and fixed-width digit tokenization for the diagnostic suite.

Every field is tokenized as a fixed number of MSB-first decimal digits
(zero-padded), so tensor shapes never depend on which split/task produced a
row. Widths are sized for N up to 13 bits (8191): N and x each fit in 4
digits, and u / x^2 (bounded by (N-1)^2) each fit in 8 digits.
"""

from __future__ import annotations

PAD = 0
SQUARE = 1
MOD = 2
SQUARE_MOD = 3
SQUARE_MOD_TRACE = 4
N_MARKER = 5
X_MARKER = 6
U_MARKER = 7
OUT = 8
DIGIT_OFFSET = 9
VOCAB_SIZE = DIGIT_OFFSET + 10  # 19

IGNORE_INDEX = -100

NUM_N_DIGITS = 4
NUM_X_DIGITS = 4         # Task C/D's x: tied to N's domain (x < N <= 9999)
NUM_U_DIGITS = 8
NUM_MOD_DIGITS = 4       # width of u mod N / x^2 mod N (always < N <= 9999)

# Task A's x is NOT tied to N (squaring alone has no modulus in the input),
# so it gets its own, much wider domain -- otherwise a 4-digit x pool (9999
# possible values) can't supply the spec's suggested 100k+10k+10k+10k unique
# rows without either exhausting the pool or violating the no-duplicate-x
# requirement.
NUM_SQUARE_X_DIGITS = 6
NUM_SQUARE_DIGITS = 12   # width of x^2 for a 6-digit x (999999^2 has 12 digits)

TASKS = ("square", "mod", "square_mod", "square_mod_trace")


def digit_tokens(value: int, width: int, reverse: bool = False) -> list[int]:
    s = f"{value:0{width}d}"
    if len(s) > width:
        raise ValueError(f"value {value} does not fit in {width} digits")
    digits = [DIGIT_OFFSET + int(c) for c in s]
    if reverse:
        digits.reverse()
    return digits


def encode_square(x: int, reverse_digits: bool = False) -> tuple[list[int], list[int]]:
    """Task A: <SQUARE> X <digits of x> <OUT>*12 -> digits of x^2."""
    input_ids = [SQUARE, X_MARKER, *digit_tokens(x, NUM_SQUARE_X_DIGITS, reverse_digits)]
    input_ids += [OUT] * NUM_SQUARE_DIGITS
    labels = [IGNORE_INDEX] * (len(input_ids) - NUM_SQUARE_DIGITS)
    labels += [t - DIGIT_OFFSET for t in digit_tokens(x * x, NUM_SQUARE_DIGITS)]
    return input_ids, labels


def encode_mod(n: int, u: int, reverse_digits: bool = False) -> tuple[list[int], list[int]]:
    """Task B: <MOD> N <digits of N> U <digits of u> <OUT>*4 -> digits of u mod N."""
    input_ids = [MOD, N_MARKER, *digit_tokens(n, NUM_N_DIGITS, reverse_digits)]
    input_ids += [U_MARKER, *digit_tokens(u, NUM_U_DIGITS, reverse_digits)]
    input_ids += [OUT] * NUM_MOD_DIGITS
    labels = [IGNORE_INDEX] * (len(input_ids) - NUM_MOD_DIGITS)
    labels += [t - DIGIT_OFFSET for t in digit_tokens(u % n, NUM_MOD_DIGITS)]
    return input_ids, labels


def encode_square_mod(n: int, x: int, reverse_digits: bool = False) -> tuple[list[int], list[int]]:
    """Task C: <SQUARE_MOD> N <digits of N> X <digits of x> <OUT>*4 -> digits of x^2 mod N."""
    input_ids = [SQUARE_MOD, N_MARKER, *digit_tokens(n, NUM_N_DIGITS, reverse_digits)]
    input_ids += [X_MARKER, *digit_tokens(x, NUM_X_DIGITS, reverse_digits)]
    input_ids += [OUT] * NUM_MOD_DIGITS
    labels = [IGNORE_INDEX] * (len(input_ids) - NUM_MOD_DIGITS)
    labels += [t - DIGIT_OFFSET for t in digit_tokens((x * x) % n, NUM_MOD_DIGITS)]
    return input_ids, labels


def encode_square_mod_trace(n: int, x: int, reverse_digits: bool = False) -> tuple[list[int], list[int]]:
    """Task D: same input as C, aux head over x^2 (8 slots) then main head over x^2 mod N (4 slots)."""
    input_ids = [SQUARE_MOD_TRACE, N_MARKER, *digit_tokens(n, NUM_N_DIGITS, reverse_digits)]
    input_ids += [X_MARKER, *digit_tokens(x, NUM_X_DIGITS, reverse_digits)]
    input_ids += [OUT] * (NUM_SQUARE_DIGITS + NUM_MOD_DIGITS)
    prefix_len = len(input_ids) - (NUM_SQUARE_DIGITS + NUM_MOD_DIGITS)
    labels = [IGNORE_INDEX] * prefix_len
    labels += [t - DIGIT_OFFSET for t in digit_tokens(x * x, NUM_SQUARE_DIGITS)]
    labels += [t - DIGIT_OFFSET for t in digit_tokens((x * x) % n, NUM_MOD_DIGITS)]
    return input_ids, labels


ENCODERS = {
    "square": encode_square,
    "mod": encode_mod,
    "square_mod": encode_square_mod,
    "square_mod_trace": encode_square_mod_trace,
}

# number of trailing output slots each task's labels/predictions occupy
OUTPUT_WIDTH = {
    "square": NUM_SQUARE_DIGITS,
    "mod": NUM_MOD_DIGITS,
    "square_mod": NUM_MOD_DIGITS,
    "square_mod_trace": NUM_SQUARE_DIGITS + NUM_MOD_DIGITS,
}

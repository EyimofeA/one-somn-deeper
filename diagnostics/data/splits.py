"""Prime/modulus pools and the six required train/test splits.

All randomness goes through an explicit random.Random instance so runs are
reproducible from a seed. Nothing here touches torch; this module only
produces plain Python ints/tuples that data/generate.py turns into records.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def primes_in_bit_range(lo_bits: int, hi_bits: int) -> list[int]:
    lo = 1 << (lo_bits - 1)
    hi = (1 << hi_bits) - 1
    return [n for n in range(max(lo, 2), hi + 1) if is_prime(n)]


def semiprimes_with_bits(primes: list[int], lo_bits: int, hi_bits: int) -> list[tuple[int, int, int]]:
    """All (N, p, q), p<q, distinct primes from `primes`, N=p*q with bit_length in range."""
    out = []
    for i, p in enumerate(primes):
        for q in primes[i + 1 :]:
            n = p * q
            if lo_bits <= n.bit_length() <= hi_bits:
                out.append((n, p, q))
    return out


def coprime_pool(n: int, p: int, q: int) -> range:
    """x in [1, n) with gcd(x, n) = 1 == x not a multiple of p or q (n = p*q, distinct primes)."""
    return (x for x in range(1, n) if x % p != 0 and x % q != 0)


def sample_coprime_x(n: int, p: int, q: int, count: int, rng: random.Random) -> list[int]:
    pool = [x for x in coprime_pool(n, p, q)]
    if count > len(pool):
        raise ValueError(f"requested {count} coprime x values but only {len(pool)} exist for N={n}")
    return rng.sample(pool, count)


@dataclass
class ModulusPool:
    """Semiprimes for the 10-11 bit ("in-distribution") regime, plus disjoint
    prime sub-pools reserved for the held-out-factor split."""

    bit_lo: int = 10
    bit_hi: int = 11
    seed: int = 0
    train_moduli: list[tuple[int, int, int]] = field(default_factory=list)
    test_modulus_moduli: list[tuple[int, int, int]] = field(default_factory=list)
    factor_train_primes: list[int] = field(default_factory=list)
    factor_test_primes: list[int] = field(default_factory=list)

    def build(self, n_train_moduli: int, n_test_moduli: int) -> None:
        rng = random.Random(self.seed)
        primes = primes_in_bit_range(4, self.bit_hi)  # small primes; product lands in bit_lo..bit_hi
        candidates = semiprimes_with_bits(primes, self.bit_lo, self.bit_hi)
        rng.shuffle(candidates)
        total = n_train_moduli + n_test_moduli
        if total > len(candidates):
            raise ValueError(
                f"requested {total} distinct moduli but only {len(candidates)} semiprimes exist "
                f"in the {self.bit_lo}-{self.bit_hi} bit range"
            )
        self.train_moduli = candidates[:n_train_moduli]
        self.test_modulus_moduli = candidates[n_train_moduli : n_train_moduli + n_test_moduli]

        # held-out-factor: partition the prime pool itself, then only keep
        # semiprimes fully inside one side or the other.
        shuffled_primes = list(primes)
        rng.shuffle(shuffled_primes)
        half = len(shuffled_primes) // 2
        self.factor_train_primes = shuffled_primes[:half]
        self.factor_test_primes = shuffled_primes[half:]


def held_out_factor_moduli(
    train_primes: list[int], test_primes: list[int], bit_lo: int, bit_hi: int, count: int, rng: random.Random
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    train_candidates = semiprimes_with_bits(train_primes, bit_lo, bit_hi)
    test_candidates = semiprimes_with_bits(test_primes, bit_lo, bit_hi)
    rng.shuffle(train_candidates)
    rng.shuffle(test_candidates)
    if count > len(train_candidates) or count > len(test_candidates):
        raise ValueError(
            f"held-out-factor split needs {count} moduli per side; have "
            f"{len(train_candidates)} train-side / {len(test_candidates)} test-side"
        )
    return train_candidates[:count], test_candidates[:count]


def assert_no_modulus_overlap(train: list[tuple[int, int, int]], test: list[tuple[int, int, int]]) -> None:
    train_ns = {n for n, _, _ in train}
    test_ns = {n for n, _, _ in test}
    overlap = train_ns & test_ns
    if overlap:
        raise AssertionError(f"modulus overlap between train/test: {sorted(overlap)[:5]}...")


def assert_no_factor_overlap(train: list[tuple[int, int, int]], test: list[tuple[int, int, int]]) -> None:
    train_primes = {p for _, p, _ in train} | {q for _, _, q in train}
    test_primes = {p for _, p, _ in test} | {q for _, _, q in test}
    overlap = train_primes & test_primes
    if overlap:
        raise AssertionError(f"prime factor overlap between train/test: {sorted(overlap)[:5]}...")


def hard_x_candidates(n: int, p: int, q: int) -> list[int]:
    """x values that stress carry chains / near-boundary remainders for this N."""
    sqrt_n = int(math.isqrt(n))
    candidates = set()
    # near sqrt(N): x^2 close to a single multiple of N
    for delta in range(-6, 7):
        x = sqrt_n + delta
        if 1 <= x < n:
            candidates.add(x)
    # near N itself: x close to N (x^2 has a large, near-boundary quotient)
    for delta in range(1, 8):
        x = n - delta
        if 1 <= x < n:
            candidates.add(x)
    # long carry-chain decimal patterns: repeated 9s / repeated 1s near x's width
    width = len(str(n))
    for pattern in ("9" * width, "1" * width, "9" * (width - 1) + "1", "1" + "9" * (width - 1)):
        x = int(pattern) % n
        if x >= 1:
            candidates.add(x)
    return sorted(c for c in candidates if math.gcd(c, n) == 1)

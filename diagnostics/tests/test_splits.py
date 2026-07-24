import random

from data import splits as sp


def test_is_prime_basic():
    assert not sp.is_prime(1)
    assert sp.is_prime(2)
    assert sp.is_prime(3)
    assert not sp.is_prime(4)
    assert sp.is_prime(97)
    assert not sp.is_prime(91)  # 7*13


def test_primes_in_bit_range_all_prime_and_in_range():
    primes = sp.primes_in_bit_range(10, 11)
    assert primes
    for p in primes:
        assert sp.is_prime(p)
        assert 10 <= p.bit_length() <= 11


def test_semiprimes_with_bits_are_products_of_distinct_primes_in_range():
    primes = sp.primes_in_bit_range(4, 11)
    sems = sp.semiprimes_with_bits(primes, 10, 11)
    assert sems
    for n, p, q in sems:
        assert p != q
        assert n == p * q
        assert 10 <= n.bit_length() <= 11


def test_modulus_pool_train_test_disjoint():
    pool = sp.ModulusPool(bit_lo=10, bit_hi=11, seed=1)
    pool.build(n_train_moduli=10, n_test_moduli=5)
    sp.assert_no_modulus_overlap(pool.train_moduli, pool.test_modulus_moduli)  # must not raise


def test_held_out_factor_split_has_zero_prime_overlap():
    pool = sp.ModulusPool(bit_lo=10, bit_hi=11, seed=2)
    pool.build(n_train_moduli=10, n_test_moduli=5)
    rng = random.Random(3)
    train, test = sp.held_out_factor_moduli(
        pool.factor_train_primes, pool.factor_test_primes, pool.bit_lo, pool.bit_hi, count=3, rng=rng
    )
    sp.assert_no_factor_overlap(train, test)  # must not raise


def test_assert_no_modulus_overlap_actually_catches_overlap():
    train = [(15, 3, 5)]
    test = [(15, 3, 5)]
    try:
        sp.assert_no_modulus_overlap(train, test)
    except AssertionError:
        pass
    else:
        raise AssertionError("expected an overlap to be detected")


def test_assert_no_factor_overlap_actually_catches_overlap():
    train = [(15, 3, 5)]
    test = [(21, 3, 7)]  # shares prime factor 3
    try:
        sp.assert_no_factor_overlap(train, test)
    except AssertionError:
        pass
    else:
        raise AssertionError("expected a shared prime factor to be detected")


def test_sample_coprime_x_is_actually_coprime():
    n, p, q = 15, 3, 5
    rng = random.Random(4)
    xs = sp.sample_coprime_x(n, p, q, count=6, rng=rng)
    assert len(xs) == 6 == len(set(xs))
    for x in xs:
        assert 1 <= x < n
        assert x % p != 0 and x % q != 0


def test_hard_x_candidates_are_coprime_and_in_range():
    n, p, q = 323, 17, 19
    xs = sp.hard_x_candidates(n, p, q)
    assert xs
    for x in xs:
        assert 1 <= x < n
        assert (x % p != 0) and (x % q != 0)

import numpy as np

from ppa_real.core.projections import GATES_24, dr, gate_bits, mirror, wheel
from ppa_real.core.sieve import primes_between


def test_sieve_matches_known_counts():
    assert len(primes_between(0, 10 ** 6)) == 78498
    assert len(primes_between(0, 1000)) + len(primes_between(1000, 10 ** 5)) == 9592


def test_digital_root_and_mirror():
    assert dr(123456789) == 9 and dr(10) == 1
    assert all(dr((9 - a) * b) in (9, 9 - dr(a * b)) for a in range(1, 9) for b in range(1, 10))
    assert mirror(2) == 7 and mirror(3, 24) == 21


def test_gates_are_three_bits():
    assert GATES_24 == wheel(24)
    for a in GATES_24:
        for b in GATES_24:
            assert gate_bits(a * b % 24) == gate_bits(a) ^ gate_bits(b)


def test_real_primes_start_at_5():
    from ppa_real.core.sieve import real_primes
    P = real_primes(0, 100)
    assert P[0] == 5 and 2 not in P and 3 not in P and len(P) == 25 - 2

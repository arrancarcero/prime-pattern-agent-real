"""Segmented sieve: the agent's source of primes."""
from __future__ import annotations

import math

import numpy as np


def small_primes(n: int) -> np.ndarray:
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return np.nonzero(s)[0]


def primes_between(lo: int, hi: int, seg: int = 10_000_000) -> np.ndarray:
    """All primes in [lo, hi), in blocks so memory stays small."""
    base = small_primes(int(math.isqrt(hi)) + 1)
    out = []
    for a in range(lo, hi, seg):
        b = min(a + seg, hi)
        s = np.ones(b - a, dtype=bool)
        for q in base:
            if q * q >= b:
                break
            start = max(q * q, ((a + q - 1) // q) * q)
            s[start - a::q] = False
        if a <= 1:
            s[: 2 - a] = False
        out.append(np.nonzero(s)[0] + a)
    return np.concatenate(out) if out else np.array([], dtype=np.int64)


START = 5   # the real-primes agent uses primes from 5 on (pre-registration [primes].start)


def real_primes(lo: int, hi: int) -> np.ndarray:
    """Primes p with max(lo, 5) <= p < hi. 1 is not prime; 2 and 3 are left out by registration."""
    return primes_between(max(lo, START), hi)

"""Layer 1, 2 and 6 of the Prime-Pattern Map: every idea enters the agent as a projection."""
from __future__ import annotations

from math import gcd

import numpy as np

UNITS_9 = (1, 2, 4, 5, 7, 8)
GATES_24 = tuple(r for r in range(24) if gcd(r, 24) == 1)
LOG2_MOD9 = {1: 0, 2: 1, 4: 2, 8: 3, 7: 4, 5: 5}


def dr(n):
    """Digital root = n mod 9 written 1..9."""
    return 1 + (n - 1) % 9


def mirror(r, M: int = 9):
    """Sign flip on the M-clock: r -> -r mod M (for M = 9 this is 9 - r)."""
    return (-r) % M if M != 9 else 9 - r


def wheel(M: int) -> tuple[int, ...]:
    """Open columns of the cylinder of circumference M."""
    return tuple(r for r in range(M) if gcd(r, M) == 1)


def gate_bits(p):
    """3-bit code of a gate mod 24; products of primes XOR their codes."""
    if isinstance(p, np.ndarray):
        m8 = p % 8
        return ((p % 3 == 2).astype(np.int64)
                | ((p % 4 == 3).astype(np.int64) << 1)
                | (((m8 == 3) | (m8 == 5)).astype(np.int64) << 2))
    return int(p % 3 == 2) | (int(p % 4 == 3) << 1) | (int(p % 8 in (3, 5)) << 2)


def joint_state(p):
    """Gate and digital root together = p mod 72 (Chinese Remainder Theorem)."""
    return p % 72


def signs_from_classes(values: np.ndarray, plus: set) -> np.ndarray:
    """Turn any projection into a +1/-1 sequence for the randomness tuple."""
    return np.where(np.isin(values, list(plus)), 1, -1).astype(np.int8)

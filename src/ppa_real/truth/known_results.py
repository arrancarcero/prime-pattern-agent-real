"""Results already in the literature. A surviving pattern that matches one of these
is logged as a REDISCOVERY, not as new. Add entries as you learn of them."""
from collections import namedtuple

Known = namedtuple("Known", "key kind test source")

KNOWN_RESULTS: tuple[Known, ...] = (
    Known("consecutive_residue_bias", "transitions",
          "repeats (diagonal) suppressed relative to a random model, any modulus, any lag",
          "Lemke Oliver & Soundararajan, PNAS 2016"),
    Known("gap_distribution", "transitions",
          "real gap frequencies differ from a random wheel model (singular series)",
          "Hardy & Littlewood 1923; Gallagher 1976"),
    Known("hasse_weil_bound", "curves",
          "|a_p| <= 2 sqrt(p) for every elliptic curve and good prime p",
          "Hasse 1933; Weil 1948; Deligne 1974 in general"),
    Known("sato_tate", "curves",
          "Frobenius angles of a non-CM elliptic curve follow (2/pi) sin^2",
          "Barnet-Lamb, Geraghty, Harris, Taylor 2011"),
    Known("chebyshev_bias", "balance",
          "slightly more primes in non-square classes (e.g. 2 mod 3, 3 mod 4)",
          "Chebyshev 1853; Rubinstein & Sarnak 1994"),
    Known("wheel_constraints", "transitions",
          "transitions forbidden or forced by small prime factors",
          "elementary (wheel factorization)"),
    Known("sub_poisson_long_windows", "fano",
          "prime counts in long windows vary less than Poisson",
          "Montgomery & Soundararajan 2004 (conditional)"),
)

RETIRED: tuple[dict, ...] = (
    {"idea": "window of exactly 24 primes correlating via the Fibonacci 24-cycle",
     "reason": "retired by the researcher 2026-11-11: mod 24 is not special"},
    {"idea": "predicting prime positions from digital roots alone",
     "reason": "replaced by: what is true of a prime's neighbours, given its position"},
)


def match_transition_residual(Z, used):
    """Does a transition residual look like the known consecutive-residue bias?
    Returns (matched: bool, diagonal_share: float)."""
    import numpy as np
    k = len(used)
    diag = np.diag(Z)
    off = Z[~np.eye(k, dtype=bool)]
    # Signature of the known bias: repeats (diagonal) fall below the model,
    # and fall further than the average other move.
    matched = bool(np.mean(diag) < 0 and np.mean(diag) < np.mean(off))
    return matched, round(float(np.mean(diag) - np.mean(off)), 2)

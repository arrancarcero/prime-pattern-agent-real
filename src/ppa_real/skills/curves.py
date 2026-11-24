"""Riemann Hypothesis for curves over finite fields (Hasse 1933, Weil 1948, Deligne 1974).

For an elliptic curve y^2 = x^3 + a x + b and a prime p of good reduction:
    #points over F_p  =  p + 1 - a_p,   with  a_p = -(sum over x of the sign of f(x))
The sign is the Legendre symbol: +1 if f(x) is a nonzero square mod p, -1 if not, 0 if f(x) = 0.

Frobenius eigenvalues: a_p = alpha + beta, alpha * beta = p. The Riemann Hypothesis for
the curve says |alpha| = |beta| = sqrt(p), i.e. alpha, beta = sqrt(p) e^(+-i theta), so
    a_p = 2 sqrt(p) cos(theta)   and   |a_p| <= 2 sqrt(p)   (never fails: a theorem).

Sato-Tate (theorem for curves without complex multiplication; Barnet-Lamb, Geraghty,
Harris, Taylor 2011): the angles theta are spread like (2/pi) sin^2(theta) on [0, pi].
A curve WITH complex multiplication, like y^2 = x^3 - x, behaves differently:
a_p = 0 for every p = 3 mod 4, and the other angles are spread evenly.
"""
import math

import numpy as np

from ppa_real.core.ledger import Finding
from ppa_real.core.sieve import small_primes
from ppa_real.skills.base import Skill

CURVES = (
    {"name": "y^2 = x^3 + x + 1", "a": 1, "b": 1, "bad": (2, 31), "cm": False},
    {"name": "y^2 = x^3 - x", "a": -1, "b": 0, "bad": (2,), "cm": True},
)


def a_p(a: int, b: int, p: int) -> int:
    x = np.arange(p, dtype=np.int64)
    f = (x * x % p * x + a * x + b) % p
    is_sq = np.zeros(p, dtype=bool)
    is_sq[(x * x) % p] = True
    signs = np.where(f == 0, 0, np.where(is_sq[f], 1, -1))
    return -int(signs.sum())


def sato_tate_z(thetas: np.ndarray, bins: int = 12) -> float:
    """Chi-square of the angle histogram against (2/pi) sin^2, turned into a z-score."""
    edges = np.linspace(0, math.pi, bins + 1)
    counts, _ = np.histogram(thetas, bins=edges)
    # integral of (2/pi) sin^2 over a bin = (1/pi)(t - sin(t)cos(t)) evaluated at the ends
    F = lambda t: (t - np.sin(t) * np.cos(t)) / math.pi
    expected = len(thetas) * (F(edges[1:]) - F(edges[:-1]))
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    return (chi2 - (bins - 1)) / math.sqrt(2 * (bins - 1))


def run(ctx, max_p=20_000, **_):
    primes = [int(p) for p in small_primes(max_p) if p > 3]
    out = []
    for c in CURVES:
        ps = [p for p in primes if p not in c["bad"]]
        ap = np.array([a_p(c["a"], c["b"], p) for p in ps], dtype=float)
        bound = 2 * np.sqrt(ps)
        ratio = np.abs(ap) / bound
        violations = int(np.sum(ratio > 1 + 1e-12))
        thetas = np.arccos(np.clip(ap / bound, -1, 1))
        out.append(Finding(
            "curves", f"Hasse/Weil bound |a_p| <= 2 sqrt(p): {c['name']}",
            "established" if violations == 0 else "check",
            f"{len(ps)} primes up to {max_p}: {violations} violations; "
            f"closest approach |a_p|/2sqrt(p) = {ratio.max():.4f} at p = {ps[int(ratio.argmax())]}",
            survived=violations == 0,
            details={"first_values": {int(p): int(v) for p, v in zip(ps[:12], ap[:12])}}))
        z = sato_tate_z(thetas)
        if c["cm"]:
            zero_share = float(np.mean([v == 0 for v, p in zip(ap, ps) if p % 4 == 3]))
            summary = (f"complex multiplication: a_p = 0 for {zero_share:.0%} of p = 3 mod 4; "
                       f"Sato-Tate fit z = {z:.1f} (expected to FAIL for a CM curve)")
            ok = zero_share == 1.0 and z > 4
        else:
            summary = f"angles fit the Sato-Tate sin^2 law: z = {z:.2f} (|z| < 4 passes)"
            ok = abs(z) < 4
        out.append(Finding("curves", f"Sato-Tate angle law: {c['name']}", "established",
                           summary, round(z, 2), survived=ok))
    return out


SKILL = Skill("curves", "Hasse-Weil bound 2 sqrt(p) and the Sato-Tate sin^2 law",
              "Do Frobenius angles of elliptic curves sit on the circle of radius sqrt(p)?",
              run, needs_primes=False)

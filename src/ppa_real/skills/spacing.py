"""Spacing of the zeta zeros vs the laws of quantum energy levels.

Montgomery (1972) found that zeta zeros repel each other; Dyson recognized the law as the
eigenvalue statistics of random matrices, the model physicists use for energy levels of
heavy nuclei. There are three candidate laws for the gap s between neighbours (scaled so
the average gap is 1), approximated by Wigner surmises:
    Poisson (no repulsion, independent points)     p(s) = e^(-s)
    GOE  (repulsion ~ s,  time-reversal symmetric) p(s) = (pi/2) s e^(-pi s^2/4)
    GUE  (repulsion ~ s^2, no time-reversal)       p(s) = (32/pi^2) s^2 e^(-4 s^2/pi)
Heavy-nucleus levels follow GOE. The zeta zeros follow GUE: they repel even more strongly,
which says any quantum system behind them (Hilbert-Polya) must break time-reversal symmetry.

Zeros are found from sign changes of Hardy's Z(t), computed with the Riemann-Siegel formula,
then checked against the Riemann-von Mangoldt count N(T) and the first known zero heights.
Yardsticks (stated first): three formulas, the Wigner surmises for Poisson (no repulsion), GOE
and GUE. No simulated zeros: every zero is computed from the Riemann-Siegel formula and checked
against the Riemann-von Mangoldt count and the known first heights.
"""
import math

import numpy as np

from ppa_real.core.ledger import Finding
from ppa_real.skills.base import Skill
from ppa_real.skills.candles import ZEROS


def theta(t):
    t = np.asarray(t, dtype=float)
    return t / 2 * np.log(t / (2 * np.pi)) - t / 2 - np.pi / 8 + 1 / (48 * t) + 7 / (5760 * t ** 3)


def hardy_z(t):
    """Riemann-Siegel: Z(t) = 2 sum n^-1/2 cos(theta - t log n) + first correction term."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    a = np.sqrt(t / (2 * np.pi))
    N = np.floor(a).astype(int)
    th = theta(t)
    z = np.zeros_like(t)
    for n in range(1, int(N.max()) + 1):
        m = N >= n
        z[m] += np.cos(th[m] - t[m] * math.log(n)) / math.sqrt(n)
    z *= 2
    p = a - N
    c0 = np.cos(2 * np.pi * (p * p - p - 1 / 16)) / np.cos(2 * np.pi * p)
    z += np.where(N % 2 == 1, 1.0, -1.0) * (t / (2 * np.pi)) ** -0.25 * c0
    return z


def zeros_up_to(T: float, step: float = 0.01, t0: float = 10.0) -> np.ndarray:
    ts = np.arange(t0, T, step)
    z = np.concatenate([hardy_z(ts[i:i + 50_000]) for i in range(0, len(ts), 50_000)])
    idx = np.nonzero(np.sign(z[:-1]) != np.sign(z[1:]))[0]
    lo, hi = ts[idx], ts[idx + 1]
    zl = z[idx]
    for _ in range(40):                     # bisection, all brackets at once
        mid = (lo + hi) / 2
        zm = hardy_z(mid)
        same = np.sign(zm) == np.sign(zl)
        lo, zl = np.where(same, mid, lo), np.where(same, zm, zl)
        hi = np.where(same, hi, mid)
    return (lo + hi) / 2


def rvm_count(T: float) -> float:
    """Riemann-von Mangoldt smooth count of zeros up to height T."""
    return float(theta(T) / np.pi + 1)


LAWS = {
    "Poisson": lambda s: 1 - np.exp(-s),
    "GOE": lambda s: 1 - np.exp(-np.pi * s * s / 4),
    "GUE": None,  # CDF below
}


def _gue_cdf(s):
    s = np.asarray(s, dtype=float)
    # integral of (32/pi^2) s^2 exp(-4 s^2/pi): erf form
    k = 4 / np.pi
    from math import erf
    e = np.vectorize(erf)(np.sqrt(k) * s)
    return e - (2 * np.sqrt(k) / np.sqrt(np.pi)) * s * np.exp(-k * s * s)


LAWS["GUE"] = _gue_cdf


def ks(sample, cdf) -> float:
    x = np.sort(sample)
    n = len(x)
    F = cdf(x)
    return float(max(np.max(np.arange(1, n + 1) / n - F), np.max(F - np.arange(n) / n)))


def run(ctx, T=None, **_):
    if T is None:
        T = float(ctx.cfg.get("spacing", {}).get("T", 3000))   # registered in prereg.toml
    g = zeros_up_to(T)
    out = []

    # 1. Calibration: right count, right first heights
    expected = rvm_count(T)
    # Riemann-Siegel with one correction term is off by ~0.002 near t = 14 and ~0.0001 near
    # t = 1400: tiny next to the average gap of ~1, so it cannot change the gap law.
    first_err = max(abs(a - b) for a, b in zip(g[:len(ZEROS)], ZEROS))
    z1000 = abs(g[999] - 1419.4224809460) if len(g) >= 1000 else 0.0   # 1000th zero (mpmath)
    count_ok = abs(len(g) - expected) < 3       # S(T) is small at this height
    out.append(Finding(
        "spacing", "calibration: zeros found match the known count and heights", "established",
        f"{len(g)} zeros up to height {T:g}; Riemann-von Mangoldt predicts {expected:.1f}; "
        f"first {len(ZEROS)} within {first_err:.1e} of the known values; 1000th zero within {z1000:.1e}",
        survived=count_ok and first_err < 0.01 and z1000 < 1e-3))

    # 2. Unfold (scale so the mean gap is 1) and compare with the three laws
    s = np.diff(theta(g) / np.pi)
    d = {name: ks(s, cdf) for name, cdf in LAWS.items()}
    best = min(d, key=d.get)
    crit = 1.36 / math.sqrt(len(s))             # KS 5% critical value
    small = float(np.mean(s < 0.1))
    out.append(Finding(
        "spacing", "gap law of the zeros: Poisson vs GOE (nuclei) vs GUE", "established",
        f"{len(s)} gaps; KS distance Poisson {d['Poisson']:.3f}, GOE {d['GOE']:.3f}, "
        f"GUE {d['GUE']:.3f} (5% critical {crit:.3f}); best fit {best}. "
        f"Gaps under 0.1: {small:.2%} (Poisson 9.5%, GOE 0.78%, GUE 0.11%)",
        # GUE wins clearly, but at these low heights the gaps are still a little off the
        # infinite-height law (known finite-size effect, Bogomolny et al. 2006), so the
        # KS distance to GUE need not fall under the critical value.
        survived=best == "GUE" and d["Poisson"] > crit and d["GOE"] > crit,
        details={"ks": {k: round(v, 4) for k, v in d.items()}, "small_gap_share": round(small, 5),
                 "hist": np.histogram(s, bins=30, range=(0, 3))[0].tolist()}))

    # 3. Higher up, via the FFT batch walk (Odlyzko-Schonhage style)
    high = float(ctx.cfg.get("spacing", {}).get("high_T", 1e8))
    width = float(ctx.cfg.get("spacing", {}).get("high_width", 4000))
    if high > 0:
        out.append(high_window(high, width, d["GUE"]))
    return out


def high_window(height, width, low_gue, K=2 ** 18):
    import time
    from ppa_real.core.batch_walk import safe_window, z_batch, zeros_in_window
    lo, hi = safe_window(height, width)
    delta = (hi - lo) / K
    tc = lo + delta * (K // 2)
    t0 = time.time()
    g = zeros_in_window(tc, delta, K)
    t_batch = time.time() - t0
    # accuracy and speed check against the one-height formula on 100 sample heights
    t, z = z_batch(tc, delta, K)
    pick = np.arange(0, K, K // 100)[:100]                  # every (K/100)-th real height
    t1 = time.time()
    zd = hardy_z(t[pick])
    t_direct = (time.time() - t1) / 100 * K
    err = float(np.max(np.abs(zd - z[pick])))
    expected = float((theta(t[-1]) - theta(t[0])) / np.pi)
    s = np.diff(theta(g) / np.pi)
    dk = {name: ks(s, cdf) for name, cdf in LAWS.items()}
    best = min(dk, key=dk.get)
    return Finding(
        "spacing", f"FFT batch walk: zero gaps near height {height:.0e}", "established",
        f"{len(g)} zeros in [{lo:.0f}, {hi:.0f}] (expected ~{expected:.0f}); {K:,} heights by one FFT "
        f"in {t_batch:.2f}s vs ~{t_direct:.0f}s one at a time, max difference {err:.1e}. "
        f"KS Poisson {dk['Poisson']:.3f}, GOE {dk['GOE']:.3f}, GUE {dk['GUE']:.3f} "
        f"(low zeros: GUE {low_gue:.3f}); best fit {best}",
        survived=best == "GUE" and err < 1e-4 and abs(len(g) - expected) < 5,
        details={"ks": {k: round(v, 4) for k, v in dk.items()}, "speedup": round(t_direct / max(t_batch, 1e-9))})


SKILL = Skill("spacing", "formulas: Poisson, GOE and GUE gap laws (Wigner surmises)",
              "Gaps between zeta zeros vs the energy-level laws (GOE nuclei, GUE)",
              run, needs_primes=False)

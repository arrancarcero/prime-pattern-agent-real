"""Standard candles: redshift and brightness of the zeta zeros in the primes.

The explicit formula writes the prime error as waves, one per zeta zero rho = beta + i gamma:
    (psi(x) - x) / sqrt(x)  ~  - sum over zeros of 2 x^(beta - 1/2) cos(gamma log x - phase) / |rho|
Read it like an astronomer reads a galaxy:
  * colour (redshift): in ordinary numbers n, a zero's wave has period ~ 2 pi x / gamma, so it
    stretches as x grows. In u = log x it is a fixed spectral line at frequency gamma.
  * brightness: after dividing by sqrt(x), a zero ON the line (beta = 1/2) shines equally at
    every distance. A zero OFF the line brightens by x^(beta - 1/2), like a supernova that
    does not dim as predicted (how dark energy was found in 1998).
So:  measured brightness far / near = e^((beta - 1/2) * du)   ->   implied beta.

Yardstick (stated first): the explicit formula with every zero at beta = 1/2, so the far/near
brightness ratio is 1. Real data only: primes from 5 on; the exact psi terms of 2, 3 and their
powers are added back as a formula. Calibration uses real data too: the lines must sit on the
known zero heights (Odlyzko's tables) and a frequency with no zero must stay dark. Sensitivity
is argued, not planted: a zero at beta changes far/near by e^((beta - 1/2) du).
Limit: this sees only zeros whose waves the data can resolve (gamma up to ~50 at x = 10^8).
It cannot prove RH: an off-line zero far higher up would be invisible at any reachable x.
"""
import math

import numpy as np

from ppa_real.core.ledger import Finding
from ppa_real.core.sieve import real_primes
from ppa_real.skills.base import Skill

# First ten zeta zero heights (Odlyzko's tables)
ZEROS = (14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
         37.586178, 40.918719, 43.327073, 48.005151, 49.773832)


def psi_curve(max_x: int, points: int = 20_000):
    """u = log x on an even grid, and f(u) = (psi(x) - x)/sqrt(x) with its mean removed."""
    P = real_primes(5, max_x).astype(np.float64)          # real primes from 5 on
    pts, w = [P], [np.log(P)]
    k = 2
    while 2.0 ** k <= max_x:
        Q = P[P ** k <= max_x]
        pts.append(Q ** k)
        w.append(np.log(Q))
        k += 1
    pts, w = np.concatenate(pts), np.concatenate(w)
    o = np.argsort(pts)
    pts, cum = pts[o], np.cumsum(w[o])
    u = np.linspace(math.log(100), math.log(max_x), points)
    x = np.exp(u)
    small = math.log(2) * np.floor(np.log(x) / math.log(2) + 1e-12) \
        + math.log(3) * np.floor(np.log(x) / math.log(3) + 1e-12)   # exact psi terms of 2, 3 and powers
    f = (cum[np.searchsorted(pts, x, side="right") - 1] + small - x) / np.sqrt(x)
    return u, f - f.mean()


def brightness(f, u, gamma) -> float:
    """Hann-windowed amplitude of the wave at frequency gamma (in log x)."""
    w = np.hanning(len(u))
    return float(abs(np.sum(w * f * np.exp(-1j * gamma * u))) / np.sum(w))


def peak_near(f, u, gamma, half_width=0.6) -> float:
    gs = np.linspace(gamma - half_width, gamma + half_width, 241)
    return float(gs[int(np.argmax([brightness(f, u, g) for g in gs]))])


def candle(f, u, gamma):
    """(near brightness, far brightness, implied beta)."""
    h = len(u) // 2
    a, b = brightness(f[:h], u[:h], gamma), brightness(f[h:], u[h:], gamma)
    du = u[h:].mean() - u[:h].mean()
    return a, b, 0.5 + math.log(b / a) / du


def run(ctx, max_x=None, **_):
    reg = ctx.cfg.get("candles", {})
    max_x = int(max_x or float(reg.get("max_x", 1e8)))
    tol = float(reg.get("tolerance", 0.03))
    nz = int(reg.get("zeros", 10))
    empty = float(reg.get("empty_frequency", 18.0))
    dark_max = float(reg.get("dark_ratio", 0.1))
    zeros = ZEROS[:nz]
    u, f = psi_curve(max_x)
    out = []

    # 1. Calibration on real data only: lines at the known heights, darkness where no zero is
    peaks = [peak_near(f, u, g) for g in zeros]
    worst = max(abs(p - g) for p, g in zip(peaks, zeros))
    dark = brightness(f, u, empty) / brightness(f, u, zeros[0])
    calibrated = worst < 0.1 and dark < dark_max
    out.append(Finding(
        "candles", "redshift: real prime spectral lines = known zeta zero heights", "established",
        f"{len(zeros)} lines within {worst:.3f} of the known heights (primes from 5 to {max_x:.0e}, "
        f"2 and 3 added back as exact formula terms); frequency {empty} (no zero) is {dark:.1%} as "
        f"bright as the first zero",
        survived=calibrated, details={"peaks": [round(p, 3) for p in peaks], "dark_ratio": dark}))

    # 2. Brightness: every candle steady (beta = 1/2)? The yardstick is the explicit formula itself:
    #    a zero at beta would scale far/near by e^((beta - 1/2) du), so the reading IS beta.
    rows = {g: candle(f, u, g) for g in zeros}
    betas = [r[2] for r in rows.values()]
    drift = max(abs(b - 0.5) for b in betas)
    h = len(u) // 2
    du = u[h:].mean() - u[:h].mean()
    out.append(Finding(
        "candles", "standard candles: implied real part of each zero", "established",
        f"implied beta for the first {len(zeros)} zeros: {min(betas):.3f} to {max(betas):.3f} "
        f"(explicit formula with every zero on the line: 0.5; registered tolerance {tol}); largest "
        f"drift {drift:.3f}. Argument for sensitivity (no planted data): over du = {du:.2f} an off-line "
        f"zero at beta = 0.5 + {tol} would change its brightness by x{math.exp(tol * du):.3f}"
        + ("" if calibrated else "  [calibration failed: verdict not trusted]"),
        survived=(drift < tol) if calibrated else None,
        details={f"{g:.2f}": {"near": round(a, 4), "far": round(b, 4), "beta": round(be, 4)}
                 for g, (a, b, be) in rows.items()}))
    return out


SKILL = Skill("candles", "explicit formula with every zero on the line: brightness far/near = 1",
              "Redshift and standard-candle test of the zeta zeros inside the primes",
              run, needs_primes=False)

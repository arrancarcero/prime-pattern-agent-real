"""FFT batch walk: the two-way walk at thousands of heights at once.

The walk skill evaluates Hardy's Z(t) one height at a time:
    Z(t) = 2 Re( e^(i theta(t)) F(t) ) + correction,   F(t) = sum_{n <= M} n^(-1/2) e^(-i t log n)
That costs M arrows per height. Near height 10^8, M ~ 4,000 arrows, and finding the zeros in a
window needs ~100,000 heights: 400 million arrow-steps.

Odlyzko and Schonhage's idea (1988): F at evenly spaced heights t_j = t_c + j*delta is a Fourier
sum, just with the frequencies log n not on a grid:
    F(t_c + j delta) = sum_n [n^(-1/2) e^(-i t_c log n)] * e^(-i j (delta log n))
A non-uniform FFT handles that: smear each arrow onto a fine even grid with a narrow Gaussian,
FFT the grid once, then undo the smearing. Cost ~ (heights + M) log(heights) instead of
heights * M. This is the gridding NUFFT of Greengard and Lee (2004), which plays the role of the
interpolation in Odlyzko's method.
"""
from __future__ import annotations

import math

import numpy as np

from ppa_real.skills.spacing import hardy_z, theta


def nufft1(x: np.ndarray, c: np.ndarray, K: int, msp: int = 14, R: int = 2) -> np.ndarray:
    """f_k = sum_n c_n e^(-i k x_n) for k = -K/2 .. K/2 - 1, points x_n anywhere (taken mod 2 pi).
    Gaussian gridding: spread, FFT on an R*K grid, then divide out the Gaussian."""
    Mr = R * K
    tau = math.pi * msp / (K * K * R * (R - 0.5))
    x = np.mod(x, 2 * math.pi)
    h = 2 * math.pi / Mr
    m0 = np.floor(x / h).astype(np.int64)
    offs = np.arange(-msp + 1, msp + 1)
    idx = m0[:, None] + offs[None, :]
    d = x[:, None] - idx * h
    grid = np.zeros(Mr, dtype=np.complex128)
    np.add.at(grid, idx % Mr, c[:, None] * np.exp(-d * d / (4 * tau)))
    G = np.fft.fft(grid) / Mr                      # G[k] = (1/Mr) sum_m grid_m e^(-i k xi_m)
    k = np.arange(-K // 2, K // 2)
    return math.sqrt(math.pi / tau) * np.exp(k * k * tau) * G[k % Mr]


def z_batch(t_center: float, delta: float, K: int) -> tuple[np.ndarray, np.ndarray]:
    """Hardy Z(t) at t_j = t_center + j*delta, j = -K/2..K/2-1, via one NUFFT.
    M is fixed at its value at the window's low end; a window of a few thousand at large t
    keeps that exact (M changes only when sqrt(t / 2 pi) crosses an integer, checked here)."""
    t = t_center + delta * np.arange(-K // 2, K // 2)
    M_lo = int(math.floor(math.sqrt(t[0] / (2 * math.pi))))
    M_hi = int(math.floor(math.sqrt(t[-1] / (2 * math.pi))))
    if M_lo != M_hi:
        raise ValueError("window crosses a Riemann-Siegel M boundary; shrink or move it")
    n = np.arange(1, M_lo + 1, dtype=np.float64)
    logn = np.log(n)
    c = n ** -0.5 * np.exp(-1j * t_center * logn)
    F = nufft1(delta * logn, c, K)
    main = 2 * np.real(np.exp(1j * theta(t)) * F)
    a = np.sqrt(t / (2 * math.pi))
    p = a - M_lo
    c0 = np.cos(2 * math.pi * (p * p - p - 1 / 16)) / np.cos(2 * math.pi * p)
    corr = (1.0 if M_lo % 2 == 1 else -1.0) * (t / (2 * math.pi)) ** -0.25 * c0
    return t, main + corr


def zeros_in_window(t_center: float, delta: float, K: int) -> np.ndarray:
    """Sign changes of the batch Z, each placed by linear interpolation between grid points."""
    t, z = z_batch(t_center, delta, K)
    i = np.nonzero(np.sign(z[:-1]) != np.sign(z[1:]))[0]
    a, b, za, zb = t[i], t[i + 1], z[i], z[i + 1]
    g = a - za * (b - a) / (zb - za)
    return g


def safe_window(height: float, width: float) -> tuple[float, float]:
    """Move a window up so it does not cross a point where sqrt(t / 2 pi) is an integer."""
    lo = height
    m = math.floor(math.sqrt(lo / (2 * math.pi)))
    nxt = 2 * math.pi * (m + 1) ** 2
    if lo + width >= nxt:
        lo = nxt + 1.0
    return lo, lo + width

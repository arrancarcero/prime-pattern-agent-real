"""Shared statistics: transition tables, residual z-scores, holdout scoring, Fano."""
from __future__ import annotations

import math

import numpy as np

# Montgomery-Soundararajan second-order constant: B = 1 - Euler's constant - log(2 pi)
B_MS = 1 - 0.5772156649015329 - math.log(2 * math.pi)


def transition_table(P: np.ndarray, M: int, lag: int = 1) -> tuple[np.ndarray, list[int]]:
    """Counts of (p_n mod M) -> (p_{n+lag} mod M), restricted to classes primes use."""
    a = P[:-lag] % M
    b = P[lag:] % M
    T = np.bincount(a * M + b, minlength=M * M).reshape(M, M).astype(np.float64)
    used = [r for r in range(M) if T[r].sum() > 0.001 * T.sum() / M]
    return T[np.ix_(used, used)], used


def residual_z(T_obs: np.ndarray, T_model: np.ndarray) -> np.ndarray:
    """Compare observed transitions with a model, row by row (row totals matched)."""
    n_obs, n_mod = T_obs.sum(), T_model.sum()
    rows = T_model.sum(axis=1, keepdims=True)
    P_mod = np.divide(T_model, rows, out=np.zeros_like(T_model), where=rows > 0)
    E = P_mod * T_obs.sum(axis=1, keepdims=True)
    var = np.maximum(E * (1 + n_obs / max(n_mod, 1)), 1.0)
    return (T_obs - E) / np.sqrt(var)


def holdout_score(Z_train: np.ndarray, Z_test: np.ndarray) -> tuple[float, float]:
    """Freeze the sign pattern from train; measure how strongly it reappears on test."""
    frozen = np.sign(Z_train)
    score = float(np.sum(frozen * Z_test) / math.sqrt(frozen.size))
    if Z_train.std() == 0 or Z_test.std() == 0:
        return score, 0.0
    corr = float(np.corrcoef(Z_train.ravel(), Z_test.ravel())[0, 1])
    return score, corr


def fano_curve(P: np.ndarray, lo: int, hi: int, Hs=(100, 1_000, 10_000, 100_000)) -> list[dict]:
    """Fano factor of prime counts in windows (trend removed) and the reference
    log(N/H)/log(N) from the Montgomery-Soundararajan variance conjecture."""
    N = (lo + hi) / 2
    rows = []
    for H in Hs:
        nbins = (hi - lo) // H
        if nbins < 10:
            continue
        sel = P[(P >= lo) & (P < lo + nbins * H)]
        counts = np.bincount((sel - lo) // H, minlength=nbins)
        mids = lo + H * (np.arange(nbins) + 0.5)
        expected = H / np.log(mids)
        F = float(np.mean((counts - expected) ** 2) / counts.mean())
        L = math.log(N)
        lead = math.log(N / H) / L
        rows.append({"H": H, "fano": round(F, 4),
                     "reference": round(lead, 4),
                     "reference_with_B": round(lead + B_MS / L, 4),
                     # what this window says the constant B is
                     "B_estimate": round((F - lead) * L, 3)})
    return rows

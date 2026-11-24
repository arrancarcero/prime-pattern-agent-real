"""Fano factor of prime counts in windows of length H.

Reference (Montgomery-Soundararajan, conditional on Hardy-Littlewood):
    F(H) ~ (log(N/H) + B) / log N,   B = 1 - Euler's constant - log(2 pi) = -1.4151
Each window size gives its own estimate of B. The middle windows (H = 10^3, 10^4)
are the fair ones: the formula is meant for H neither tiny nor a large share of N,
and we count primes one each instead of weighting by log p.
"""
import statistics

from ppa_real.core.ledger import Finding
from ppa_real.core.stats import B_MS, fano_curve
from ppa_real.skills.base import Skill
from ppa_real.truth import truth_tuple

FAIR_WINDOWS = (1_000, 10_000)


def run(ctx, **_):
    train = fano_curve(ctx.P_train, *ctx.train)
    test = fano_curve(ctx.P_test, *ctx.test)
    fair = [r["B_estimate"] for r in train + test if r["H"] in FAIR_WINDOWS]
    B_hat = statistics.mean(fair)
    spread = (max(fair) - min(fair)) / 2
    v = truth_tuple.verify("ms_constant_B", None, B_hat)
    below = all(r["fano"] < 1 for r in test)
    falling = all(a["fano"] > b["fano"] for a, b in zip(test, test[1:]))
    agrees = v.ok is not None
    return [Finding(
        "fano", "Fano factor and the second-order constant B", "conjecture",
        f"measured B = {B_hat:.3f} +/- {spread:.3f} (predicted {B_MS:.4f}); "
        f"{'agrees' if agrees else 'outside tolerance'}; below 1: {below}; falls with H: {falling}",
        survived=below and falling and agrees,
        details={"B_measured": round(B_hat, 4), "B_spread": round(spread, 4),
                 "B_predicted": round(B_MS, 4), "fair_windows": list(FAIR_WINDOWS),
                 "train": train, "test": test})]


SKILL = Skill("fano", "Poisson (F = 1) and the two-term Montgomery-Soundararajan curve",
              "Do primes count out more evenly than random events, by the predicted constant B?",
              run)

import numpy as np

from ppa_real.core.ledger import Finding
from ppa_real.skills.base import Skill

WHEELS = (2, 3, 4, 5, 6, 10, 12, 15, 30, 210)


def run(ctx, length=2 ** 20, top=10, **_):
    lo = ctx.test[0]
    x = np.zeros(length)
    sel = ctx.primes.window(lo, min(lo + length, ctx.test[1])) - lo
    x[sel] = 1.0
    x -= x.mean()
    S = np.abs(np.fft.rfft(x)) ** 2
    S[0] = 0
    f = np.fft.rfftfreq(length)
    peaks, unexplained = [], []
    for i in np.argsort(S)[::-1][:top]:
        tol = 2.0 / length
        w = next((M for M in WHEELS if abs(f[i] * M - round(f[i] * M)) < M * tol), None)
        peaks.append({"period": round(1 / f[i], 3), "wheel": w})
        if w is None:
            unexplained.append(round(1 / f[i], 3))
    label = "candidate" if unexplained else "established"
    return [Finding("spectrum", "FFT peaks of the prime indicator", label,
                    "all top peaks are wheel periods" if not unexplained
                    else f"unexplained periods: {unexplained}",
                    survived=not unexplained, details={"peaks": peaks})]


SKILL = Skill("spectrum", "wheel periods (divisors of 210 and small products)",
              "Does the prime indicator hold any period the wheel does not explain?", run)

import numpy as np

from ppa_real.core.ledger import Finding
from ppa_real.core.projections import gate_bits
from ppa_real.skills.base import Skill


def run(ctx, **_):
    from ppa_real.core.primestore import to_numpy
    cnt = np.zeros(24, dtype=np.int64)
    first = None
    for P in ctx.primes.blocks(*ctx.test):
        cnt += to_numpy(ctx.primes.xp.bincount(P % 24, minlength=24))
        if first is None and len(P) > 2001:
            first = to_numpy(P[:2001])
    g = np.nonzero(cnt)[0]
    c = cnt[g]
    P = first
    spread = float((c.max() - c.min()) / c.mean())
    a, b = P[:2000], P[1:2001]
    # reduce before multiplying: primes near 1e10 overflow 64-bit products (fixed 2026-11-24)
    xor_ok = bool(np.all(gate_bits((a % 24) * (b % 24) % 24) == (gate_bits(a % 24) ^ gate_bits(b % 24))))
    return [Finding("gates", "gate occupancy and 3-bit XOR rule", "established",
                    f"8 gates, max-min spread {spread:.4%}; XOR rule holds: {xor_ok}",
                    survived=xor_ok and spread < 0.01,
                    details={"counts": {int(k): int(v) for k, v in zip(g, c)}})]


SKILL = Skill("gates", "equal share per gate (prime number theorem in progressions)",
              "Are the eight gates evenly used, and does the XOR code hold?", run)

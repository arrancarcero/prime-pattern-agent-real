import numpy as np

from ppa_real.core.ledger import Finding
from ppa_real.core.projections import gate_bits
from ppa_real.skills.base import Skill


def run(ctx, **_):
    P = ctx.P_test[ctx.P_test > 3]
    g, c = np.unique(P % 24, return_counts=True)
    spread = float((c.max() - c.min()) / c.mean())
    a, b = P[:2000], P[1:2001]
    xor_ok = bool(np.all(gate_bits((a * b) % 24) == (gate_bits(a % 24) ^ gate_bits(b % 24))))
    return [Finding("gates", "gate occupancy and 3-bit XOR rule", "established",
                    f"8 gates, max-min spread {spread:.4%}; XOR rule holds: {xor_ok}",
                    survived=xor_ok and spread < 0.01,
                    details={"counts": {int(k): int(v) for k, v in zip(g, c)}})]


SKILL = Skill("gates", "equal share per gate (prime number theorem in progressions)",
              "Are the eight gates evenly used, and does the XOR code hold?", run)

"""Digital-root transitions compared with the gap rule (gap distribution given p mod 6)."""
import numpy as np

from ppa_real.core.ledger import Finding
from ppa_real.core.projections import UNITS_9, dr
from ppa_real.core.stats import holdout_score, residual_z
from ppa_real.skills.base import Skill


def _tables(P):
    P = P[P > 3]
    a, b = dr(P[:-1]), dr(P[1:])
    idx = {r: i for i, r in enumerate(UNITS_9)}
    T = np.zeros((6, 6))
    np.add.at(T, (np.vectorize(idx.get)(a), np.vectorize(idx.get)(b)), 1)
    g, base = np.diff(P), P[:-1]
    N = np.zeros((6, 6))
    for cls in (1, 5):
        m = base % 6 == cls
        gaps, cnt = np.unique(g[m], return_counts=True)
        pg = cnt / cnt.sum()
        for r in (r for r in UNITS_9 if r % 3 == cls % 3):
            n_r = np.sum(m & (dr(base) == r))
            for gap, w in zip(gaps, pg):
                nxt = int(dr(r + gap))
                if nxt in idx:
                    N[idx[r], idx[nxt]] += n_r * w
    return T, N


def run(ctx, **_):
    th = ctx.cfg["thresholds"]
    T1, N1 = _tables(ctx.P_train)
    T2, N2 = _tables(ctx.P_test)
    z, corr = holdout_score(residual_z(T1, N1), residual_z(T2, N2))
    survived = z > th["z_survive"] and corr > th["corr_survive"]
    raw = T2 / T2.sum(axis=1, keepdims=True)
    return [Finding("dr_gap", "digital-root transitions beyond the gap rule",
                    "candidate" if survived else "not-supported",
                    "raw bias is the gap rule (Layer 3.1); only the leftover counts",
                    round(z, 2), round(corr, 3), survived,
                    details={"raw_shares_test": np.round(raw, 4).tolist()})]


SKILL = Skill("dr_gap", "gap distribution conditioned on p mod 6",
              "Is there digital-root structure beyond what gap sizes force?", run)

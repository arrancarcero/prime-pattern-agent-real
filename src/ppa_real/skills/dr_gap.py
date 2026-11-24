"""Digital-root transitions compared with the gap rule (gap distribution given p mod 6)."""
import numpy as np

from ppa_real.core.ledger import Finding
from ppa_real.core.projections import UNITS_9, dr
from ppa_real.core.stats import holdout_score, residual_z
from ppa_real.skills.base import Skill


GMAX = 4096          # prime gaps below 1e12 are under 600


def _tables(store, lo, hi):
    """One pass: dr transition counts, gap counts per class mod 6, and (class, dr) counts."""
    from ppa_real.core.primestore import to_numpy
    lut = np.full(10, 0, dtype=np.int64)
    for i, r in enumerate(UNITS_9):
        lut[r] = i
    T = np.zeros(36, dtype=np.int64)
    G = {1: np.zeros(GMAX, dtype=np.int64), 5: np.zeros(GMAX, dtype=np.int64)}
    nr = np.zeros(6 * 10, dtype=np.int64)
    for P, _ in store.blocks_with_carry(lo, hi, carry=1):
        xp = store.xp
        L = xp.asarray(lut)
        base, nxt = P[:-1], P[1:]
        a, b = dr(base), dr(nxt)
        T += to_numpy(xp.bincount(L[a] * 6 + L[b], minlength=36))
        g = nxt - base
        c6 = base % 6
        for cls in (1, 5):
            G[cls] += to_numpy(xp.bincount(g[c6 == cls], minlength=GMAX))[:GMAX]
        nr += to_numpy(xp.bincount(c6 * 10 + a, minlength=60))
    T = T.reshape(6, 6).astype(np.float64)
    idx = {r: i for i, r in enumerate(UNITS_9)}
    N = np.zeros((6, 6))
    for cls in (1, 5):
        gaps = np.nonzero(G[cls])[0]
        cnt = G[cls][gaps]
        pg = cnt / cnt.sum()
        for r in (r for r in UNITS_9 if r % 3 == cls % 3):
            n_r = nr[cls * 10 + r]
            for gap, w in zip(gaps, pg):
                nxt = int(dr(r + gap))
                if nxt in idx:
                    N[idx[r], idx[nxt]] += n_r * w
    return T, N


def run(ctx, **_):
    th = ctx.cfg["thresholds"]
    T1, N1 = _tables(ctx.primes, *ctx.train)
    T2, N2 = _tables(ctx.primes, *ctx.test)
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

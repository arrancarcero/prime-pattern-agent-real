"""Twin primes, judged only by real primes and the Hardy-Littlewood formula.

No simulated twins. Every yardstick is a formula or a lemma checked on every twin:

1. Count. pi_2(x) against Hardy-Littlewood: 2 C2 * integral_2^x dt / log^2 t.
2. Pair correlation. For each h = 6, 12, ..., 180: how many lower twins p have p + h also a
   lower twin? Hardy-Littlewood for the 4-tuple {0, 2, h, h+2} predicts
       S4(h) * integral dt / (log t log(t+2) log(t+h) log(t+h+2)),
       S4(h) = prod_p (1 - nu_p(h)/p) / (1 - 1/p)^4,  nu_p(h) = #{0, 2, h, h+2 mod p}.
   This is the exact form of 'tiny gaps': every short-range pattern lives in these counts.
3. Digital-root repeat, as an argument chain:
   Lemma 1 (proved): every lower twin p >= 5 is 5 mod 6 (p, p+2 avoid 2 and 3).
   Lemma 2 (proved): dr(p) = dr(q)  <=>  p = q mod 9.
   Reduction: two lower twins share a digital root <=> their distance is 0 mod 18.
   Prediction (Hardy-Littlewood): among twin pairs at distance h <= 180, the share with
   h = 0 mod 18 is sum_{18 | h} S4(h) I(h) / sum_h S4(h) I(h). Checked against real pairs.
   Open step: consecutive twins need 'no twin in between', which Hardy-Littlewood pair
   counts alone do not give; the consecutive share is reported, measured only.
4. Fano per block. With density d and r(h) = S4(h) / S2^2, a block of length L has
       Fano = 1 - d + (2d / L) * sum_{h=1}^{L-1} (L - h) (r(h) - 1)
   (r(h) = 0 where the 4-tuple is impossible). Measured as the variance of
   (count - Hardy-Littlewood expected count) over the mean expected count, within each 1e7
   stretch after the first, averaged; z from the spread across stretches.
Lemmas are checked on every twin (a single failure would refute them).
"""
import math

import numpy as np

from ppa_real.core.ledger import Finding
from ppa_real.core.sieve import real_primes, small_primes
from ppa_real.skills.base import Skill

C2 = 0.6601618158468696
S2 = 2 * C2                                      # singular series of {0, 2}


def lower_twins(max_x: int) -> np.ndarray:
    P = real_primes(5, max_x)
    return P[:-1][np.diff(P) == 2]


def r_of_h(h: np.ndarray, p_max: int = 10 ** 5) -> np.ndarray:
    """r(h) = S4(h) / S2^2 for integer h >= 1, products over primes up to p_max."""
    h = np.asarray(h, dtype=np.int64)
    logr = np.where(h % 2 == 0, math.log(2.0), -np.inf)          # p = 2: (1/2)/(1/2)^2
    for p in small_primes(p_max)[1:]:
        p = int(p)
        hm = h % p
        b = 2 % p
        c = hm
        d = (hm + 2) % p
        nu = 1 + (b != 0) + ((c != 0) & (c != b)) + ((d != 0) & (d != b) & (d != c))
        f = 1 - nu / p
        logr = logr + np.where(f > 0, np.log(np.maximum(f, 1e-300)), -np.inf) - 2 * math.log(1 - 2 / p)
    return np.exp(logr)


def _I4(h, x):
    from scipy.integrate import quad
    return quad(lambda t: 1 / (math.log(t) * math.log(t + 2) * math.log(t + h) * math.log(t + h + 2)),
                5, x - h - 2, limit=500)[0]


def run(ctx, max_x=None, **_):
    reg = ctx.cfg.get("twins", {})
    th = ctx.cfg.get("thresholds", {"formula_ratio": 0.02, "formula_z": 4.0})
    max_x = int(max_x or ctx.cfg.get("range", {}).get("max_n", 10 ** 8))
    H = int(reg.get("pair_h_max", 180))
    L = int(reg.get("fano_block", 30030))
    stretch = int(float(reg.get("fano_stretch", 1e7)))
    p_max = int(float(reg.get("singular_series_primes", 1e5)))
    from scipy.integrate import quad
    T = lower_twins(max_x)
    out = []

    # 1. count
    pred = S2 * quad(lambda t: 1 / math.log(t) ** 2, 2, max_x, limit=500)[0]
    n = len(T) + 1                                   # + (3, 5): the theorem counts it; the data starts at 5
    ratio = n / pred
    out.append(Finding("twins", f"twin prime count to {max_x:.0e} vs Hardy-Littlewood", "established",
                       f"{n:,} twin pairs (data from 5 on, plus (3, 5) for the formula); predicted "
                       f"{pred:,.0f}; ratio {ratio:.5f} (registered tolerance {th['formula_ratio']})",
                       survived=abs(ratio - 1) < th["formula_ratio"]))

    # 2. pair correlation h = 6 .. H
    hs = np.arange(6, H + 1, 6)
    rh = r_of_h(hs, p_max)
    obs = np.array([int(np.isin(T + h, T).sum()) for h in hs])
    prd = np.array([S2 * S2 * r * _I4(int(h), max_x) for h, r in zip(hs, rh)])
    z = (obs - prd) / np.sqrt(prd)
    agg = obs.sum() / prd.sum()
    ok2 = np.max(np.abs(z)) <= th["formula_z"] and abs(agg - 1) <= th["formula_ratio"]
    worst = int(np.argmax(np.abs(z)))
    out.append(Finding(
        "twins", f"pair correlation of twins, h = 6..{H}, vs Hardy-Littlewood 4-tuples", "established",
        f"{obs.sum():,} twin pairs within {H}; predicted {prd.sum():,.0f} (ratio {agg:.4f}); largest "
        f"|z| = {abs(z[worst]):.1f} at h = {hs[worst]} ({obs[worst]:,} vs {prd[worst]:,.0f}). "
        + ("Matches the formula: short gaps between twins follow Hardy-Littlewood."
           if ok2 else "Does NOT match the registered tolerance: check by hand."),
        survived=bool(ok2),
        details={"h": hs.tolist(), "observed": obs.tolist(), "predicted": np.round(prd, 1).tolist(),
                 "z": np.round(z, 2).tolist(), "r": np.round(rh, 4).tolist()}))

    # 3. digital-root repeat: lemmas, reduction, prediction, open step
    lemma1 = bool(np.all(T % 6 == 5))
    dr = 1 + (T - 1) % 9
    g = np.diff(T)
    same = dr[1:] == dr[:-1]
    reduction = bool(np.array_equal(same, g % 18 == 0))
    m18 = hs % 18 == 0
    share_pred = float(prd[m18].sum() / prd.sum())
    share_obs = float(obs[m18].sum() / obs.sum())
    se = math.sqrt(share_pred * (1 - share_pred) / obs.sum())
    z3 = (share_obs - share_pred) / se
    consec = float(same.mean())
    ok3 = lemma1 and reduction and abs(z3) <= th["formula_z"]
    out.append(Finding(
        "twins", "digital-root repeat between twins: argument chain", "established",
        f"Lemma 1 (every lower twin is 5 mod 6): {'holds' if lemma1 else 'FAILS'} on all {len(T):,}. "
        f"Lemma 2 + reduction (same dr <=> gap = 0 mod 18): {'holds' if reduction else 'FAILS'} on all "
        f"{len(g):,} consecutive gaps. Prediction (Hardy-Littlewood): {share_pred:.4f} of twin pairs "
        f"within {H} sit at h = 0 mod 18; real {share_obs:.4f} (z = {z3:+.1f}). Open step: consecutive "
        f"twins repeat their dr {consec:.4f} of the time (1/3 with no structure), measured only: "
        f"'no twin in between' needs more than pair counts.",
        survived=bool(ok3),
        details={"lemma1": lemma1, "reduction": reduction, "share_pred": share_pred,
                 "share_obs": share_obs, "z": z3, "consecutive_repeat": consec}))

    # 4. Fano per block, predicted from r(h)
    hh = np.arange(1, L)
    rr = r_of_h(hh, p_max)
    corr_sum = float(np.sum((L - hh) * (rr - 1)))
    blocks = np.bincount((T // L).astype(np.int64))
    nb = max_x // L
    blocks = np.pad(blocks, (0, max(0, nb - len(blocks))))[:nb].astype(float)
    centers = (np.arange(nb) + 0.5) * L
    expected = L * S2 / (np.log(centers) * np.log(centers + 2))
    per = stretch // L
    meas, predf = [], []
    for i in range(1, nb // per):
        sl = slice(i * per, (i + 1) * per)
        e = expected[sl]
        meas.append(float(np.var(blocks[sl] - e) / e.mean()))
        dloc = e.mean() / L
        predf.append(1 - dloc + 2 * dloc / L * corr_sum)
    if len(meas) >= 2:
        m, pf = float(np.mean(meas)), float(np.mean(predf))
        se4 = float(np.std(meas, ddof=1) / math.sqrt(len(meas)))
        z4 = (m - pf) / se4
        ok4 = abs(z4) <= th["formula_z"]
        msg = (f"measured {m:.4f} over {len(meas)} stretches; predicted from the Hardy-Littlewood pair "
               f"correlation {pf:.4f} (z = {z4:+.1f}); no structure would give 1. "
               + ("Matches: the regularity is what Hardy-Littlewood predicts." if ok4
                  else "Does NOT match the registered tolerance: check by hand."))
    else:
        m = pf = z4 = None
        ok4 = None
        msg = f"range too small: needs at least 3 stretches of {stretch:.0e}"
    out.append(Finding("twins", f"Fano factor of twins per {L}-block vs Hardy-Littlewood", "established",
                       msg, survived=ok4, details={"measured": m, "predicted": pf, "z": z4}))

    # 5. measured-only consecutive gaps, with the argument chain
    stretch_idx = (T[1:] // stretch).astype(np.int64)
    mean_gap = np.bincount(stretch_idx, g) / np.maximum(np.bincount(stretch_idx), 1)
    s = g / mean_gap[stretch_idx]
    tiny, huge = float(np.mean(s < 0.25)), float(np.mean(s > 3))
    out.append(Finding(
        "twins", "consecutive twin gaps (measured only)", "measured",
        f"tiny (< 0.25 x average) {tiny:.4f}, huge (> 3 x average) {huge:.4f}; with no structure "
        f"{1 - math.exp(-0.25):.4f} and {math.exp(-3):.4f}. Reduction: a tiny consecutive gap is a "
        f"twin pair at short distance with no twin between; the short-distance counts are tested "
        f"above against Hardy-Littlewood. Open step: the 'none between' condition.",
        survived=None, details={"tiny": tiny, "huge": huge}))
    return out


SKILL = Skill("twins", "Hardy-Littlewood formulas (count, 4-tuple pair correlation, Fano) and proved lemmas",
              "Twin primes from 5 on: count, pair correlation, dr argument chain, Fano, consecutive gaps",
              run, needs_primes=False)

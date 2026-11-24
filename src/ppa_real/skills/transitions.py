"""Transitions p_n mod M -> p_{n+lag} mod M, in two stages.

Real primes only (from 5 on); both yardsticks are built from the same real primes.
Stage A, yardstick = independence: expected count a -> b = (row total of a) x (share of b
  among all real transitions). No simulated primes. Real primes are expected to beat this:
  that is the known consecutive-residue bias (Lemke Oliver & Soundararajan 2016).
Stage B, null = the gap rule: the next class is this class plus the gap, with the gap
  distribution taken from the coarser class p mod C, where C = gcd(M, 210).
  If this survives, the finer residue mod M carries information about the gap that
  the coarse class does not. That would be new structure, so it is the real test.
  When C = M, stage B is a tautology (the gap rule explains everything by construction)
  and is skipped.
"""
from math import gcd

import numpy as np

from ppa_real.core.ledger import Finding
from ppa_real.core.stats import holdout_score, residual_z, transition_table
from ppa_real.skills.base import Skill
from ppa_real.truth.known_results import match_transition_residual


def _vs_independence(P, M, lag):
    """Real transitions against independence built from the same real primes."""
    T, used = transition_table(P, M, lag)
    E = np.outer(T.sum(axis=1), T.sum(axis=0)) / T.sum()
    return residual_z(T, E), used, T


def _gap_rule(P, M, lag, used):
    C = gcd(M, 210)
    d = P[lag:] - P[:-lag]
    base = P[:-lag]
    E = np.zeros((len(used), len(used)))
    pos = {r: i for i, r in enumerate(used)}
    for c in {r % C for r in used}:
        m = base % C == c
        h = np.bincount(d[m] % M, minlength=M).astype(float)
        h /= h.sum()
        for a in (r for r in used if r % C == c):
            n_a = np.count_nonzero(base % M == a)
            for b in used:
                E[pos[a], pos[b]] += n_a * h[(b - a) % M]
    return E


def run(ctx, M=9, lag=1, **_):
    th = ctx.cfg["thresholds"]
    name = f"transitions mod {M}, lag {lag}"
    params = {"M": M, "lag": lag}
    ZA_tr, used, T_tr = _vs_independence(ctx.P_train, M, lag)
    ZA_te, used_te, T_te = _vs_independence(ctx.P_test, M, lag)
    if used != used_te or len(used) < 2:
        return [Finding("transitions", name, "check", "classes differ between train and test",
                        params=params)]
    zA, cA = holdout_score(ZA_tr, ZA_te)
    survA = zA > th["z_survive"] and cA > th["corr_survive"]
    known, diag_gap = match_transition_residual(ZA_te, used)

    tautology = gcd(M, 210) == M
    zB = cB = None
    survB = False
    if not tautology:
        ZB_tr = residual_z(T_tr, _gap_rule(ctx.P_train, M, lag, used))
        ZB_te = residual_z(T_te, _gap_rule(ctx.P_test, M, lag, used))
        zB, cB = holdout_score(ZB_tr, ZB_te)
        survB = zB > th["z_survive"] and cB > th["corr_survive"]

    if survB:
        label, summary = "candidate", (f"mod {M} carries gap information beyond mod "
                                       f"{gcd(M, 210)}: check by hand")
    elif survA and not tautology:
        label = "rediscovery"
        summary = ("beats independence, but the real gap distribution explains all of it "
                   "(Hardy-Littlewood gap structure"
                   + ("; repeat-suppression shape, Lemke Oliver & Soundararajan 2016)" if known
                      else ")"))
    elif survA and known:
        label = "rediscovery"
        summary = ("beats independence with the known repeat-suppression shape "
                   "(Lemke Oliver & Soundararajan 2016); "
                   + ("gap rule explains it by construction" if tautology
                      else "gap rule explains the rest"))
    elif survA:
        label, summary = "candidate", "beats independence with an unfamiliar shape: check by hand"
    else:
        label, summary = "not-supported", "consistent with independence"

    mi = [used.index((-r) % M) for r in used]
    return [Finding("transitions", name, label, summary,
                    round(zB if zB is not None else zA, 2),
                    round(cB if cB is not None else cA, 3),
                    survB if not tautology else survA, params=params,
                    details={"stageA_vs_independence": {"z": round(zA, 1), "corr": round(cA, 3),
                                                  "survived": survA, "known_shape": known,
                                                  "diag_minus_offdiag": diag_gap},
                             "stageB_vs_gap_rule": None if tautology else
                             {"z": round(zB, 2), "corr": round(cB, 3), "survived": survB,
                              "coarse_class": gcd(M, 210)},
                             "classes": used,
                             "mirror_reflect_corr": round(float(np.corrcoef(
                                 ZA_te.ravel(), ZA_te[np.ix_(mi, mi)].ravel())[0, 1]), 3),
                             "mirror_reverse_corr": round(float(np.corrcoef(
                                 ZA_te.ravel(), ZA_te.T.ravel())[0, 1]), 3),
                             "residual_z_test": np.round(ZA_te, 1).tolist()})]


SKILL = Skill("transitions", "stage A: independence from the same real primes; stage B: real gap rule",
              "Do consecutive primes carry residue information that gap sizes do not explain?",
              run)

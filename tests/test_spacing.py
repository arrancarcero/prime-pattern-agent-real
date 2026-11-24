from ppa_real.skills.base import Context
from ppa_real.skills.spacing import SKILL, hardy_z, zeros_up_to


def test_first_zero_and_sign_change():
    g = zeros_up_to(60)
    assert abs(g[0] - 14.1347) < 0.01 and len(g) == 13     # 13 zeros below height 60
    assert hardy_z(14.0)[0] * hardy_z(14.3)[0] < 0


def test_spacing_prefers_gue():
    cal, law, high = SKILL.run(Context({}, (0, 0), (0, 0), None, None), T=2000)
    assert cal.survived, cal.summary
    assert law.survived and law.details["ks"]["GUE"] < law.details["ks"]["GOE"], law.summary
    assert high.survived, high.summary                     # FFT batch walk near 1e8


def test_batch_walk_matches_direct():
    import numpy as np
    from ppa_real.core.batch_walk import safe_window, z_batch
    lo, hi = safe_window(1e7, 100)
    K = 2 ** 12
    delta = (hi - lo) / K
    t, z = z_batch(lo + delta * K // 2, delta, K)
    pick = np.arange(0, K, 97)
    assert np.max(np.abs(hardy_z(t[pick]) - z[pick])) < 1e-5

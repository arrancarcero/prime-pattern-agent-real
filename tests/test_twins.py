import numpy as np

from ppa_real.skills.base import Context
from ppa_real.skills.twins import SKILL, lower_twins, r_of_h


def test_lemmas_and_counts():
    T = lower_twins(10 ** 6)
    assert T[0] == 5 and len(T) + 1 == 8169            # + (3, 5) = pi_2(1e6)
    assert np.all(T[1:] % 6 == 5)                       # lemma 1 (5 itself is 5 mod 6 too)
    dr = 1 + (T - 1) % 9
    assert np.array_equal(dr[1:] == dr[:-1], np.diff(T) % 18 == 0)   # lemma 2 + reduction


def test_singular_series():
    r = r_of_h(np.array([2, 4, 6, 8, 30]))
    assert r[0] == 0 and r[1] == 0 and r[3] == 0        # impossible 4-tuples
    assert r[2] > 2 and r[4] > r[2]                     # 30 = 2*3*5 is favoured
    assert abs(r_of_h(np.arange(1, 30031)).mean() - 1) < 0.01   # Gallagher: average 1


def test_twins_skill_small():
    out = SKILL.run(Context({"thresholds": {"formula_ratio": 0.02, "formula_z": 4.0},
                             "twins": {"fano_stretch": 1e6}}, (0, 0), (0, 0), None, None),
                    max_x=10 ** 7)
    assert out[0].survived, out[0].summary
    assert out[2].details["lemma1"] and out[2].details["reduction"]

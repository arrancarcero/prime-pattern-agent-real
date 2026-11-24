from ppa_real.skills.base import Context
from ppa_real.skills.walk import SKILL, eta_averaged, eta_stops, two_way


def test_walks():
    home, arrows, cross = SKILL.run(Context({}, (0, 0), (0, 0), None, None), N=100_000)
    assert home.survived, home.summary
    assert arrows.survived, arrows.summary
    assert cross.survived, cross.summary


def test_mirror_walk_lands_on_real_line():
    M, path = two_way(236.5242296658)
    assert M == 6 and abs(path[-1].imag) < 1e-12
    assert abs(eta_averaged(eta_stops(14.134725142, 200_000))[-1]) < 1e-5

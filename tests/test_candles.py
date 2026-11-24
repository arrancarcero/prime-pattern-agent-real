from ppa_real.skills.base import Context
from ppa_real.skills.candles import SKILL


def test_candles_real_only():
    cal, bright = SKILL.run(Context({}, (0, 0), (0, 0), None, None), max_x=10 ** 7)
    assert cal.survived, cal.summary          # lines on the known heights, empty frequency dark
    assert bright.survived, bright.summary    # every zero reads beta ~ 1/2

import shutil

import pytest

from ppa_real.agent.loop import run_agent
from ppa_real.prereg import RegistrationError, load, set_range


def _reg(tmp_path):
    shutil.copy("prereg.toml", tmp_path / "prereg.toml")
    reg = load(tmp_path / "prereg.toml")
    set_range(reg, 1e7)
    reg["output"]["runs_dir"] = str(tmp_path)
    return reg


def test_agent_runs_and_records_registration(tmp_path):
    reg = _reg(tmp_path)
    reg["planner"]["max_steps"] = 6
    reg["graveyard"] = {"enforce": False}      # test the planner itself, not the graveyard
    ledger = run_agent(reg, "explore")
    assert ledger.findings[0].skill == "calibrate" and ledger.findings[0].survived
    assert len(ledger.findings) >= 4
    assert ledger.meta["prereg_fingerprint"] == reg["_fingerprint"]


def test_tests_cannot_change_without_new_version(tmp_path):
    _reg(tmp_path)                              # registers version 1 in the lock
    p = tmp_path / "prereg.toml"
    p.write_text(p.read_text().replace("z_survive = 4.0", "z_survive = 3.0"))
    with pytest.raises(RegistrationError):
        load(p)                                 # changed tests, same version: refused
    import re
    v = int(re.search(r"version = (\d+)", p.read_text()).group(1))
    p.write_text(p.read_text().replace(f"version = {v}", f"version = {v + 1}"))
    assert load(p)["_version"] == v + 1         # a new version is allowed and recorded


def test_only_registered_ranges(tmp_path):
    reg = _reg(tmp_path)
    with pytest.raises(RegistrationError):
        set_range(reg, 3e8)


def test_graveyard_skips_buried_and_reopens(tmp_path):
    from ppa_real.truth.graveyard import Graveyard
    reg = _reg(tmp_path)
    reg["planner"]["max_steps"] = 3
    ledger = run_agent(reg, "explore")
    assert ledger.meta.get("graveyard_skipped")
    g = Graveyard(tmp_path)
    assert g.check("transitions:M=9,lag=1", max_n=1e8).status == "buried"
    assert g.check("transitions:M=9,lag=1", max_n=1e11).status == "reopened"

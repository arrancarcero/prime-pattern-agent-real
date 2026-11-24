"""Load the pre-registration and enforce: tests may not change without a new version.

prereg.lock remembers the fingerprint of every registered version. If prereg.toml's text
changes while its version number stays the same, the agent refuses to run. Changing a test
means declaring version + 1, which is recorded in the lock, and every ledger names the
version and fingerprint its results were judged under.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    tomllib = None


def _parse(text: str) -> dict:
    if tomllib is not None:
        return tomllib.loads(text)
    import ast
    out, sec = {}, None
    buf = ""
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.startswith("[") and not buf:
            sec = out.setdefault(line.strip()[1:-1], {})
            continue
        buf += line.strip() + " "
        if buf.count("[") > buf.count("]"):
            continue
        k, v = buf.split("=", 1)
        sec[k.strip()] = ast.literal_eval(v.strip())
        buf = ""
    return out


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.replace("\r\n", "\n").encode()).hexdigest()[:16]


class RegistrationError(SystemExit):
    pass


def load(path: str | Path = "prereg.toml", lock_path: str | Path | None = None, enforce: bool = True) -> dict:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    reg = _parse(text)
    fp = fingerprint(text)
    version = int(reg["registration"]["version"])
    lock_p = Path(lock_path) if lock_path else p.with_suffix(".lock")
    lock = json.loads(lock_p.read_text()) if lock_p.exists() else {}
    known = lock.get(str(version))
    if known is None:
        lock[str(version)] = {"fingerprint": fp, "registered": reg["registration"].get("registered")}
        lock_p.write_text(json.dumps(lock, indent=1))
    elif known["fingerprint"] != fp and enforce:
        raise RegistrationError(
            f"prereg.toml changed but its version is still {version} (registered fingerprint "
            f"{known['fingerprint']}, now {fp}). Tests may not change after registration: "
            f"set [registration] version = {max(map(int, lock)) + 1} to register new tests.")
    reg["_fingerprint"] = fp
    reg["_version"] = version
    reg["range"] = {"max_n": int(float(reg["ranges"]["default"]))}
    reg.setdefault("output", {"runs_dir": "runs"})
    return reg


def set_range(reg: dict, max_n: float) -> dict:
    allowed = [int(float(a)) for a in reg["ranges"]["allowed"]]
    n = int(float(max_n))
    if n not in allowed:
        raise RegistrationError(f"max_n = {n:.0e} is not a registered range; registered: "
                                + ", ".join(f"{a:.0e}" for a in allowed))
    reg["range"]["max_n"] = n
    return reg

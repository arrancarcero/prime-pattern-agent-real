"""The graveyard: a separate layer that remembers what was WRONG, so it is not tested again.

Two layers:
  GRAVEYARD (frozen tuple, fingerprinted like the truth tuple)
      Ideas we tested and buried by hand, each with the claim, the test that killed it,
      the evidence (numbers), what was tried, and the rule that lets it be dug up again.
  runs/graveyard.json (learned layer, written by the agent)
      Every planner hypothesis that failed its tests, with the range it was tried at.

Before the planner runs a hypothesis it asks check(). A buried idea is skipped, unless
something has changed that could change the answer: a bigger range, a method not yet
tried, or new data. "Wrong" can mean "not yet tested far enough", so every grave says how
to reopen it. Ideas proven impossible (by counting or by proof) have no reopen rule.

Real-primes version: only graves about real primes are kept. The transitions grave's decisive
test was the real gap rule (stage B), which this version runs unchanged.
The graveyard is not a list of shame: half of these are the user's own ideas, and each one
taught something (the dead end is often where the real result was found).
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import namedtuple
from pathlib import Path

Grave = namedtuple("Grave", "key claim killed_by evidence tried reopen_if")
Check = namedtuple("Check", "status key reason")      # status: "buried" | "reopened" | "clear"

NEVER = {}    # reopen_if for ideas ruled out by proof or counting

GRAVEYARD: tuple[Grave, ...] = (
    Grave("transitions:beyond_gaps",
          "consecutive primes' residues (mod up to 72, lags 1-3) hold a pattern beyond what prime gaps explain",
          "transitions skill, stage B (gap-rule null)",
          "primes to 1e10: every pattern up to mod 72, lag 3 explained by gaps; no candidates",
          {"max_n": 1e10, "max_modulus": 72, "max_lag": 3},
          {"max_n_above": 1e10, "max_modulus_above": 72, "max_lag_above": 3}),
    Grave("transitions:fibonacci_24",
          "a window of exactly 24 primes correlates through the Fibonacci 24-cycle",
          "the researcher, 2026-11-11",
          "mod 24 is not special: its gates are a 3-bit XOR group like any (Z/24)*",
          {}, NEVER),
    Grave("digital_root:predict_positions",
          "digital roots alone predict where the next prime is",
          "dr_gap skill and the researcher",
          "replaced by: what is true of a prime's neighbours, given its position",
          {}, NEVER),
    Grave("zeta:negative_primes_one_third_line",
          "a 'negative primes' zeta, with a sign flip, has its zeros on the 1/3 line",
          "zeta walks, Layer 9 (2026-11-22)",
          "the sign-flip walks did not come home on the 1/3 line; single-gate (Hurwitz-type) "
          "functions have zeros that wander off any one line (Davenport-Heilbronn)",
          {"method": ("eta walk", "single-gate walk")},
          {"method_not_in": ("eta walk", "single-gate walk")}),
)

FINGERPRINT = hashlib.sha256(repr(GRAVEYARD).encode()).hexdigest()[:16]
_BY_KEY = {g.key: g for g in GRAVEYARD}


def _reopens(rule: dict, ctx: dict) -> str | None:
    """Return the reason an idea may be dug up, or None if it stays buried."""
    for k, v in rule.items():
        if k.endswith("_above"):
            field = k[: -len("_above")]
            if ctx.get(field) is not None and ctx[field] > v:
                return f"{field} = {ctx[field]:g} is beyond {v:g}"
        elif k.endswith("_not_in"):
            field = k[: -len("_not_in")]
            if ctx.get(field) is not None and ctx[field] not in v:
                return f"new {field}: {ctx[field]}"
    return None


def hypothesis_key(skill: str, params: dict) -> str:
    return f"{skill}:" + ",".join(f"{k}={params[k]}" for k in sorted(params))


class Graveyard:
    """Frozen graves + the learned layer in runs/graveyard.json."""

    def __init__(self, runs_dir: str | Path = "runs"):
        self.path = Path(runs_dir) / "graveyard.json"
        self.learned = json.loads(self.path.read_text()) if self.path.exists() else []

    def check(self, key: str, **ctx) -> Check:
        # 1. a frozen grave for this exact idea
        g = _BY_KEY.get(key)
        if g is not None:
            why = _reopens(g.reopen_if, ctx)
            return Check("reopened", key, why) if why else Check("buried", key, g.evidence)
        # 2. planner transitions fall under the frozen 'beyond gaps' grave
        if key.startswith("transitions:M="):
            p = dict(kv.split("=") for kv in key.split(":", 1)[1].split(","))
            gg = _BY_KEY["transitions:beyond_gaps"]
            covered = int(p["M"]) <= gg.tried["max_modulus"] and int(p["lag"]) <= gg.tried["max_lag"]
            if covered:
                why = _reopens({"max_n_above": gg.tried["max_n"]}, ctx)
                learned_fail = any(e["key"] == key and ctx.get("max_n", 0) <= e["max_n"] for e in self.learned)
                if not why:
                    return Check("buried", key, gg.evidence)
                if not learned_fail:
                    return Check("reopened", key, why)
        # 3. the learned layer: same hypothesis failed before at this range or larger
        for e in self.learned:
            if e["key"] == key:
                if ctx.get("max_n", 0) > e["max_n"]:
                    return Check("reopened", key, f"max_n = {ctx['max_n']:g} beyond the {e['max_n']:g} it failed at")
                return Check("buried", key, f"failed at max_n = {e['max_n']:g}: {e['summary']}")
        return Check("clear", key, "")

    def bury(self, key: str, max_n: float, summary: str) -> None:
        self.learned = [e for e in self.learned if e["key"] != key]
        self.learned.append({"key": key, "max_n": max_n, "summary": summary[:300],
                             "buried": time.strftime("%Y-%m-%d %H:%M")})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.learned, indent=1))

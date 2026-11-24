"""Command line for the real-primes agent.

    ppa-real run [--intent ...] [--max N] [--dig]   run under the registered tests
    ppa-real prereg                                  show the registration and its fingerprint
    ppa-real skills | truth | contract | graveyard [key]

Every run reads prereg.toml. Test settings cannot be overridden from the command line; only a
registered range may be chosen with --max.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ppa-real", description="Prime-Pattern Agent, real primes only")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the agent under the registered tests")
    r.add_argument("--intent", default="all",
                   help="all | explore | dr_gap | fano | spectrum | gates | curves | candles | spacing | walk | twins")
    r.add_argument("--max", type=float, help="one of the registered ranges in prereg.toml")
    r.add_argument("--prereg", default="prereg.toml")
    r.add_argument("--dig", action="store_true", help="ignore the graveyard: retest buried ideas")
    r.add_argument("--gpu", action="store_true", help="sieve and count on an NVIDIA GPU (CuPy), cross-checked against the CPU")
    r.add_argument("--cache", default=".prime_cache",
                   help="folder for the prime cache (2 bytes per prime; 'none' to sieve every pass)")
    p = sub.add_parser("prereg", help="show the registration, its version and fingerprint")
    p.add_argument("--prereg", default="prereg.toml")
    sub.add_parser("skills", help="list skills and their yardsticks")
    t = sub.add_parser("truth", help="self-test the truth tuple")
    t.add_argument("--exp", type=int, default=8)
    sub.add_parser("contract", help="print the agent contract")
    gy = sub.add_parser("graveyard", help="list buried ideas, or check one: ppa-real graveyard [key]")
    gy.add_argument("key", nargs="?")
    a = ap.parse_args(argv)

    if a.cmd == "run":
        from ppa_real.agent.loop import run_agent
        from ppa_real.prereg import load, set_range
        reg = load(a.prereg)
        if a.max:
            set_range(reg, a.max)
        if a.dig:
            reg.setdefault("graveyard", {})["enforce"] = False
        reg["compute"] = {"gpu": a.gpu, "cache": None if a.cache.lower() == "none" else a.cache}
        print(f"pre-registration v{reg['_version']} ({reg['_fingerprint']}), primes from "
              f"{reg['primes']['start']}, range {reg['range']['max_n']:.0e}")
        run_agent(reg, a.intent)
    elif a.cmd == "prereg":
        from ppa_real.prereg import load
        reg = load(a.prereg)
        print(f"version {reg['_version']}  fingerprint {reg['_fingerprint']}  "
              f"registered {reg['registration'].get('registered')}")
        print("seen before registration:")
        for s in reg["registration"].get("seen_before_registration", []):
            print("  -", s)
        print(Path(a.prereg).read_text(encoding="utf-8"))
    elif a.cmd == "skills":
        from ppa_real.skills import REGISTRY
        for s in REGISTRY.values():
            print(f"{s.name:12} yardstick: {s.null_model}\n{'':12} {s.description}")
    elif a.cmd == "truth":
        from ppa_real.truth import truth_tuple
        print("fingerprint", truth_tuple.FINGERPRINT)
        bad = 0
        for e in range(1, a.exp + 1):
            from ppa_real.core.sieve import primes_between
            v = truth_tuple.verify("pi", 10 ** e, len(primes_between(0, 10 ** e)))
            bad += v.ok is False
            print(f"pi(1e{e}) {'ok' if v.ok else v.note}")
        raise SystemExit(1 if bad else 0)
    elif a.cmd == "contract":
        print((Path(__file__).resolve().parents[2] / "CONTRACT.md").read_text(encoding="utf-8"))
    elif a.cmd == "graveyard":
        from ppa_real.truth.graveyard import FINGERPRINT, GRAVEYARD, Graveyard
        gy = Graveyard("runs")
        if a.key:
            c = gy.check(a.key)
            print(f"{c.status.upper()}: {c.key}\n  {c.reason or 'no grave for this idea'}")
        else:
            print(f"graveyard fingerprint {FINGERPRINT}: {len(GRAVEYARD)} frozen graves, "
                  f"{len(gy.learned)} learned in runs/graveyard.json")
            for g in GRAVEYARD:
                reopen = ", ".join(f"{k} {v}" for k, v in g.reopen_if.items()) or "never (proof or counting)"
                print(f"\n[{g.key}]\n  claim:    {g.claim}\n  killed by: {g.killed_by}\n"
                      f"  evidence: {g.evidence}\n  reopen if: {reopen}")


if __name__ == "__main__":
    main()

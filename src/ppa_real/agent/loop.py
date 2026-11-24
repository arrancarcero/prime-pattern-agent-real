"""The agent loop: calibrate -> fixed skills -> explore (plan, test, learn) -> ledger."""
from __future__ import annotations

import time
from collections import Counter

from ppa_real.agent.planner import Planner
from ppa_real.core.ledger import Ledger
from ppa_real.core.sieve import real_primes
from ppa_real.skills import REGISTRY
from ppa_real.skills.base import Context
from ppa_real.truth.graveyard import FINGERPRINT as GRAVE_FP
from ppa_real.truth.graveyard import Graveyard, hypothesis_key

FIXED = ("dr_gap", "fano", "spectrum", "gates", "curves", "candles", "spacing", "walk", "twins")


def build_context(cfg: dict) -> Context:
    max_n = cfg["range"]["max_n"]
    train = (max_n // 10, max_n // 2)
    test = (max_n // 2, max_n)
    t = time.time()
    P_train, P_test = real_primes(*train), real_primes(*test)
    print(f"sieved {len(P_train) + len(P_test):,} primes in {time.time() - t:.1f}s "
          f"(train {train[0]:.0e}-{train[1]:.0e}, test {test[0]:.0e}-{test[1]:.0e})")
    return Context(cfg, train, test, P_train, P_test)


def run_agent(cfg: dict, intent: str = "all") -> Ledger:
    ledger = Ledger(cfg["output"]["runs_dir"], {"config": cfg, "intent": intent,
                                                "prereg_version": cfg.get("_version"),
                                                "prereg_fingerprint": cfg.get("_fingerprint"),
                                                "started": time.strftime("%Y-%m-%d %H:%M:%S")})
    needs_primes = intent not in ("curves", "candles", "spacing", "walk", "twins")
    ctx = build_context(cfg) if needs_primes else Context(cfg, (0, 0), (0, 0), None, None)

    if needs_primes:  # contract rule 1
        print("calibrate")
        cal = ledger.add(REGISTRY["calibrate"].run(ctx)[0])
        if cal.survived is False:
            ledger.save()
            raise SystemExit("Sieve disagrees with the truth tuple. Stopping.")

    if intent == "all":
        fixed = FIXED
    elif intent in REGISTRY and intent not in ("transitions", "calibrate"):
        fixed = (intent,)
    else:
        fixed = ()
    for name in fixed:
        print(name)
        for f in REGISTRY[name].run(ctx):
            ledger.add(f)

    if intent in ("all", "explore", "transitions"):
        print("explore")
        planner = Planner(cfg)
        gy_cfg = cfg.get("graveyard", {})
        enforce = gy_cfg.get("enforce", True)
        graves = Graveyard(cfg["output"]["runs_dir"])
        max_n = float(cfg["range"]["max_n"])
        skipped, steps = [], 0
        while steps < cfg["planner"]["max_steps"]:
            h = planner.next()
            if h is None:
                break
            key = hypothesis_key(h.skill, h.params)
            chk = graves.check(key, max_n=max_n, max_modulus=h.params.get("M"), max_lag=h.params.get("lag"))
            if enforce and chk.status == "buried":
                skipped.append({"key": key, "why": chk.reason})
                continue                       # skipped ideas do not use up a step
            steps += 1
            f = REGISTRY[h.skill].run(ctx, **h.params)[0]
            f.details["planner_reason"] = h.reason
            if chk.status == "reopened":
                f.details["dug_up_from_graveyard"] = chk.reason
            ledger.add(f)
            planner.learn(h, f)
            if f.survived is False:
                graves.bury(key, max_n, f.summary)
        if skipped:
            ledger.meta["graveyard_skipped"] = skipped
            print(f"graveyard: skipped {len(skipped)} buried idea(s) (fingerprint {GRAVE_FP}); "
                  f"first: {skipped[0]['key']} -- {skipped[0]['why'][:90]}")
            print("  to test them anyway: ppa run --dig   (or raise range.max_n past where they failed)")

    counts = Counter(f.label for f in ledger.findings)
    print("summary:", dict(counts))
    cands = [f.name for f in ledger.findings if f.label == "candidate"]
    print("candidates for a human to check:", cands or "none")
    print("ledger written to", ledger.save())
    return ledger

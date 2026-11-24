from ppa_real.core.ledger import Finding
from ppa_real.skills.base import Skill
from ppa_real.truth import truth_tuple


def run(ctx, **_):
    max_n = ctx.test[1]
    total = ctx.primes.count(0, max_n)                               # primes >= 5, streamed
    v = truth_tuple.verify("pi", max_n, total + 2)                 # theorem counts 2 and 3 too
    label = "established" if v.ok else "check"
    note = "no known value for this max_n" if v.ok is None else ("matches" if v.ok else v.note)
    return [Finding("calibrate", f"primes from 5 to {max_n:.0e}: {total:,} (= pi - 2)", label,
                    f"truth tuple {truth_tuple.FINGERPRINT}: {note}",
                    survived=v.ok, details={"fingerprint": truth_tuple.FINGERPRINT})]


SKILL = Skill("calibrate", "truth tuple: exact pi(x), minus 2 and 3",
              "Sieve must reproduce pi(max_n) before anything else runs.", run)

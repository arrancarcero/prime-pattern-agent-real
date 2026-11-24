"""The planner: proposes the next hypothesis from what has been found so far.

Pattern-matching rules for finding further concepts:
  seed      every start modulus at every start lag
  deepen    a surviving pattern is retested one lag further (does it persist?)
  combine   two surviving moduli are joined at their lcm (the Chinese Remainder /
            moire idea: does the joint state hold more than its two shadows?)
  refine    a candidate (survived, not known) is retested at 2M and 3M
Candidates jump the queue. Retired ideas are never proposed.
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from math import lcm

from ppa_real.truth.known_results import RETIRED


@dataclass(order=True)
class Hypothesis:
    priority: int
    order: int
    skill: str = field(compare=False)
    params: dict = field(compare=False)
    reason: str = field(compare=False, default="seed")

    @property
    def key(self):
        return (self.skill, tuple(sorted(self.params.items())))


class Planner:
    def __init__(self, cfg: dict):
        self.cfg = cfg["planner"]
        self._q: list[Hypothesis] = []
        self._seen: set = set()
        self._count = itertools.count()
        self.survivors: list[int] = []
        self.retired = {r["idea"] for r in RETIRED}
        for M in self.cfg["start_moduli"]:
            for lag in self.cfg["start_lags"]:
                self.propose("transitions", {"M": M, "lag": lag}, 5, "seed")

    def propose(self, skill, params, priority, reason):
        h = Hypothesis(priority, next(self._count), skill, params, reason)
        if h.key in self._seen:
            return
        if params.get("M", 0) > self.cfg["max_modulus"] or params.get("lag", 1) > self.cfg["max_lag"]:
            return
        self._seen.add(h.key)
        heapq.heappush(self._q, h)

    def next(self) -> Hypothesis | None:
        return heapq.heappop(self._q) if self._q else None

    def learn(self, h: Hypothesis, finding) -> None:
        stage_a = finding.details.get("stageA_vs_independence", {}).get("survived")
        if not (finding.survived or stage_a):
            return
        M, lag = h.params["M"], h.params["lag"]
        self.propose("transitions", {"M": M, "lag": lag + 1}, 4, f"deepen mod {M}")
        for other in self.survivors:
            L = lcm(M, other)
            if L not in (M, other):
                self.propose("transitions", {"M": L, "lag": 1}, 3, f"combine {M} and {other}")
        if finding.label == "candidate":
            for k in (2, 3):
                self.propose("transitions", {"M": k * M, "lag": lag}, 1, f"refine candidate mod {M}")
        if M not in self.survivors:
            self.survivors.append(M)

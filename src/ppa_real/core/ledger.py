"""The ledger: every result, with its honesty label, written to runs/."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

LABELS = ("established", "conjecture", "hypothesis", "speculation", "retired",
          "rediscovery", "candidate", "not-supported", "check", "measured")   # measured: no yardstick, argument chain given


@dataclass
class Finding:
    skill: str
    name: str
    label: str
    summary: str
    z_test: float | None = None
    corr: float | None = None
    survived: bool | None = None
    params: dict = field(default_factory=dict)
    details: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.label not in LABELS:
            raise ValueError(f"unknown label {self.label!r}")


class Ledger:
    def __init__(self, runs_dir: str | Path, meta: dict):
        self.dir = Path(runs_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta = meta
        self.findings: list[Finding] = []

    def add(self, f: Finding) -> Finding:
        self.findings.append(f)
        mark = {True: "SURVIVED", False: "faded", None: ""}[f.survived]
        z = "" if f.z_test is None else f"z={f.z_test:+.1f}"
        print(f"  [{f.label:13}] {f.name:48} {z:>9} {mark}")
        return f

    def save(self) -> Path:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = self.dir / f"ledger-{stamp}.json"
        payload = {"meta": self.meta, "findings": [asdict(f) for f in self.findings]}
        path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
        return path

"""What every skill looks like, and the shared run context."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class Context:
    cfg: dict
    train: tuple[int, int]
    test: tuple[int, int]
    P_train: np.ndarray
    P_test: np.ndarray
    cache: dict = field(default_factory=dict)


@dataclass
class Skill:
    name: str
    null_model: str          # stated before any data is looked at (contract rule 2)
    description: str
    run: Callable            # run(ctx, **params) -> list[Finding]
    needs_primes: bool = True

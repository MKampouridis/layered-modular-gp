"""Interface every arm implements. Selection, linear scaling, the compute
counter, logging and the run loop are shared (lmgp.run); only the
representation and its variation operators differ between arms."""
from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from ..core.program import Program


class Arm(Protocol):
    name: str

    def init_population(self, n: int, rng: np.random.Generator) -> list[Program]: ...

    def vary(self, parents: Sequence[Program], rng: np.random.Generator) -> Program:
        """Produce one offspring from the given parents (at least two are
        supplied). The offspring must satisfy the size and depth caps."""
        ...

    def is_valid(self, p: Program) -> bool: ...

    def describe(self) -> dict: ...

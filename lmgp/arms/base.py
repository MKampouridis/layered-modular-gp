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


def acceptable(arm, child, parents) -> bool:
    """Shared offspring rule for every arm: within the caps, and different
    from both parents in its active code. An offspring that changed only
    dead code (an inactive module or tree, an unread ADF argument) or
    nothing at all computes exactly what a parent computes, so evaluating
    it would spend budget on no new information. It is retried like an
    invalid offspring (max_tries, then the first parent is copied). CGP's
    single-active mutation guarantees this by construction; without the rule
    29-47% of A2, A4 and A6 offspring were such no-ops, against 0-6% for A1
    and A3 (pre-CERES review, 3 Nov 2026)."""
    if not arm.is_valid(child):
        return False
    k = child.active_key
    return all(k != p.active_key for p in parents[:2])

"""Task container shared by all benchmarks."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.primitives import PrimitiveSet


@dataclass
class Task:
    name: str
    kind: str  # "regression" or "boolean"
    pset: PrimitiveSet
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray | None = None
    y_val: np.ndarray | None = None
    X_test: np.ndarray | None = None
    y_test: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n_train(self) -> int:
        return len(self.y_train)

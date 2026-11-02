"""Boolean benchmarks: even-n parity and the k-address multiplexer.

Fitness cases are the full truth table, so there is no train/test split; the
measure of success is an exact solution (zero errors on all cases).
"""
from __future__ import annotations

import itertools

import numpy as np

from ..core.primitives import multiplexer_set, parity_set
from .task import Task


def truth_table(n: int) -> np.ndarray:
    return np.array(list(itertools.product([False, True], repeat=n)), dtype=bool)


def even_parity(n: int) -> Task:
    X = truth_table(n)
    y = (X.sum(axis=1) % 2 == 0)
    return Task(f"parity{n}", "boolean", parity_set(n), X, y, meta={"n_bits": n})


def multiplexer(k: int) -> Task:
    """k address bits, 2^k data bits: 6-mux (k=2), 11-mux (k=3), 20-mux (k=4).
    Inputs are ordered address bits first (a0 is the most significant), then
    data bits d0..d(2^k-1)."""
    n = k + 2 ** k
    X = truth_table(n)
    addr = np.zeros(len(X), dtype=int)
    for i in range(k):
        addr = addr * 2 + X[:, i].astype(int)
    y = X[np.arange(len(X)), k + addr]
    return Task(f"mux{n}", "boolean", multiplexer_set(n), X, y, meta={"address_bits": k, "n_bits": n})

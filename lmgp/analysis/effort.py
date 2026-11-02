"""Koza's computational effort (Koza 1992, 1994) and a compute-unit version.

Given independent runs with the resource used at first solution (or None if
unsolved), P(r) is the fraction of runs solved within resource r, and

    E = min_r  r * R(z, P(r)),   R = ceil(log(1 - z) / log(1 - P(r))),  z = 0.99.

Koza's original form counts individuals processed, r = M * (i + 1) for
population M and generation i. We also report r in node-evaluation units (our
compute counter), which is the fair measure when program sizes differ between
arms. The estimator is biased for few runs (Christensen and Oppacher 2002;
Niehaus and Banzhaf 2003), so we report bootstrap intervals and success rates
with it.
"""
from __future__ import annotations

import math

import numpy as np


def effort(resource_at_solution, z: float = 0.99) -> float:
    """resource_at_solution: per-run value (np.nan/None = not solved)."""
    r = np.array([np.nan if v is None else float(v) for v in resource_at_solution])
    n = len(r)
    solved = np.sort(r[np.isfinite(r)])
    if len(solved) == 0:
        return math.inf
    best = math.inf
    for k, res in enumerate(solved, start=1):
        if k < len(solved) and solved[k] == res:
            continue  # evaluate at the last of tied values
        p = k / n
        R = 1 if p >= 1 else math.ceil(math.log(1 - z) / math.log(1 - p))
        best = min(best, res * R)
    return best


def bootstrap_effort(resource_at_solution, n_boot: int = 2000, seed: int = 0, z: float = 0.99):
    r = list(resource_at_solution)
    rng = np.random.default_rng(seed)
    vals = np.array([effort([r[i] for i in rng.integers(len(r), size=len(r))], z) for _ in range(n_boot)])
    finite = vals[np.isfinite(vals)]
    lo, hi = (np.percentile(finite, [2.5, 97.5]) if len(finite) else (math.inf, math.inf))
    return effort(r, z), float(lo), float(hi), float(np.mean(~np.isfinite(vals)))

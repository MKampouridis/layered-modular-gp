"""Lexicase selection (Spector 2012; Helmuth, Spector and Matheson 2015) and
epsilon-lexicase with semi-dynamic epsilon (La Cava, Spector and Danai 2016;
La Cava, Helmuth, Spector and Moore 2019).

errors: array (pop_size, n_cases), lower is better.
"""
from __future__ import annotations

import numpy as np


def mad_epsilon(errors: np.ndarray) -> np.ndarray:
    """Per-case median absolute deviation of the population's errors."""
    med = np.median(errors, axis=0)
    return np.median(np.abs(errors - med), axis=0)


def lexicase_select(errors: np.ndarray, n: int, rng: np.random.Generator, epsilon=None) -> np.ndarray:
    """Select n parent indices.

    epsilon=None gives standard lexicase (exact ties). With a per-case
    epsilon vector, a candidate survives a case if its error is within epsilon
    of the best error *among the current pool* (semi-dynamic).
    """
    pop, n_cases = errors.shape
    # Collapse identical error vectors: selection only depends on the vectors.
    uniq, inverse = np.unique(errors, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    groups = [np.flatnonzero(inverse == g) for g in range(len(uniq))]
    counts = np.array([len(g) for g in groups], dtype=float)
    eps = None if epsilon is None else np.asarray(epsilon)
    out = np.empty(n, dtype=np.int64)
    all_idx = np.arange(len(uniq))
    for s in range(n):
        pool = all_idx
        for c in rng.permutation(n_cases):
            col = uniq[pool, c]
            best = col.min()
            thr = best if eps is None else best + eps[c]
            pool = pool[col <= thr]
            if len(pool) == 1:
                break
        # Uniform over the surviving *individuals* (not distinct vectors).
        if len(pool) > 1:
            w = counts[pool]
            g = pool[rng.choice(len(pool), p=w / w.sum())]
        else:
            g = pool[0]
        members = groups[g]
        out[s] = members[rng.integers(len(members))]
    return out


def epsilon_lexicase_select(errors: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return lexicase_select(errors, n, rng, epsilon=mad_epsilon(errors))


def tournament_select(fitness: np.ndarray, n: int, rng: np.random.Generator, size: int = 7) -> np.ndarray:
    """Tournament selection on a scalar fitness (lower is better). Provided
    for Koza-style reference runs; not the default for any arm."""
    cand = rng.integers(len(fitness), size=(n, size))
    return cand[np.arange(n), np.argmin(fitness[cand], axis=1)]

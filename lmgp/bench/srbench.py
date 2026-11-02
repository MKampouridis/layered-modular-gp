"""SRBench loader (La Cava et al. 2021) via the `pmlb` package.

Splits follow SRBench: 75% train / 25% test, shuffled with the run's seed, and
training data capped at 10,000 rows. We additionally hold out 25% of the
training portion as the validation set used to choose the final model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.primitives import regression_set
from ..core.rng import make_rng
from .task import Task

CACHE = Path(__file__).resolve().parents[2] / "data" / "pmlb"

# Black-box subset for the G1 sanity check: small, non-linear (GP beats linear
# regression in SRBench), with published gplearn and EPLEX results.
G1_SUBSET = ("1027_ESL", "230_machine_cpu", "561_cpu", "556_analcatdata_apnea2", "227_cpu_small")


def load_xy(name: str, cache: Path = CACHE):
    import pmlb

    cache.mkdir(parents=True, exist_ok=True)
    X, y = pmlb.fetch_data(name, return_X_y=True, local_cache_dir=str(cache))
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def srbench_task(name: str, split_seed: int, max_train: int = 10_000, val_frac: float = 0.25) -> Task:
    X, y = load_xy(name)
    rng = make_rng(split_seed, 7)
    idx = rng.permutation(len(y))
    n_test = int(round(0.25 * len(y)))
    test, rest = idx[:n_test], idx[n_test:]
    rest = rest[:max_train]
    n_val = int(round(val_frac * len(rest)))
    val, train = rest[:n_val], rest[n_val:]
    return Task(
        f"srbench_{name}", "regression", regression_set(X.shape[1]),
        X[train], y[train], X[val], y[val], X[test], y[test],
        meta={"dataset": name, "split_seed": split_seed, "n_rows": len(y), "n_features": X.shape[1]},
    )

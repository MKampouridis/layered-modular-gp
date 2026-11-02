"""Linear scaling of program outputs (Keijzer 2003)."""
from __future__ import annotations

import numpy as np

BIG = 1e30  # error assigned to non-finite predictions


def linear_scale(f: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (a, b) minimising mean((y - (a + b f))^2).

    If f is constant or non-finite, b = 0 and a = mean(y)."""
    if not np.all(np.isfinite(f)):
        return float(np.mean(y)), 0.0
    fm = f.mean()
    df = f - fm
    var = float(np.dot(df, df))
    if var < 1e-12 * max(1.0, float(np.dot(f, f))) or not np.isfinite(var):
        return float(np.mean(y)), 0.0
    b = float(np.dot(df, y - y.mean()) / var)
    a = float(y.mean() - b * fm)
    return a, b


def scaled_predictions(f: np.ndarray, a: float, b: float) -> np.ndarray:
    with np.errstate(all="ignore"):
        return a + b * f

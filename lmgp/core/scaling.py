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


def linear_readout(F: np.ndarray, y: np.ndarray, alpha: float = 1e-6):
    """Ridge least squares y ~ a + F w over the columns of F (A4's readout).

    Columns are standardised for the solve; non-finite or constant columns
    get weight 0. `alpha` is relative to the number of rows. Returns (a, w)."""
    n, k = F.shape
    w = np.zeros(k)
    ok = np.all(np.isfinite(F), axis=0)
    if ok.any():
        G = F[:, ok]
        mu = G.mean(axis=0)
        sd = G.std(axis=0)
        use = sd > 1e-12 * np.maximum(1.0, np.abs(mu))
        if use.any():
            Z = (G[:, use] - mu[use]) / sd[use]
            yc = y - y.mean()
            A = Z.T @ Z + alpha * n * np.eye(Z.shape[1])
            try:
                beta = np.linalg.solve(A, Z.T @ yc)
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(A, Z.T @ yc, rcond=None)[0]
            wk = np.zeros(G.shape[1])
            wk[use] = beta / sd[use]
            w[ok] = wk
            if not np.all(np.isfinite(w)):
                w[:] = 0.0
    a = float(y.mean() - (F[:, w != 0].mean(axis=0) @ w[w != 0])) if np.any(w != 0) else float(y.mean())
    return a, w


def scaled_predictions(f: np.ndarray, a: float, b) -> np.ndarray:
    """a + b f for a single output, or a + F w for a readout matrix."""
    with np.errstate(all="ignore"):
        if np.ndim(f) == 2:
            nz = np.flatnonzero(b)  # zero-weight columns may be non-finite
            return a + f[:, nz] @ b[nz]
        return a + b * f

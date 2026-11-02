"""Stage 1 synthetic compositional regression tasks.

Each target composes small two-argument sub-functions over L levels
(1-4), with the sub-function of each level called R times (1-4), plus Gaussian
noise at a fraction of the clean target's standard deviation.

Construction (fixed rule, all choices drawn from the task seed):

* Inputs x0..x(d-1), d = 5, sampled uniformly on [-2, 2]. Separate train,
  validation and test samples.
* Each level uses a different sub-function from LIBRARY.
* Level 1: R calls, each on a distinct unordered pair of distinct inputs.
* Level l > 1: R calls. Call r takes as first argument unit (r mod R_prev) of
  level l-1, so every unit of the level below is used; the second argument is
  drawn from the other units of level l-1 and the raw inputs, without
  repeating an unordered pair within the level.
* Every unit is standardised (zero mean, unit variance) using statistics from
  a fixed reference sample of 20,000 inputs, so the signal neither explodes
  nor vanishes with depth. Standardisation is affine, so correlation-based
  structure recovery is unaffected.
* Target = sum of the level-L units.
* Noise: N(0, (noise * sd(clean train target))^2), added to train and
  validation targets. Test targets are noise-free (we measure how well the
  true function is learned).

The task records the sub-function definitions and the full ground-truth
structure (units, arguments and standardisation constants) in `meta`, which the
structure-recovery metric needs later.
"""
from __future__ import annotations

import numpy as np

from ..core.primitives import regression_set
from ..core.rng import make_rng
from .task import Task

# Fixed library of sub-functions; each is expressible in the regression set.
LIBRARY = {
    "g0": ("a*b", lambda a, b: a * b),
    "g1": ("sin(a) + cos(b)", lambda a, b: np.sin(a) + np.cos(b)),
    "g2": ("a / sqrt(1 + b^2)", lambda a, b: a / np.sqrt(1 + b * b)),
    "g3": ("a * cos(b)", lambda a, b: a * np.cos(b)),
    "g4": ("(a - b) * (a + b)", lambda a, b: (a - b) * (a + b)),
    "g5": ("sin(a * b)", lambda a, b: np.sin(a * b)),
}

N_VARS = 5
X_RANGE = (-2.0, 2.0)
N_REF = 20_000
SIZES = {"train": 256, "val": 256, "test": 2048}


def build_structure(levels: int, reuse: int, rng: np.random.Generator) -> dict:
    names = list(LIBRARY)
    fns = [names[i] for i in rng.permutation(len(names))[:levels]]
    units = []  # each: {"id", "level", "fn", "args"}
    prev: list[str] = []
    inputs = [f"x{j}" for j in range(N_VARS)]
    for lvl in range(1, levels + 1):
        used: set = set()
        cur = []
        for r in range(reuse):
            if lvl == 1:
                while True:
                    a, b = rng.choice(N_VARS, size=2, replace=False)
                    pair = frozenset((int(a), int(b)))
                    if pair not in used:
                        break
                used.add(pair)
                args = [inputs[a], inputs[b]]
            else:
                first = prev[r % len(prev)]
                cands = [u for u in prev if u != first] + inputs
                order = rng.permutation(len(cands))
                for o in order:
                    pair = frozenset((first, cands[o]))
                    if pair not in used:
                        break
                used.add(pair)
                args = [first, cands[o]]
                if rng.random() < 0.5:  # random argument order
                    args.reverse()
            uid = f"u{lvl}_{r}"
            units.append({"id": uid, "level": lvl, "fn": fns[lvl - 1], "args": args})
            cur.append(uid)
        prev = cur
    return {"subfunctions_used": fns, "units": units, "output": prev}


def evaluate_structure(struct: dict, X: np.ndarray, standardise=True) -> np.ndarray:
    env = {f"x{j}": X[:, j] for j in range(X.shape[1])}
    for u in struct["units"]:
        v = LIBRARY[u["fn"]][1](env[u["args"][0]], env[u["args"][1]])
        if standardise:
            v = (v - u["mean"]) / u["std"]
        env[u["id"]] = v
    return sum(env[o] for o in struct["output"])


def compositional_task(levels: int, reuse: int, noise: float, instance: int = 0, sizes=None) -> Task:
    """Instance `instance` of the (levels, reuse, noise) cell. The structure
    and data depend only on (levels, reuse, instance), so the three noise
    levels share the same underlying function and inputs."""
    sizes = sizes or SIZES
    base_seed = 1_000_003 * levels + 10_007 * reuse + instance
    srng = make_rng(base_seed, 0)
    struct = build_structure(levels, reuse, srng)

    # Standardisation constants from a fixed reference sample.
    Xref = make_rng(base_seed, 1).uniform(*X_RANGE, size=(N_REF, N_VARS))
    env = {f"x{j}": Xref[:, j] for j in range(N_VARS)}
    for u in struct["units"]:
        v = LIBRARY[u["fn"]][1](env[u["args"][0]], env[u["args"][1]])
        u["mean"], u["std"] = float(v.mean()), float(v.std() + 1e-12)
        env[u["id"]] = (v - u["mean"]) / u["std"]

    drng = make_rng(base_seed, 2)
    X = {k: drng.uniform(*X_RANGE, size=(n, N_VARS)) for k, n in sizes.items()}
    y = {k: evaluate_structure(struct, X[k]) for k in sizes}
    nrng = make_rng(base_seed, 3, int(round(noise * 1000)))
    sd = float(np.std(y["train"]))
    for k in ("train", "val"):
        y[k] = y[k] + nrng.normal(0.0, noise * sd, size=len(y[k])) if noise > 0 else y[k]

    meta = {
        "levels": levels,
        "reuse": reuse,
        "noise": noise,
        "instance": instance,
        "n_vars": N_VARS,
        "x_range": list(X_RANGE),
        "library": {k: v[0] for k, v in LIBRARY.items()},
        "structure": struct,
        "noise_sd": noise * sd,
    }
    name = f"comp_L{levels}_R{reuse}_n{noise:g}_i{instance}"
    return Task(name, "regression", regression_set(N_VARS), X["train"], y["train"],
                X["val"], y["val"], X["test"], y["test"], meta=meta)

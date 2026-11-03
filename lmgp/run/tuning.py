"""Equal-effort tuning (research plan, Experimental protocol): every arm gets
the same random-search budget, 20 configurations, evaluated on separate
tuning tasks that are never used for screening or testing.

The ranges below cover each arm's main operator and shape parameters. Each
arm's configuration 0 is its default (the published or textbook setting), so
tuning can only improve on it; configurations 1..19 are drawn at random from
a fixed seed. Ranking: for each tuning task, configurations are ranked by
median validation MSE (regression) or by median best training error, then
median units to solution (Boolean); the mean rank over tasks decides.
"""
from __future__ import annotations

import numpy as np

N_CONFIGS = 20
TUNING_SEED = 20261103

# Tuning tasks: synthetic instances 100-102 (screening and test use 0-2) and
# even-5 parity. Even-5 is therefore NOT a screening task (parity has no
# instances: the task is its whole truth table). Budget 1e10, the middle of
# the ladder; 5 seeds each.
TUNING_TASKS = [
    {"kind": "comp", "levels": 2, "reuse": 2, "noise": 0.01, "instance": 100},
    {"kind": "comp", "levels": 3, "reuse": 3, "noise": 0.01, "instance": 101},
    {"kind": "comp", "levels": 4, "reuse": 2, "noise": 0.01, "instance": 102},
    {"kind": "parity", "n": 5},
]
TUNING_BUDGET = 1e10
TUNING_SEEDS = [100, 101, 102, 103, 104]  # seed-list indices kept apart from screening (0-29)
ARMS = ["a1", "a2", "a3", "a3_lb1", "a3_lbmax", "a4", "a5", "a6", "a6_inputs"]


def _rates(rng, names, alpha=1.0):
    w = rng.dirichlet([alpha] * len(names))
    return {n: round(float(v), 3) for n, v in zip(names, w)}


def sample(arm: str, rng: np.random.Generator) -> dict:
    """One random configuration (arm_params) for `arm`."""
    if arm in ("a1", "a2"):
        p = {"p_crossover": round(float(rng.uniform(0.5, 1.0)), 3),
             "mutation_depth": int(rng.integers(2, 6)),
             "init_depth": [2, int(rng.integers(4, 7))]}
        if arm == "a2":
            p["adf_arities"] = [[2], [2, 3], [2, 2, 3], [3, 3]][int(rng.integers(4))]
        return p
    # CGP: the grid shape is what there is to tune (single-active mutation has
    # no rate). n_const is not tuned: it has no effect on Boolean tasks.
    if arm == "a3":
        cols = int(rng.choice([50, 100, 200, 300]))
        return {"rows": int(rng.choice([1, 2, 4])), "cols": cols,
                "levels_back": max(1, int(cols * float(rng.choice([1.0, 0.5, 0.2]))))}
    if arm in ("a3_lb1", "a3_lbmax"):
        # 6 columns and the levels-back setting are the ablation itself (and
        # the depth cap), so only the number of rows is tuned.
        return {"rows": int(rng.choice(np.arange(2, 41, 2)))}
    if arm == "a4":
        p = _rates(rng, ["p_macro", "p_crossover", "p_mutation"])
        # Keep the published shape (20 trees: the first 10 read the inputs,
        # window 10) in proportion: half the trees read the inputs and the
        # window is half the trees. With input_trees and window fixed at 10,
        # 10 trees had no second layer and 30 trees almost never initialised
        # within the cap (pre-CERES review).
        n = int(rng.choice([10, 14, 20, 24]))
        p.update({"n_trees": n, "input_trees": n // 2, "window": n // 2})
        return p
    if arm in ("a5", "a6", "a6_inputs"):
        p = _rates(rng, ["p_crossover", "p_mutation", "p_rewire", "p_encapsulate", "p_delete"])
        p.update({"max_modules": int(rng.choice([4, 8, 12])),
                  "module_max_depth": int(rng.choice([2, 3, 4])),
                  "mutation_depth": int(rng.choice([1, 2, 3]))})
        return p
    raise ValueError(arm)


def effective(arm: str, params: dict) -> str:
    """The settings an arm actually runs with, on a Boolean and a regression
    task: two configurations are duplicates if both strings are equal."""
    import dataclasses
    import json
    from types import SimpleNamespace

    from ..core.primitives import parity_set, regression_set
    from .runner import build_arm

    out = []
    for pset in (parity_set(5), regression_set(5)):
        a = build_arm(arm, SimpleNamespace(pset=pset), params)
        c = dataclasses.asdict(getattr(a, "a4", None) or a.cfg)
        c.pop("extra", None)
        if hasattr(a, "nc"):
            c["n_const"] = a.nc  # what CGP really uses (0 on Boolean tasks)
        out.append(json.dumps(c, sort_keys=True, default=str))
    return " | ".join(out)


def configurations(arm: str) -> list[dict]:
    """The 20 configurations of `arm`: the default first, then 19 random,
    all distinct in the settings actually used (redrawn on a duplicate), so
    every arm gets 20 genuinely different configurations."""
    rng = np.random.default_rng([TUNING_SEED, ARMS.index(arm)])
    out, seen = [{}], {effective(arm, {})}
    for _ in range(10_000):
        if len(out) == N_CONFIGS:
            return out
        p = sample(arm, rng)
        e = effective(arm, p)
        if e not in seen:
            seen.add(e)
            out.append(p)
    raise RuntimeError(f"{arm}: fewer than {N_CONFIGS} distinct configurations in its tuning space")


def tuning_batch() -> dict:
    """The whole tuning campaign as one batch config (cases = arm x config)."""
    cases = [{"arm": arm, "arm_params": cfg, "tuning_config": i}
             for arm in ARMS for i, cfg in enumerate(configurations(arm))]
    return {"base": {"budget": TUNING_BUDGET, "pop_size": "auto", "max_generations": 10 ** 9,
                     "checkpoint_every": 25},
            "cases": cases, "grid": {"task": TUNING_TASKS}, "seeds": TUNING_SEEDS}


def rank_configs(df):
    """Mean rank of each (arm, tuning_config) over the tuning tasks; lower is
    better. `df` holds one row per run (final metrics plus the config)."""
    import pandas as pd

    rows = []
    for (arm, task), g in df.groupby(["arm", "task"]):
        if g["kind"].iloc[0] == "boolean":
            m = g.groupby("tuning_config").agg(err=("best_train", "median"),
                                               units=("units_to_solution_or_budget", "median"))
            # lexicographic: error first, then units to solution
            m["rank"] = (m["err"].rank(method="min") * 1e6 + m["units"].rank(method="min")).rank(method="average")
        else:
            m = g.groupby("tuning_config").agg(err=("val_mse", "median"))
            m["rank"] = m["err"].rank(method="average")
        m = m.reset_index()
        m["arm"], m["task"] = arm, task
        rows.append(m)
    r = pd.concat(rows)
    return (r.groupby(["arm", "tuning_config"])["rank"].mean().rename("mean_rank").reset_index()
            .sort_values(["arm", "mean_rank", "tuning_config"]))

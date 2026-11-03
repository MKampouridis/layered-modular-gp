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
# even-5 parity. Budget 1e10, the middle of the ladder; 5 seeds each.
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
    if arm == "a3":
        cols = int(rng.choice([50, 100, 200]))
        return {"cols": cols, "levels_back": cols, "n_const": int(rng.choice([1, 2, 4]))}
    if arm in ("a3_lb1", "a3_lbmax"):
        return {"n_const": int(rng.choice([1, 2, 4]))}  # the grid is the ablation, fixed
    if arm == "a4":
        p = _rates(rng, ["p_macro", "p_crossover", "p_mutation"])
        p["n_trees"] = int(rng.choice([10, 20, 30]))
        return p
    if arm in ("a5", "a6", "a6_inputs"):
        p = _rates(rng, ["p_crossover", "p_mutation", "p_rewire", "p_encapsulate", "p_delete"])
        p.update({"max_modules": int(rng.choice([4, 8, 12])),
                  "module_max_depth": int(rng.choice([2, 3, 4])),
                  "mutation_depth": int(rng.choice([1, 2, 3]))})
        return p
    raise ValueError(arm)


def configurations(arm: str) -> list[dict]:
    """The 20 configurations of `arm`: the default first, then 19 random."""
    rng = np.random.default_rng([TUNING_SEED, ARMS.index(arm)])
    return [{}] + [sample(arm, rng) for _ in range(N_CONFIGS - 1)]


def tuning_batch() -> dict:
    """The whole tuning campaign as one batch config (cases = arm x config)."""
    cases = [{"arm": arm, "arm_params": cfg, "tuning_config": i}
             for arm in ARMS for i, cfg in enumerate(configurations(arm))]
    return {"base": {"budget": TUNING_BUDGET, "pop_size": "auto", "max_generations": 10 ** 9},
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

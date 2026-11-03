"""Config-driven runner: one run = (arm, task, budget, seed).

Outputs (in out_dir/<run_id>/):
  generations.parquet  per-generation log
  final.parquet        one row of final metrics
  final.json           same, plus best program and environment versions
  ckpt.pkl             checkpoint (if enabled), removed on completion

Usage:
  python -m lmgp.run.runner CONFIG.yaml [--index I] [--out DIR]
CONFIG is either a single run config or a batch (keys base/grid/seeds); for a
batch, --index selects the run (e.g. the array-job task id).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

from ..arms.a1_tree import make_a1
from ..arms.a2_adf import make_a2
from ..arms.hier import VARIANTS as HIER_VARIANTS
from ..arms.hier import HierArmConfig, make_hier
from ..arms.tree import TreeArmConfig
from ..bench.boolean import even_parity, multiplexer
from ..bench.srbench import srbench_task
from ..bench.synthetic import compositional_task
from ..bench.task import Task
from ..core.counter import ComputeCounter
from ..core.rng import make_rng, seed_for
from ..core.scaling import BIG, linear_scale, scaled_predictions
from ..selection.lexicase import epsilon_lexicase_select, lexicase_select, tournament_select
from .config import DEFAULTS, deep_merge, expand_batch, load_yaml, population_for_budget, run_id


# ---------------------------------------------------------------- building --
def build_task(spec: dict, seed: int) -> Task:
    k = spec["kind"]
    if k == "parity":
        return even_parity(spec["n"])
    if k == "mux":
        return multiplexer(spec["k"])
    if k == "comp":
        return compositional_task(spec["levels"], spec["reuse"], spec["noise"], spec.get("instance", 0))
    if k == "srbench":
        return srbench_task(spec["name"], split_seed=seed)
    raise ValueError(f"unknown task kind {k}")


def _make_cfg(cls, params: dict):
    names = {f.name for f in fields(cls)}
    p = dict(params or {})
    unknown = set(p) - names
    if unknown:
        raise ValueError(f"unknown arm params {unknown}")
    for f in fields(cls):
        if f.name in p and isinstance(p[f.name], list):
            p[f.name] = tuple(p[f.name])
    return cls(**p)


def build_arm(name: str, task: Task, params: dict):
    if name in ("a1", "a2"):
        cfg = _make_cfg(TreeArmConfig, params)
        return make_a1(task.pset, cfg) if name == "a1" else make_a2(task.pset, cfg)
    if name in HIER_VARIANTS:
        return make_hier(name, task.pset, _make_cfg(HierArmConfig, params))
    raise ValueError(f"unknown arm {name}")


# -------------------------------------------------------------- evaluation --
def fitness(prog, X, y, kind):
    """Return (per-case errors, scalar fitness, scaling). Fitness is the number
    of wrong cases (boolean) or MSE after linear scaling (regression)."""
    with np.errstate(all="ignore"):
        f = prog.evaluate(X)
        if kind == "boolean":
            err = (f != y).astype(float)
            return err, float(err.sum()), None
        a, b = linear_scale(f, y)
        err = np.abs(y - scaled_predictions(f, a, b))
    err[~np.isfinite(err)] = BIG
    err = np.minimum(err, BIG)
    return err, float(np.mean(err ** 2)), (a, b)


def apply_model(prog, scaling, X):
    with np.errstate(all="ignore"):
        return scaled_predictions(prog.evaluate(X), *scaling)


def mse(pred, y):
    with np.errstate(all="ignore"):
        v = float(np.mean((y - pred) ** 2))
    return v if np.isfinite(v) else BIG


def r2(pred, y):
    return 1.0 - mse(pred, y) / float(np.var(y))


# --------------------------------------------------------------------- run --
def versions() -> dict:
    def git(*a):
        try:
            return subprocess.check_output(["git", *a], cwd=Path(__file__).parent, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    from importlib.metadata import PackageNotFoundError, version

    def ver(pkg):
        try:
            return version(pkg)
        except PackageNotFoundError:
            return None

    return {
        "python": platform.python_version(),
        **{pkg: ver(pkg) for pkg in ("numpy", "pandas", "scipy", "pyarrow", "pyyaml", "pmlb")},
        "platform": platform.platform(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
    }


class Run:
    def __init__(self, cfg: dict, out_dir: str | Path | None = None):
        self.cfg = deep_merge(DEFAULTS, cfg)
        if self.cfg["pop_size"] == "auto":
            self.cfg["pop_size"] = population_for_budget(self.cfg["budget"])
        self.seed = seed_for(self.cfg["seed_index"])
        self.task = build_task(self.cfg["task"], self.seed)
        self.arm = build_arm(self.cfg["arm"], self.task, self.cfg["arm_params"])
        self.id = run_id(self.cfg)
        self.out = Path(out_dir) / self.id if out_dir else None
        self.kind = self.task.kind

    # selection
    def select(self, errors, fit, n, rng):
        s = self.cfg["selection"]
        if s == "tournament":
            return tournament_select(fit, n, rng, self.cfg["tournament_size"])
        if s == "lexicase" or (s == "auto" and self.kind == "boolean"):
            return lexicase_select(errors, n, rng)
        if s in ("epsilon_lexicase", "auto"):
            return epsilon_lexicase_select(errors, n, rng)
        raise ValueError(s)

    def execute(self) -> dict:
        cfg, task, arm = self.cfg, self.task, self.arm
        ck = self.out / "ckpt.pkl" if self.out else None
        if ck and ck.exists():
            st = pickle.loads(ck.read_bytes())
            rng = np.random.default_rng()
            rng.bit_generator.state = st["rng"]
            counter = ComputeCounter.from_state(st["counter"])
            gen, pop, errs, fits, scal = st["gen"], st["pop"], st["errs"], st["fits"], st["scal"]
            S, log = st["S"], st["log"]
        else:
            rng = make_rng(self.seed, 1)
            counter = ComputeCounter(cfg["budget"])
            S = {"best_fit": np.inf, "best_prog": None, "best_scal": None,
                 "champ_val": np.inf, "champ": None, "champ_scal": None,
                 "solved": False, "units_to_solution": None, "gen_to_solution": None,
                 "exhausted": False}
            log = []
            gen = 0
            pop = arm.init_population(cfg["pop_size"], rng)
            pop, errs, fits, scal = self._evaluate(pop, counter, S, gen)
            if not pop:
                raise ValueError(f"budget {cfg['budget']} cannot pay for a single program evaluation")
            self._log(log, gen, pop, fits, scal, counter, S)

        N = cfg["pop_size"]
        while (not S["exhausted"] and gen + 1 < cfg["max_generations"]
               and not (S["solved"] and cfg["stop_on_solution"])):
            gen += 1
            E = np.vstack(errs)
            idx = self.select(E, np.asarray(fits), 2 * N, rng)
            children = [arm.vary([pop[idx[2 * i]], pop[idx[2 * i + 1]]], rng) for i in range(N)]
            children, cerrs, cfits, cscal = self._evaluate(children, counter, S, gen)
            if len(children) < N:  # budget ran out mid-generation
                if children:
                    self._log(log, gen, children, cfits, cscal, counter, S, partial=True)
                break
            pop, errs, fits, scal = children, cerrs, cfits, cscal
            self._log(log, gen, pop, fits, scal, counter, S)
            if ck and cfg["checkpoint_every"] and gen % cfg["checkpoint_every"] == 0:
                ck.parent.mkdir(parents=True, exist_ok=True)
                tmp = ck.with_suffix(".tmp")
                tmp.write_bytes(pickle.dumps({
                    "rng": rng.bit_generator.state, "counter": counter.state(), "gen": gen, "pop": pop,
                    "errs": errs, "fits": fits, "scal": scal, "S": S, "log": log}))
                os.replace(tmp, ck)

        final = self._final(S, counter, gen)
        if self.out:
            self.out.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(log).to_parquet(self.out / "generations.parquet", index=False)
            row = {k: v for k, v in final.items() if not isinstance(v, (dict, list))}
            pd.DataFrame([row]).to_parquet(self.out / "final.parquet", index=False)
            (self.out / "final.json").write_text(json.dumps(final, indent=2, default=str))
            if ck and ck.exists():
                ck.unlink()
        self.log = log
        return final

    def _evaluate(self, progs, counter, S, gen):
        """Evaluate programs in order, charging each; stop at the budget."""
        task = self.task
        n = task.n_train
        out_p, out_e, out_f, out_s = [], [], [], []
        for p in progs:
            cost = counter.cost(p, n)
            if not counter.can_afford(cost):
                S["exhausted"] = True
                break
            counter.charge(cost)
            e, f, s = fitness(p, task.X_train, task.y_train, self.kind)
            out_p.append(p); out_e.append(e); out_f.append(f); out_s.append(s)
            if f < S["best_fit"]:
                S["best_fit"], S["best_prog"], S["best_scal"] = f, p, s
            if self.kind == "boolean" and f == 0 and not S["solved"]:
                S["solved"], S["units_to_solution"], S["gen_to_solution"] = True, counter.units, gen
                S["evals_to_solution"] = counter.evals
                if self.cfg["stop_on_solution"]:
                    S["exhausted"] = False
                    return out_p, out_e, out_f, out_s
        # Model selection on validation: the generation's training-best program.
        if self.kind == "regression" and out_p and task.X_val is not None:
            i = int(np.argmin(out_f))
            counter.log_validation(counter.cost(out_p[i], len(task.y_val)))
            v = mse(apply_model(out_p[i], out_s[i], task.X_val), task.y_val)
            if v < S["champ_val"]:
                S["champ_val"], S["champ"], S["champ_scal"] = v, out_p[i], out_s[i]
        return out_p, out_e, out_f, out_s

    def _log(self, log, gen, pop, fits, scal, counter, S, partial=False):
        sizes = np.array([p.expanded_size() for p in pop])
        stored = np.array([p.stored_size() for p in pop])
        depths = np.array([p.depth() for p in pop])
        i = int(np.argmin(fits))
        log.append({
            "gen": gen, "partial": partial, "units": counter.units, "evals": counter.evals,
            "val_units": counter.val_units, "wall": counter.wall,
            "best_train": float(fits[i]), "median_train": float(np.median(fits)),
            "best_size": int(sizes[i]), "mean_size": float(sizes.mean()), "max_size": int(sizes.max()),
            "mean_stored": float(stored.mean()), "mean_depth": float(depths.mean()),
            "n_unique": len({p.key() for p in pop}),
            "champion_val": float(S["champ_val"]) if np.isfinite(S["champ_val"]) else None,
            "solved": S["solved"],
        })

    def _final(self, S, counter, gen) -> dict:
        task, cfg = self.task, self.cfg
        out = {
            "run_id": self.id, "arm": cfg["arm"], "task": task.name, "budget": cfg["budget"],
            "seed_index": cfg["seed_index"], "seed": self.seed, "pop_size": cfg["pop_size"],
            "generations": gen, "units": counter.units, "evals": counter.evals,
            "val_units": counter.val_units, "val_evals": counter.val_evals, "wall": counter.wall,
            "best_train": S["best_fit"],
            **getattr(self.arm, "init_stats", {}),
        }
        if self.kind == "boolean":
            p = S["best_prog"]
            out.update({"solved": S["solved"], "units_to_solution": S["units_to_solution"],
                        "gen_to_solution": S["gen_to_solution"],
                        "evals_to_solution": S.get("evals_to_solution")})
        else:
            p, sc = S["champ"], S["champ_scal"]
            if p is None:  # no validation set: fall back to training best
                p, sc = S["best_prog"], S["best_scal"]
            tr = apply_model(p, sc, task.X_train)
            te = apply_model(p, sc, task.X_test)
            out.update({"train_mse": mse(tr, task.y_train), "train_r2": r2(tr, task.y_train),
                        "val_mse": S["champ_val"], "test_mse": mse(te, task.y_test),
                        "test_r2": r2(te, task.y_test), "scale_a": sc[0], "scale_b": sc[1]})
        out.update({"model_size": p.expanded_size(), "model_stored": p.stored_size(),
                    "model_depth": p.depth(), "model": str(p),
                    "config": cfg, "arm_desc": self.arm.describe(), "task_meta": task.meta,
                    "versions": versions()})
        return out


def run_config(cfg: dict, out_dir=None) -> dict:
    return Run(cfg, out_dir).execute()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--index", type=int, default=None, help="run index within a batch")
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    raw = load_yaml(a.config)
    if "grid" in raw or "base" in raw:
        runs = expand_batch(raw)
        idx = a.index if a.index is not None else int(os.environ.get("TASK_INDEX", "0"))
        cfg = runs[idx]
    else:
        cfg = raw
    res = run_config(cfg, a.out)
    keys = ["run_id", "units", "evals", "wall", "best_train", "solved", "test_r2"]
    print(json.dumps({k: res.get(k) for k in keys}, default=str))


if __name__ == "__main__":
    sys.exit(main())

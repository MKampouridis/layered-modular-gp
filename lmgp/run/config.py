"""Config handling. A run is (arm, task, budget, seed). A batch file lists
values for any key; the batch expands to the Cartesian product, ordered
deterministically, so an array-job index maps to exactly one run."""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
from pathlib import Path

import yaml

DEFAULTS = {
    "arm": "a1",
    "arm_params": {},
    "task": {"kind": "parity", "n": 4},
    "budget": None,  # node evaluations; None = unlimited (then max_generations binds)
    "pop_size": 500,
    "max_generations": 100,
    "selection": "auto",  # auto: lexicase (boolean) / epsilon-lexicase (regression); or "tournament"
    "tournament_size": 7,
    "seed_index": 0,
    "checkpoint_every": 0,  # generations; 0 disables
    "stop_on_solution": True,  # boolean tasks only
}


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path) -> dict:
    return yaml.safe_load(Path(path).read_text()) or {}


def expand_batch(batch: dict) -> list[dict]:
    """batch = {"base": {...}, "grid": {"dotted.key": [values...]}, "seeds": N or [indices]}."""
    base = deep_merge(DEFAULTS, batch.get("base", {}))
    grid = batch.get("grid", {})
    seeds = batch.get("seeds", 1)
    seed_list = list(range(seeds)) if isinstance(seeds, int) else list(seeds)
    keys = list(grid)
    runs = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        cfg = copy.deepcopy(base)
        for k, v in zip(keys, combo):
            d = cfg
            parts = k.split(".")
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = copy.deepcopy(v)
        for s in seed_list:
            c = copy.deepcopy(cfg)
            c["seed_index"] = s
            runs.append(c)
    return runs


def task_label(task: dict) -> str:
    k = task["kind"]
    if k == "parity":
        return f"parity{task['n']}"
    if k == "mux":
        return f"mux{task['k'] + 2 ** task['k']}"
    if k == "comp":
        return f"comp_L{task['levels']}_R{task['reuse']}_n{task['noise']:g}_i{task.get('instance', 0)}"
    if k == "srbench":
        return f"srb_{task['name']}"
    raise ValueError(k)


def run_id(cfg: dict) -> str:
    """Readable, unique id: arm, task, budget, seed plus a short hash of the
    full config."""
    h = hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:8]
    b = "inf" if cfg.get("budget") is None else f"{float(cfg['budget']):.0e}".replace("+", "")
    return f"{cfg['arm']}__{task_label(cfg['task'])}__b{b}__s{cfg['seed_index']:03d}__{h}"

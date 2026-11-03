"""Campaign safeguards from the pre-CERES review (3 Nov 2026): tuning tasks
kept out of screening, distinct tuning configurations, completeness checks,
crash-safe outputs and the run lock."""
import importlib.util
import os
import time
from pathlib import Path

import pandas as pd
import pytest
import yaml

from lmgp.run.config import expand_batch, run_id, task_label
from lmgp.run.tuning import ARMS, TUNING_TASKS, configurations, effective

ROOT = Path(__file__).resolve().parents[1]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_no_tuning_task_is_screened():
    scr = _script("make_screening")
    screened = {task_label(t) for t in scr.stage1_tasks()}
    assert not screened & {task_label(t) for t in TUNING_TASKS}


def test_every_arm_gets_20_distinct_configurations():
    for arm in ARMS:
        c = configurations(arm)
        assert len(c) == 20 and len({effective(arm, p) for p in c}) == 20, arm


def test_tuning_checkpoints():
    batch = yaml.safe_load((ROOT / "configs" / "wp2" / "tuning.yaml").read_text())
    assert batch["base"]["checkpoint_every"] > 0


def _fake_screening(tmp_path, monkeypatch, scr, keep):
    """A tiny screening batch with fake results for the first `keep` runs."""
    out = tmp_path / "wp2"
    out.mkdir()
    batch = {"base": {"pop_size": "auto", "max_generations": 10},
             "cases": [{"arm": "a1"}, {"arm": "a5"}],
             "grid": {"budget": scr.BUDGETS, "task": [{"kind": "parity", "n": 6}, {"kind": "mux", "k": 3}]},
             "seeds": 6}
    (out / "screening.yaml").write_text(yaml.safe_dump(batch))
    (out / "tuned.yaml").write_text(yaml.safe_dump({"g2_arm": "a5"}))
    monkeypatch.setattr(scr, "OUT", out)
    res = tmp_path / "res"
    runs = [c for c in expand_batch(batch)]
    for c in runs[:keep]:
        d = res / run_id(c)
        d.mkdir(parents=True)
        a5 = c["arm"] == "a5"
        pd.DataFrame([{"best_train": 0.0 if a5 else 1.0, "solved": a5,
                       "units_to_solution": 1e6 if a5 else None}]).to_parquet(d / "final.parquet")
    return res, len(runs)


def test_g2_refuses_incomplete_results(tmp_path, monkeypatch):
    scr = _script("make_screening")
    res, n = _fake_screening(tmp_path, monkeypatch, scr, keep=5)
    with pytest.raises(SystemExit, match="incomplete"):
        scr.g2(str(res), None)
    _, verdict = scr.g2(str(res), None, allow_partial=True)
    assert verdict is None  # INCOMPLETE, never PASS


def test_g2_denominator_is_all_planned_tasks(tmp_path, monkeypatch):
    scr = _script("make_screening")
    res, n = _fake_screening(tmp_path, monkeypatch, scr, keep=10 ** 9)
    table, verdict = scr.g2(str(res), None)
    assert verdict is True and set(table.task) == {"parity6", "mux11"}


def test_tuning_report_refuses_incomplete_results(tmp_path, monkeypatch):
    tun = _script("make_tuning")
    monkeypatch.setattr(tun, "OUT", tmp_path)
    (tmp_path / "tuning.yaml").write_text(yaml.safe_dump(
        {"base": {"budget": 1e6}, "cases": [{"arm": "a1", "tuning_config": 0}],
         "grid": {"task": [{"kind": "parity", "n": 5}]}, "seeds": [100, 101]}))
    with pytest.raises(SystemExit, match="incomplete"):
        tun.report(str(tmp_path / "none"))
    assert not (tmp_path / "tuned.yaml").exists()


def test_run_lock_and_atomic_outputs(tmp_path):
    from lmgp.run.runner import LOCK_STALE_S, Run, RunLocked

    cfg = {"arm": "a1", "task": {"kind": "parity", "n": 3}, "pop_size": 20, "max_generations": 3}
    r = Run(cfg, tmp_path)
    r.out.mkdir(parents=True)
    lock = r.out / "RUNNING"
    lock.write_text("other host\n")
    with pytest.raises(RunLocked):
        Run(cfg, tmp_path).execute()  # a live lock: skip, do not run twice
    old = time.time() - LOCK_STALE_S - 10
    os.utime(lock, (old, old))
    res = Run(cfg, tmp_path).execute()  # a stale lock is taken over
    assert not lock.exists()
    assert {p.name for p in r.out.iterdir()} == {"generations.parquet", "final.json", "final.parquet"}
    row = pd.read_parquet(r.out / "final.parquet").iloc[0]
    assert row["git_commit"] == res["versions"]["git_commit"]


def test_g2_boolean_score_rewards_solving_with_less_compute():
    """Antigravity's case: both arms solve every run, A5 with 10x less
    compute. Scoring final error alone gave a tie (no win possible)."""
    scr = _script("make_screening")
    cfg = {"task": {"kind": "parity", "n": 6}, "budget": 1e10}
    a5 = [scr.run_score({"solved": True, "units_to_solution": 1e8, "best_train": 0.0}, cfg)] * 10
    a1 = [scr.run_score({"solved": True, "units_to_solution": 1e9, "best_train": 0.0}, cfg)] * 10
    from scipy.stats import mannwhitneyu
    assert mannwhitneyu(a5, a1, alternative="less").pvalue < 0.01
    # any solved run beats any unsolved run; unsolved runs ordered by error
    solved = scr.run_score({"solved": True, "units_to_solution": 1e10, "best_train": 0.0}, cfg)
    near = scr.run_score({"solved": False, "units_to_solution": None, "best_train": 1.0}, cfg)
    far = scr.run_score({"solved": False, "units_to_solution": None, "best_train": 9.0}, cfg)
    assert solved < near < far


def test_mse_never_prefers_non_finite_predictions():
    import numpy as np
    from lmgp.run.runner import mse

    y = np.zeros(10)
    finite_huge = mse(np.full(10, 1e20), y)
    assert mse(np.full(10, np.nan), y) >= finite_huge
    assert mse(np.full(10, np.inf), y) >= finite_huge
    assert mse(np.full(10, 1e160), y) >= finite_huge


def test_offspring_differ_from_parents_in_active_code():
    """Shared rule: an offspring identical to a parent in active code is
    retried (only the max_tries fallback may copy a parent)."""
    import numpy as np
    from lmgp.core.primitives import parity_set, regression_set
    from lmgp.run.runner import build_arm
    from types import SimpleNamespace

    for arm_name in ["a1", "a2", "a3", "a4", "a5", "a6", "a6_inputs", "a5_copy", "a5_frozen"]:
        for pset in (parity_set(5), regression_set(3)):
            arm = build_arm(arm_name, SimpleNamespace(pset=pset), {})
            rng = np.random.default_rng(4)
            pop = arm.init_population(40, rng)
            same = 0
            for _ in range(300):
                i, j = rng.integers(len(pop), size=2)
                c = arm.vary([pop[i], pop[j]], rng)
                same += c is pop[i]  # the fallback returns the parent object itself
                if c is not pop[i]:
                    assert c.active_key not in (pop[i].active_key, pop[j].active_key), arm_name
            # (fallback copies are rare in evolved populations, 0-6% per arm, but
            # common for some random parents, e.g. A2 whose ADFs are never called)


def test_cgp_constant_charged_per_read():
    import numpy as np
    from lmgp.arms.a3_cgp import make_a3
    from lmgp.core.primitives import regression_set
    from lmgp.core.reference import reference_eval_graph

    arm = make_a3("a3", regression_set(1), rows=1, cols=2, levels_back=2, n_const=1)
    add = next(i for i, f in enumerate(arm.pset.functions) if f.name == "add")
    funcs = [add, add]
    conns = [(("const", 0), ("const", 0)), (("node", 0), ("const", 0))]
    p = arm.build(((0.5,), tuple(funcs), tuple(conns), ("node", 1)))
    # add(add(c, c), c): three constant reads, two additions, as in a tree
    assert p.cost_per_case() == 5 == p.expanded_size()
    X = np.array([[0.0], [1.0]])
    out, cnt = reference_eval_graph(p, X)
    assert cnt == 10 and np.allclose(out, 1.5)


def test_a4_tuning_keeps_published_proportions():
    for p in configurations("a4")[1:]:
        assert p["input_trees"] == p["window"] == p["n_trees"] // 2


def test_generation_count_at_budget_boundary():
    from lmgp.run.runner import Run

    cfg = {"arm": "a1", "task": {"kind": "parity", "n": 3}, "pop_size": 20, "max_generations": 10 ** 6,
           "stop_on_solution": False, "budget": 2e5}
    r = Run(cfg)
    res = r.execute()
    evaluated = {row["gen"] for row in r.log}
    assert res["generations"] == max(evaluated)  # never a generation with no evaluation

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
        pd.DataFrame([{"best_train": 0.0 if c["arm"] == "a5" else 1.0}]).to_parquet(d / "final.parquet")
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

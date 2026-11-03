"""Unit tests for selection, linear scaling and the benchmarks."""
import numpy as np

from lmgp.bench.boolean import even_parity, multiplexer
from lmgp.bench.synthetic import compositional_task, evaluate_structure
from lmgp.core.scaling import linear_scale
from lmgp.selection.lexicase import epsilon_lexicase_select, lexicase_select, mad_epsilon


# ---- selection ---------------------------------------------------------------
def test_lexicase_always_picks_dominant_individual():
    rng = np.random.default_rng(0)
    E = rng.integers(1, 5, size=(30, 10)).astype(float)
    E[7] = 0.0
    assert set(lexicase_select(E, 200, rng)) == {7}


def test_lexicase_specialists_each_selected():
    # Each individual is uniquely best on one case: all should be selected.
    E = np.ones((5, 5)) - np.eye(5)
    sel = lexicase_select(E, 2000, np.random.default_rng(1))
    counts = np.bincount(sel, minlength=5)
    assert counts.min() > 300


def test_lexicase_ties_uniform_over_individuals():
    E = np.zeros((4, 3))
    E[3] = 1.0  # three identical elites, one loser
    sel = lexicase_select(E, 6000, np.random.default_rng(2))
    counts = np.bincount(sel, minlength=4)
    assert counts[3] == 0 and counts[:3].min() > 1800


def test_epsilon_lexicase_semi_dynamic():
    E = np.array([[0.0, 5.0], [0.05, 0.0], [10.0, 10.0], [9.0, 9.0], [8.0, 8.0]])
    eps = mad_epsilon(E)
    assert eps.shape == (2,)
    sel = epsilon_lexicase_select(E, 2000, np.random.default_rng(3))
    # individual 1 is within epsilon of the best on case 0 and best on case 1
    assert np.bincount(sel, minlength=5)[1] > np.bincount(sel, minlength=5)[0]
    assert set(sel) <= {0, 1}


# ---- linear scaling ------------------------------------------------------------
def test_linear_scaling_recovers_affine_map():
    rng = np.random.default_rng(4)
    f = rng.normal(size=100)
    a, b = linear_scale(f, 3.0 - 2.0 * f)
    assert np.isclose(a, 3.0) and np.isclose(b, -2.0)


def test_linear_scaling_constant_and_nonfinite():
    y = np.arange(5.0)
    assert linear_scale(np.ones(5), y) == (2.0, 0.0)
    assert linear_scale(np.array([1, np.inf, 2, 3, 4.0]), y) == (2.0, 0.0)


# ---- benchmarks ----------------------------------------------------------------
def test_parity_truth_table():
    t = even_parity(4)
    assert t.X_train.shape == (16, 4)
    assert t.y_train.sum() == 8
    assert all(t.y_train[i] == (t.X_train[i].sum() % 2 == 0) for i in range(16))


def test_multiplexer_known_solution():
    t = multiplexer(2)  # 6-mux: a0 a1 d0 d1 d2 d3
    X = t.X_train
    a0, a1 = X[:, 0], X[:, 1]
    sol = np.where(a0, np.where(a1, X[:, 5], X[:, 4]), np.where(a1, X[:, 3], X[:, 2]))
    assert np.array_equal(sol, t.y_train)
    assert multiplexer(3).X_train.shape == (2048, 11)


def test_compositional_task_ground_truth_and_reproducibility():
    t = compositional_task(3, 2, 0.1, instance=0)
    s = t.meta["structure"]
    assert len(s["units"]) == 6 and len(set(s["subfunctions_used"])) == 3
    np.testing.assert_allclose(evaluate_structure(s, t.X_test), t.y_test)  # test is noise-free
    t2 = compositional_task(3, 2, 0.1, instance=0)
    assert np.array_equal(t.y_train, t2.y_train)
    t0 = compositional_task(3, 2, 0.0, instance=0)
    assert np.array_equal(t0.X_train, t.X_train)  # noise levels share inputs and structure
    resid = t.y_train - t0.y_train
    assert 0.05 < resid.std() / t0.y_train.std() < 0.15
    # every unit of each level below the top feeds the next level
    for lvl in (1, 2):
        ids = {u["id"] for u in s["units"] if u["level"] == lvl}
        used = {a for u in s["units"] if u["level"] == lvl + 1 for a in u["args"]}
        assert ids <= used


def test_compositional_targets_nondegenerate():
    for L in range(1, 5):
        for R in range(1, 5):
            t = compositional_task(L, R, 0.0)
            assert np.std(t.y_train) > 0.3


# ---- computational effort --------------------------------------------------------
def test_koza_effort():
    from lmgp.analysis.effort import effort

    assert effort([5, 5, 5]) == 5
    assert effort([100] * 5 + [None] * 5) == 700  # P = 0.5 -> R = 7
    assert effort([None, None]) == float("inf")
    # min over thresholds: 2 runs solved early, 8 late
    assert effort([10, 10] + [100] * 8) == min(10 * 21, 100 * 1)


def test_task_spec_replaced_not_merged():
    from lmgp.run.config import expand_batch, run_id
    from lmgp.run.runner import Run

    runs = expand_batch({"grid": {"task": [{"kind": "mux", "k": 2}]}, "seeds": 1})
    assert runs[0]["task"] == {"kind": "mux", "k": 2}
    assert Run(runs[0]).id == run_id(runs[0])


def test_tuning_configs_and_ranking():
    import pandas as pd

    from lmgp.run.config import expand_batch
    from lmgp.run.tuning import ARMS, configurations, rank_configs, tuning_batch

    for arm in ARMS:
        c = configurations(arm)
        assert len(c) == 20 and c[0] == {} and c == configurations(arm)  # default first, deterministic
    runs = expand_batch(tuning_batch())
    assert len(runs) == 9 * 20 * 4 * 5
    df = pd.DataFrame([
        {"arm": "a1", "task": "t", "kind": "regression", "tuning_config": 0, "val_mse": 2.0},
        {"arm": "a1", "task": "t", "kind": "regression", "tuning_config": 1, "val_mse": 1.0},
        {"arm": "a1", "task": "p", "kind": "boolean", "tuning_config": 0, "best_train": 0.0,
         "units_to_solution_or_budget": 5.0},
        {"arm": "a1", "task": "p", "kind": "boolean", "tuning_config": 1, "best_train": 0.0,
         "units_to_solution_or_budget": 9.0},
    ])
    r = rank_configs(df).set_index("tuning_config")["mean_rank"]
    assert r[0] == r[1] == 1.5  # 1 wins regression, 0 wins parity (same error, fewer units)

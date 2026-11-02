"""G1 property tests: offspring are valid, size and depth limits hold, and a
run is deterministic for a given seed."""
import numpy as np
from hypothesis import given, settings, strategies as st

from lmgp.core.program import TreeProgram
from lmgp.core.tree_ops import generate, is_well_formed, subtree_end
from lmgp.run.runner import run_config

from conftest import ARMS


def well_formed(arm, p: TreeProgram) -> bool:
    return all(is_well_formed(arm.arch, b, p.codes[b]) for b in range(arm.arch.n_branches)) and all(
        len(c) == len(v) for c, v in zip(p.codes, p.vals))


def test_initial_population_valid(arm):
    rng = np.random.default_rng(1)
    pop = arm.init_population(200, rng)
    assert len(pop) == 200
    for p in pop:
        assert well_formed(arm, p)
        assert p.expanded_size() <= arm.cfg.max_size
        assert p.depth() <= arm.cfg.max_depth
    # duplicates are rejected where possible
    assert len({p.key() for p in pop}) >= 190


def test_offspring_valid_over_many_generations(arm):
    """Breed for many rounds with random parents; every offspring must be a
    well-formed program within the size and depth caps."""
    rng = np.random.default_rng(2)
    pop = arm.init_population(60, rng)
    for _ in range(30):
        new = []
        for _ in range(len(pop)):
            i, j = rng.integers(len(pop), size=2)
            c = arm.vary([pop[i], pop[j]], rng)
            assert well_formed(arm, c)
            assert c.expanded_size() <= arm.cfg.max_size
            assert c.depth() <= arm.cfg.max_depth
            new.append(c)
        pop = new


def test_operators_produce_change(arm):
    """Variation is not a no-op: most offspring differ from the first parent."""
    rng = np.random.default_rng(3)
    pop = arm.init_population(100, rng)
    changed = sum(arm.vary([pop[i], pop[(i + 1) % 100]], rng).key() != pop[i].key() for i in range(100))
    assert changed >= 70


def test_size_cap_enforced_under_pressure():
    """With a tiny size cap the operators must still never exceed it."""
    from lmgp.arms.a2_adf import make_a2
    from lmgp.arms.tree import TreeArmConfig
    from lmgp.core.primitives import parity_set

    arm = make_a2(parity_set(5), TreeArmConfig(max_size=40, max_depth=6, init_depth=(1, 3)))
    rng = np.random.default_rng(4)
    pop = arm.init_population(50, rng)
    for _ in range(500):
        i, j = rng.integers(50, size=2)
        c = arm.vary([pop[i], pop[j]], rng)
        assert c.expanded_size() <= 40 and c.depth() <= 6
        pop[i] = c


@settings(max_examples=200, deadline=None)
@given(seed=st.integers(0, 2 ** 32 - 1), depth=st.integers(0, 6), arm_i=st.integers(0, len(ARMS) - 1),
       full=st.booleans())
def test_generate_respects_depth_and_grammar(seed, depth, arm_i, full):
    arm = ARMS[arm_i][1]
    rng = np.random.default_rng(seed)
    for b in range(arm.arch.n_branches):
        codes, vals = generate(arm.arch, b, depth, "full" if full else "grow", rng)
        assert is_well_formed(arm.arch, b, codes)
        assert subtree_end(codes, 0, arm.arch.arity) == len(codes)
    p = arm.random_program(depth, "full" if full else "grow", rng)
    if arm.arch.n_branches == 1:
        assert p.depth() <= depth
        if full:
            assert p.depth() == depth


def _strip_wall(log):
    return [{k: v for k, v in row.items() if k != "wall"} for row in log]


def _run(cfg):
    from lmgp.run.runner import Run

    r = Run(cfg)
    final = r.execute()
    return final, _strip_wall(r.log)


def test_run_is_deterministic_boolean():
    cfg = {"arm": "a2", "task": {"kind": "parity", "n": 3}, "pop_size": 80, "max_generations": 8,
           "seed_index": 5, "stop_on_solution": False}
    f1, l1 = _run(cfg)
    f2, l2 = _run(cfg)
    assert l1 == l2
    assert f1["model"] == f2["model"] and f1["units"] == f2["units"]


def test_run_is_deterministic_regression():
    cfg = {"arm": "a1", "task": {"kind": "comp", "levels": 2, "reuse": 1, "noise": 0.1},
           "pop_size": 60, "max_generations": 6, "seed_index": 3}
    f1, l1 = _run(cfg)
    f2, l2 = _run(cfg)
    assert l1 == l2
    assert f1["model"] == f2["model"] and f1["test_mse"] == f2["test_mse"]


def test_different_seeds_differ():
    base = {"arm": "a1", "task": {"kind": "parity", "n": 3}, "pop_size": 50, "max_generations": 3,
            "stop_on_solution": False}
    f1, _ = _run({**base, "seed_index": 0})
    f2, _ = _run({**base, "seed_index": 1})
    assert f1["model"] != f2["model"]


def test_checkpoint_resume_matches_uninterrupted(tmp_path):
    """Kill a run after a checkpoint, resume it, and get the same result as an
    uninterrupted run."""
    import pickle

    from lmgp.run.runner import Run

    cfg = {"arm": "a1", "task": {"kind": "comp", "levels": 1, "reuse": 2, "noise": 0.0},
           "pop_size": 40, "max_generations": 6, "seed_index": 2, "checkpoint_every": 2}
    full = Run(cfg, tmp_path / "a").execute()

    # Produce a checkpoint at generation 4 by running with max_generations=5
    # but keeping the checkpoint file (simulate a crash after gen 4).
    r = Run({**cfg, "max_generations": 5}, tmp_path / "b")
    orig_final = r._final
    r._final = lambda *a: (_ for _ in ()).throw(KeyboardInterrupt)  # crash before cleanup
    try:
        r.execute()
    except KeyboardInterrupt:
        pass
    ck = tmp_path / "b" / r.id / "ckpt.pkl"
    assert ck.exists() and pickle.loads(ck.read_bytes())["gen"] == 4
    # Resume under the full config, pointing at the same checkpoint.
    r2 = Run(cfg, tmp_path / "b")
    r2.id = r.id
    r2.out = tmp_path / "b" / r.id
    resumed = r2.execute()
    for k in ("model", "units", "evals", "test_mse", "generations"):
        assert resumed[k] == full[k], k

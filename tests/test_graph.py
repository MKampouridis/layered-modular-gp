"""WP2 tests for GraphProgram and the hierarchical arms (A5, A6, A6-inputs,
A5-copy, A5-frozen): counter semantics, inlined size, operator validity and
determinism."""
import numpy as np
import pytest

from lmgp.arms.hier import VARIANTS, HierArmConfig, make_hier
from lmgp.core.counter import ComputeCounter
from lmgp.core.graph import ERC, REF, GraphProgram, input_ref
from lmgp.core.primitives import multiplexer_set, parity_set, regression_set
from lmgp.core.reference import reference_eval_graph

PSETS = {"regression": lambda: regression_set(3), "parity": lambda: parity_set(4),
         "mux": lambda: multiplexer_set(6)}
CFG = HierArmConfig(max_size=200, max_depth=12)
HARMS = [(f"{v}-{p}", make_hier(v, mk(), CFG)) for v in VARIANTS for p, mk in PSETS.items()]


@pytest.fixture(params=HARMS, ids=[a[0] for a in HARMS])
def harm(request):
    return request.param[1]


def data(pset, n=23, seed=0):
    rng = np.random.default_rng(seed)
    if pset.dtype is bool:
        return rng.random((n, pset.n_vars)) < 0.5
    return rng.uniform(-3, 3, size=(n, pset.n_vars))


def bred(arm, n=60, gens=4, seed=0):
    rng = np.random.default_rng(seed)
    pop = arm.init_population(n, rng)
    for _ in range(gens):
        pop = [arm.vary([pop[i], pop[(i + 7) % n]], rng) for i in range(n)]
    return pop


# ---- counter -------------------------------------------------------------------------
def _add_mm(cached):
    # M0 = mul(x0, x1); out = add(M0, M0)
    ps = regression_set(3)
    add, mul = 0, 2
    return GraphProgram(ps, [1], [[mul, REF, REF], [add, REF, REF]],
                        [[0, input_ref(0), input_ref(1)], [0, 0, 0]], cached=cached)


def test_cached_module_charged_once_per_evaluation():
    p = _add_mm(True)
    assert p.cost_per_case() == 3 + 3  # module once, plus the output tree (add + two reads)
    assert ComputeCounter.cost(p, 10) == 60
    assert reference_eval_graph(p, data(p.pset, 10))[1] == 60


def test_copy_mode_charges_every_read():
    p = _add_mm(False)
    assert p.cost_per_case() == 3 + 2 * 3
    assert reference_eval_graph(p, data(p.pset, 10))[1] == 90


def test_inlined_size_and_depth_hand_example():
    p = _add_mm(True)
    assert p.expanded_size() == 7 and p.depth() == 2  # add(mul(x0,x1), mul(x0,x1))
    assert p.stored_size() == 6


def test_inactive_modules_cost_nothing():
    ps = regression_set(3)
    # M0 is never read: out = sin(x2)
    p = GraphProgram(ps, [1], [[2, REF, REF], [4, REF]], [[0, input_ref(0), input_ref(1)], [0, input_ref(2)]])
    assert p.active == (False,) and p.cost_per_case() == 2


def test_cost_and_outputs_match_reference(harm):
    X = data(harm.pset)
    for p in bred(harm):
        out_ref, count = reference_eval_graph(p, X)
        assert count == p.cost_per_case() * X.shape[0]
        out = p.evaluate(X)
        if harm.pset.dtype is bool:
            assert np.array_equal(out, out_ref)
        else:
            np.testing.assert_allclose(out, out_ref, rtol=1e-9, atol=1e-12, equal_nan=True)


def test_copy_equals_cached_semantics_and_costs_at_least_as_much():
    arm = make_hier("a5", regression_set(3), CFG)
    X = data(arm.pset)
    for p in bred(arm):
        q = GraphProgram(p.pset, p.levels, p.codes, p.args, cached=False)
        np.testing.assert_allclose(p.evaluate(X), q.evaluate(X), equal_nan=True)
        assert q.cost_per_case() >= p.cost_per_case()
        assert q.expanded_size() == p.expanded_size()


def test_identical_programs_cost_the_same(harm):
    for p in bred(harm, gens=2):
        q = GraphProgram(p.pset, list(p.levels), [list(c) for c in p.codes], [list(a) for a in p.args], p.cached)
        assert q is not p and q == p
        assert q.cost_per_case() == p.cost_per_case() and q.expanded_size() == p.expanded_size()


def _inline(p, b=None):
    """Explicit textual inlining: returns the prefix codes/args of the plain tree."""
    b = p.out if b is None else b
    codes, args = [], []
    for c, a in zip(p.codes[b], p.args[b]):
        if c == REF and a >= 0:
            cc, aa = _inline(p, a)
            codes += cc
            args += aa
        else:
            codes.append(c)
            args.append(a)
    return codes, args


def test_expanded_size_and_depth_equal_explicit_inlining(harm):
    X = data(harm.pset)
    for p in bred(harm):
        c, a = _inline(p)
        q = GraphProgram(p.pset, [], [c], [a])
        assert q.n_modules == 0 and len(c) == p.expanded_size()
        assert q.depth() == p.depth() and q.cost_per_case() == len(c)
        if harm.pset.dtype is bool:
            assert np.array_equal(q.evaluate(X), p.evaluate(X))
        else:
            np.testing.assert_allclose(q.evaluate(X), p.evaluate(X), rtol=1e-9, atol=1e-12, equal_nan=True)


# ---- operators -------------------------------------------------------------------------
def test_offspring_valid_and_acyclic(harm):
    for p in bred(harm, n=80, gens=8, seed=3):
        assert harm.is_valid(p)  # caps, levels, legal REFs, body depths
        assert p.is_well_formed()  # REFs point to earlier modules only: a DAG
        assert p.expanded_size() <= harm.cfg.max_size and p.depth() <= harm.cfg.max_depth


@pytest.mark.parametrize("op", ["crossover", "mutate", "rewire", "encapsulate", "delete"])
def test_each_operator_valid_or_rejected(op):
    for p, mk in PSETS.items():
        arm = make_hier("a5", mk(), CFG)
        rng = np.random.default_rng(5)
        pop = arm.init_population(40, rng)
        changed = 0
        for i in range(200):
            a, b = pop[i % 40], pop[(i * 7 + 1) % 40]
            child = arm.crossover(a, b, rng) if op == "crossover" else getattr(arm, op)(a, rng)
            assert child.is_well_formed()
            if arm.is_valid(child):
                changed += child != a
        assert changed > 20


def test_strict_connectivity():
    arm = make_hier("a5", regression_set(3), CFG)
    for p in bred(arm):
        for b in range(p.out + 1):
            lvl = p.levels[b] if b < p.out else arm.cfg.levels + 1
            for c, a in zip(p.codes[b], p.args[b]):
                if c == REF:
                    if lvl == 1:
                        assert a < 0
                    else:
                        assert a >= 0 and p.levels[a] == lvl - 1


def test_frozen_levels_never_change():
    arm = make_hier("a5_frozen", regression_set(3), CFG)
    pop = bred(arm, n=50, gens=6)
    L = arm.cfg.levels
    ref = [(l, c, a) for l, c, a in zip(pop[0].levels, pop[0].codes, pop[0].args) if l < L]
    frozen = list(zip(*arm.frozen))
    for p in pop:
        mine = [(l, c, a) for l, c, a in zip(p.levels, p.codes, p.args) if l < L]
        assert mine == ref
        assert [(l, tuple(c), tuple(a)) for l, c, a in frozen] == mine


def test_hier_run_deterministic_and_budget_exact():
    from lmgp.run.runner import Run

    for arm in ("a5", "a6", "a5_copy"):
        cfg = {"arm": arm, "task": {"kind": "comp", "levels": 2, "reuse": 2, "noise": 0.0},
               "pop_size": 60, "max_generations": 10_000, "budget": 2_000_000, "seed_index": 2}
        a, b = Run(dict(cfg)).execute(), Run(dict(cfg)).execute()
        assert a["units"] == b["units"] and a["best_train"] == b["best_train"] and a["model"] == b["model"]
        assert a["units"] <= 2_000_000


# ---- A4 ------------------------------------------------------------------------------------
from lmgp.arms.a4_mmtgp import make_a4  # noqa: E402
from lmgp.core.graph import ReadoutGraph  # noqa: E402


@pytest.mark.parametrize("pname", list(PSETS))
def test_a4_valid_window_and_accounting(pname):
    arm = make_a4(PSETS[pname](), max_size=300)
    X = data(arm.pset)
    for p in bred(arm, n=40, gens=6):
        assert arm.is_valid(p) and p.n_modules == 20
        for x in range(20):
            for c, a in zip(p.codes[x], p.args[x]):
                if c == REF:
                    assert (a < 0 and x < 10) or (a >= 0 and x - 10 <= a < x)
        if isinstance(p, ReadoutGraph):
            # every tree runs once; cost = sum of tree sizes = reference count over all trees
            F = p.evaluate(X)
            for k in range(20):
                q = GraphProgram(p.pset, p.levels, list(p.codes[:20]) + [[REF]], list(p.args[:20]) + [[k]])
                out, cnt = reference_eval_graph(q, X)
                np.testing.assert_allclose(F[:, k], out, rtol=1e-9, atol=1e-12, equal_nan=True)
            assert p.cost_per_case() == sum(len(c) for c in p.codes[:20])
            assert p.expanded_size() == sum(len(_inline(p, k)[0]) for k in range(20))
        else:
            out, cnt = reference_eval_graph(p, X)
            assert cnt == p.cost_per_case() * X.shape[0]


def test_linear_readout_recovers_combination():
    from lmgp.core.scaling import linear_readout, scaled_predictions

    rng = np.random.default_rng(0)
    F = rng.normal(size=(200, 5))
    F[:, 4] = np.inf  # a broken tree gets weight 0 and does not poison predictions
    y = 1.5 + F[:, :4] @ np.array([2.0, -1.0, 0.0, 0.5])
    a, w = linear_readout(F, y)
    assert np.isclose(a, 1.5, atol=1e-3) and np.allclose(w[:4], [2, -1, 0, 0.5], atol=1e-3) and w[4] == 0
    assert np.all(np.isfinite(scaled_predictions(F, a, w)))


# ---- A3 (CGP) --------------------------------------------------------------------------------
from lmgp.arms.a3_cgp import VARIANTS as CGP_VARIANTS  # noqa: E402
from lmgp.arms.a3_cgp import make_a3  # noqa: E402


@pytest.mark.parametrize("v", list(CGP_VARIANTS))
@pytest.mark.parametrize("pname", list(PSETS))
def test_cgp_valid_and_cost_matches_reference(v, pname):
    arm = make_a3(v, PSETS[pname]())
    X = data(arm.pset)
    for p in bred(arm, n=40, gens=6):
        assert arm.is_valid(p) and p.is_well_formed()
        out, cnt = reference_eval_graph(p, X)
        assert cnt == p.cost_per_case() * X.shape[0]  # only active nodes are executed and charged
        if arm.pset.dtype is bool:
            assert np.array_equal(out, p.evaluate(X))
        else:
            np.testing.assert_allclose(out, p.evaluate(X), rtol=1e-9, atol=1e-12, equal_nan=True)


def test_cgp_single_active_mutation_changes_an_active_gene():
    arm = make_a3("a3", regression_set(3))
    rng = np.random.default_rng(1)
    for p in arm.init_population(30, rng):
        q = arm.mutate(p, rng)
        # the active part of the genotype differs (the phenotype graph changed)
        assert (q.codes, q.args) != (p.codes, p.args) or q.genes[3] != p.genes[3]


def test_cgp_levels_back_one_is_strictly_layered():
    arm = make_a3("a3_lb1", regression_set(3))
    rows = arm.cfg.rows
    for p in bred(arm, n=30, gens=5):
        consts, funcs, conns, out = p.genes
        assert out[0] == "node" and out[1] // rows == arm.cfg.cols - 1
        for i in range(arm.n_nodes):
            for s in conns[i]:
                if i // rows == 0:
                    assert s[0] in ("in", "const")
                else:
                    assert s[0] == "node" and s[1] // rows == i // rows - 1

"""G1 compute-counter tests.

* The analytic cost equals the number of nodes actually executed by an
  independent reference interpreter, for random programs of every arm.
* Identical programs cost the same.
* Every ADF call is charged in full (no caching in A1/A2).
* The runner never exceeds the budget and charges exactly the sum of the
  evaluated programs' costs.
"""
import numpy as np
import pytest

from lmgp.arms.a2_adf import make_a2
from lmgp.arms.tree import TreeArmConfig
from lmgp.core.counter import BudgetExhausted, ComputeCounter
from lmgp.core.primitives import parity_set
from lmgp.core.program import TreeProgram
from lmgp.core.reference import reference_eval

from conftest import data_for


def test_cost_matches_reference_node_count(arm):
    rng = np.random.default_rng(10)
    X = data_for(arm, n=13)
    pop = arm.init_population(60, rng)
    for _ in range(3):  # include bred programs, not just initial ones
        pop = [arm.vary([pop[i], pop[(i + 7) % len(pop)]], rng) for i in range(len(pop))]
    for p in pop:
        out_ref, count = reference_eval(p, X)
        assert count == p.cost_per_case() * X.shape[0]
        out = p.evaluate(X)
        if arm.arch.pset.dtype is bool:
            assert np.array_equal(out, out_ref)
        else:
            np.testing.assert_allclose(out, out_ref, rtol=1e-9, atol=1e-12, equal_nan=True)


def test_identical_programs_cost_the_same(arm):
    rng = np.random.default_rng(11)
    for p in arm.init_population(50, rng):
        q = TreeProgram(arm.arch, [list(c) for c in p.codes], [list(v) for v in p.vals])
        assert q is not p and q == p
        assert q.cost_per_case() == p.cost_per_case()
        assert ComputeCounter.cost(q, 64) == ComputeCounter.cost(p, 64)
        assert q.expanded_size() == p.expanded_size() and q.depth() == p.depth()


def _prog(arm, branches):
    """Build a program from symbol names, e.g. [["and", "ADF0", "x0", "x1", ...], [...]]."""
    names = {s.name: i for i, s in enumerate(arm.arch.symbols)}
    codes = [[names[t] for t in br] for br in branches]
    return TreeProgram(arm.arch, codes, [[0.0] * len(c) for c in codes])


def test_repeated_adf_calls_are_each_charged():
    arm = make_a2(parity_set(2), TreeArmConfig(), adf_arities=(2, 3))
    # RPB = and(ADF0(x0, x1), ADF0(x0, x1)); ADF0 = or(ARG0, ARG1); ADF1 = ARG2
    p = _prog(arm, [["and", "ADF0", "x0", "x1", "ADF0", "x0", "x1"], ["or", "ARG0", "ARG1"], ["ARG2"]])
    # and(1) + 2 * [call(0) + args(2) + body: or(1) + ARG reads(0)] = 7: each call is
    # executed in full, and with each argument read once it costs exactly the inlined tree
    assert p.cost_per_case() == 7
    assert p.stored_size() == 7 + 3 + 1
    assert p.expanded_size() == 7  # and(or(x0, x1), or(x0, x1))
    assert reference_eval(p, np.array([[True, False]]))[1] == 7


def test_by_value_argument_read_twice_is_computed_once():
    arm = make_a2(parity_set(2), TreeArmConfig(), adf_arities=(2, 3))
    # RPB = ADF0(and(x0, x1), x0); ADF0 = or(ARG0, ARG0)
    p = _prog(arm, [["ADF0", "and", "x0", "x1", "x0"], ["or", "ARG0", "ARG0"], ["ARG2"]])
    # args: and(x0, x1) = 3 and x0 = 1, each executed once; body: or = 1 -> 5
    assert p.cost_per_case() == 5
    assert p.expanded_size() == 7  # or(and(x0, x1), and(x0, x1))
    assert reference_eval(p, np.array([[True, False]]))[1] == 5


def test_hierarchical_adf_cost_and_depth():
    arm = make_a2(parity_set(2), TreeArmConfig(), adf_arities=(2, 3))
    # ADF1 = ADF0(ARG0, nand(ARG1, ARG2)) ; ADF0 = or(ARG0, ARG1)
    # RPB = ADF1(x0, x1, x0)
    p = _prog(arm, [["ADF1", "x0", "x1", "x0"], ["or", "ARG0", "ARG1"],
                    ["ADF0", "ARG0", "nand", "ARG1", "ARG2"]])
    # RPB: call(0) + x0, x1, x0 (3); ADF1: call(0) + ARG0(0) + nand(1) + ARG1, ARG2 (0);
    # ADF0: or(1) + ARGs(0) -> 5, the inlined size
    assert p.cost_per_case() == 5
    # inlined: or(x0, nand(x1, x0)) -> depth 2, size 5
    assert p.depth() == 2
    assert p.expanded_size() == 5
    X = np.array([[a, b] for a in (False, True) for b in (False, True)])
    out, count = reference_eval(p, X)
    assert count == p.cost_per_case() * 4
    assert np.array_equal(out, X[:, 0] | ~(X[:, 1] & X[:, 0]))


def test_counter_enforces_budget():
    c = ComputeCounter(100)
    c.charge(60)
    assert not c.can_afford(41) and c.can_afford(40)
    with pytest.raises(BudgetExhausted):
        c.charge(41)
    c.charge(40)
    assert c.units == 100 and c.evals == 2


@pytest.mark.parametrize("arm", ["a1", "a2"])
@pytest.mark.parametrize("task", [{"kind": "parity", "n": 4},
                                  {"kind": "comp", "levels": 2, "reuse": 2, "noise": 0.0}])
def test_runner_charges_exact_costs_and_stops_at_budget(arm, task, monkeypatch):
    from lmgp.core import counter as counter_mod
    from lmgp.run.runner import Run

    charged = []
    orig = counter_mod.ComputeCounter.charge

    def spy(self, cost):
        charged.append(cost)
        return orig(self, cost)

    monkeypatch.setattr(counter_mod.ComputeCounter, "charge", spy)
    budget = 3_000_000
    r = Run({"arm": arm, "task": task, "pop_size": 100, "max_generations": 10_000, "budget": budget,
             "seed_index": 1, "stop_on_solution": False})
    final = r.execute()
    assert final["units"] == sum(charged) <= budget
    assert final["evals"] == len(charged)
    # It stopped because the next program was unaffordable, not early.
    assert budget - final["units"] < max(charged) * 50
    # Validation is logged separately and never charged.
    if task["kind"] == "comp":
        assert final["val_units"] > 0


def _inline(p):
    """Build the equivalent plain tree (codes in the same architecture) by
    textual inlining of every ADF call."""
    arch = p.arch

    def expand(b, i, args):
        """Return (codes, vals, next_index) for the subtree at position i of
        branch b, with ARG references replaced by `args` (lists of codes/vals)."""
        code = p.codes[b][i]
        s = arch.symbols[code]
        if s.kind.name == "ARG":
            c, v = args[s.ref]
            return list(c), list(v), i + 1
        if s.arity == 0:
            return [code], [p.vals[b][i]], i + 1
        j = i + 1
        kids = []
        for _ in range(s.arity):
            c, v, j = expand(b, j, args)
            kids.append((c, v))
        if s.kind.name == "PRIM":
            return [code] + sum((k[0] for k in kids), []), [0.0] + sum((k[1] for k in kids), []), j
        c, v, _ = expand(s.ref + 1, 0, kids)
        return c, v, j

    c, v, _ = expand(0, 0, ())
    return TreeProgram(arch, [c] + [list(x) for x in p.codes[1:]], [v] + [list(x) for x in p.vals[1:]]), c


def test_expanded_size_and_depth_equal_explicit_inlining():
    from conftest import ARMS

    for name, arm in ARMS:
        if not arm.arch.adf_arities:
            continue
        rng = np.random.default_rng(12)
        X = data_for(arm, n=11)
        pop = arm.init_population(40, rng)
        for _ in range(3):
            pop = [arm.vary([pop[i], pop[(i + 5) % 40]], rng) for i in range(40)]
        for p in pop:
            q, codes = _inline(p)
            assert not any(arm.arch.symbols[c].kind.name == "CALL" for c in codes)
            assert p.expanded_size() == len(codes)
            out_p, out_q = p.evaluate(X), q.evaluate(X)
            if arm.arch.pset.dtype is bool:
                assert np.array_equal(out_p, out_q)
            else:
                np.testing.assert_allclose(out_p, out_q, rtol=1e-9, atol=1e-12, equal_nan=True)
            assert q.depth() == p.depth()
            # the inlined tree has no calls, so its cost equals its size
            assert q.expanded_size() == q.cost_per_case() == p.expanded_size()


def test_code_in_unread_adf_argument_counts_against_the_cap():
    """Regression (3 Nov 2026): an argument the ADF body never reads has no
    inlined size but is still evaluated and charged. Without the stored-size
    check such code bloated without limit (A2 on even-6 reached 23k nodes)."""
    arm = make_a2(parity_set(2), TreeArmConfig(max_size=12), adf_arities=(2, 3))
    # RPB = ADF1(x0, x1, big); ADF1 = ARG1, so `big` is never read
    big = ["and"] * 5 + ["x0"] * 6
    p = _prog(arm, [["ADF1", "x0", "x1", *big], ["or", "ARG0", "ARG1"], ["ARG1"]])
    assert p.expanded_size() == 1  # inlines to x1
    assert p.cost_per_case() == 2 + len(big)  # but every argument is executed and charged
    assert p.functional_size() == 2 + len(big) + 1  # ADF1 = ARG1 has no functional node
    assert not arm.is_valid(p)


def test_functional_size_never_binds_without_hidden_code():
    """With every ADF called and every argument read, functional size is at
    most expanded size, so the extra check cannot bind (and for A1 they are equal)."""
    arm = make_a2(parity_set(2), TreeArmConfig(), adf_arities=(2, 3))
    p = _prog(arm, [["and", "ADF0", "x0", "x1", "ADF1", "x0", "x1", "x1"], ["or", "ARG0", "ARG1"],
                    ["nand", "ADF0", "ARG0", "ARG1", "ARG2"]])
    # RPB: and + 5 leaves; ADF0: or; ADF1: nand. Inlined: and(or(x0, x1), nand(or(x0, x1), x1))
    assert p.functional_size() == 8 and p.expanded_size() == 9

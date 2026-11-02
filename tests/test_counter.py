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
    # and(1) + 2 * [call(1) + args(2) + body(3)] = 13
    assert p.cost_per_case() == 13
    assert p.stored_size() == 7 + 3 + 1
    assert reference_eval(p, np.array([[True, False]]))[1] == 13


def test_hierarchical_adf_cost_and_depth():
    arm = make_a2(parity_set(2), TreeArmConfig(), adf_arities=(2, 3))
    # ADF1 = ADF0(ARG0, nand(ARG1, ARG2)) ; ADF0 = or(ARG0, ARG1)
    # RPB = ADF1(x0, x1, x0)
    p = _prog(arm, [["ADF1", "x0", "x1", "x0"], ["or", "ARG0", "ARG1"],
                    ["ADF0", "ARG0", "nand", "ARG1", "ARG2"]])
    adf0 = 3
    adf1 = 5 + adf0  # 5 stored nodes + one call's body
    assert p.cost_per_case() == 4 + adf1
    # inlined: or(x0, nand(x1, x0)) under two call nodes -> depth 4
    assert p.depth() == 4
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

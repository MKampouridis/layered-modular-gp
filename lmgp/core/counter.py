"""The single compute counter used by every arm.

Unit: one node executed on one training case. The cost of evaluating a
program on the training set is `program.cost_per_case() * n_train_cases`;
each arm's `cost_per_case` defines what "node executed" means for its
representation (see the arm modules), and must satisfy two invariants that the
test-suite checks:

1. identical programs cost the same;
2. a cached module output is charged once per program evaluation.

Agreed accounting rules (Michael, 2 November 2026):

* Only training-set evaluations count against the budget. Validation
  evaluations (used to choose the final model) are counted separately and
  logged but not charged. Test evaluations are neither charged nor logged as
  compute.
* No caching of fitness across programs or generations: every evaluation of
  every offspring is charged, even if the program is a clone.
* Linear scaling and selection are not charged (same overhead for all arms).
* A run stops the moment the next evaluation would exceed the budget; that
  offspring and the rest of its generation are discarded, so every arm uses
  at most exactly the budget.

What counts as a node (3 November 2026, after the WP2 independent review):
one unit per node of the program's equivalent inlined tree that is actually
executed. Function, variable and constant nodes cost 1. Nodes that only pass
on a value already computed (an ADF call node, an ADF argument read, a read
of a cached module output) cost 0, because they vanish when the program is
inlined. So a program with no reuse costs exactly what the equivalent tree
costs in A1, and every saving comes from genuinely shared computation
(arguments passed by value, cached module outputs). Before this date, call,
argument and module-read nodes cost 1 each, which charged the modular arms
20-50% more than A1 for the same function.

Wall-clock time and the number of program evaluations are logged alongside.
"""
from __future__ import annotations

import time


class BudgetExhausted(Exception):
    pass


class ComputeCounter:
    def __init__(self, budget: float | None):
        self.budget = None if budget is None else int(budget)
        self.units = 0
        self.evals = 0
        self.val_units = 0
        self.val_evals = 0
        self._t0 = time.perf_counter()
        self._elapsed_before = 0.0  # carried over when resuming from a checkpoint

    @staticmethod
    def cost(program, n_cases: int) -> int:
        return int(program.cost_per_case()) * int(n_cases)

    def can_afford(self, cost: int) -> bool:
        return self.budget is None or self.units + cost <= self.budget

    def charge(self, cost: int) -> None:
        if not self.can_afford(cost):
            raise BudgetExhausted
        self.units += int(cost)
        self.evals += 1

    def log_validation(self, cost: int) -> None:
        self.val_units += int(cost)
        self.val_evals += 1

    @property
    def wall(self) -> float:
        return self._elapsed_before + time.perf_counter() - self._t0

    def state(self) -> dict:
        return {
            "budget": self.budget,
            "units": self.units,
            "evals": self.evals,
            "val_units": self.val_units,
            "val_evals": self.val_evals,
            "wall": self.wall,
        }

    @classmethod
    def from_state(cls, s: dict) -> "ComputeCounter":
        c = cls(s["budget"])
        c.units, c.evals = s["units"], s["evals"]
        c.val_units, c.val_evals = s["val_units"], s["val_evals"]
        c._elapsed_before = s["wall"]
        return c

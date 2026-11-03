"""Graph programs shared by the module-based arms (A3 CGP, A4, A5, A6).

A `GraphProgram` is an ordered tuple of modules followed by an output tree.
Each module (and the output tree) is a prefix tree over the primitive set
whose leaves are ERCs or REF terminals. A REF reads either a raw input x_j or
the output of an *earlier* module, so the program is a DAG and index order is
a topological order.

Bodies are stored as parallel tuples:

* ``codes``: function index into ``pset.functions`` (>= 0), or REF (-1), or
  ERC (-2);
* ``args``: for REF the target (``-1 - j`` for input x_j, ``k >= 0`` for
  module k); for ERC the constant; unused (0) for functions.

Every module carries a ``level``. The arm decides which REFs a level may hold;
this class only requires that targets are inputs or earlier modules.

Compute accounting (the same counter as every other arm, see
`lmgp.core.counter`): one unit per node of the equivalent inlined tree that
is executed, per training case.

* Function, ERC and input-REF nodes cost 1 (an input REF is a variable leaf).
* A module REF costs 0: it only passes on a value already computed, and it
  vanishes when the module is inlined (as an ADF call node does in A2).
* ``cached=True`` (A3-A6): every *active* module (reachable from the output)
  is executed once per program evaluation and its output cached, so a module
  output is charged once however often it is read. Inactive modules are not
  executed and cost nothing.
* ``cached=False``: every module REF re-executes that module (recursively),
  so the cost equals the expanded size, as for the equivalent tree.

A program in which every module is read once therefore costs exactly what
tree GP pays for the same function; sharing makes it cheaper. (Changed on 3
Nov 2026 from "every REF costs 1" after the WP2 independent review, which
showed that rule charged the graph arms 20-50% more than A1 for the same
function; see docs/progress.md.)

Size: ``expanded_size`` and ``depth`` are those of the fully inlined tree, in
which every module REF is replaced by a copy of the module's inlined body
(the REF node disappears, as an ADF call node does in A2). The size cap uses
these, so it means the same thing for every arm.
"""
from __future__ import annotations

from functools import cached_property

import numpy as np

from .primitives import PrimitiveSet

REF = -1
ERC = -2


def input_ref(j: int) -> int:
    return -1 - j


class GraphProgram:
    """Immutable modular DAG program."""

    __slots__ = ("pset", "levels", "codes", "args", "cached", "__dict__")

    def __init__(self, pset: PrimitiveSet, levels, codes, args, cached: bool = True):
        # The last body is the output tree; levels has one entry per module.
        self.pset = pset
        self.levels = tuple(int(x) for x in levels)
        self.codes = tuple(tuple(int(c) for c in b) for b in codes)
        self.args = tuple(tuple(b) for b in args)
        self.cached = bool(cached)
        assert len(self.codes) == len(self.args) == len(self.levels) + 1

    # ---- structure ----------------------------------------------------------
    @property
    def n_modules(self) -> int:
        return len(self.levels)

    @property
    def out(self) -> int:
        """Index of the output tree among the bodies."""
        return len(self.levels)

    def arity(self, code: int) -> int:
        return self.pset.functions[code].arity if code >= 0 else 0

    def refs(self, b: int) -> list[int]:
        """Module targets (>= 0) referenced by body b, with repetition."""
        return [int(a) for c, a in zip(self.codes[b], self.args[b]) if c == REF and a >= 0]

    @cached_property
    def active(self) -> tuple[bool, ...]:
        """Which modules are reachable from the output tree."""
        act = [False] * self.n_modules
        stack = self.refs(self.out)
        while stack:
            k = stack.pop()
            if not act[k]:
                act[k] = True
                stack.extend(self.refs(k))
        return tuple(act)

    def is_well_formed(self) -> bool:
        for b in range(self.out + 1):
            need = 1
            for c, a in zip(self.codes[b], self.args[b]):
                if need <= 0:
                    return False
                need += self.arity(c) - 1
                if c == REF:
                    if a >= 0 and (b != self.out and a >= b or a >= self.n_modules):
                        return False
                    if a < 0 and -1 - a >= self.pset.n_vars:
                        return False
            if need != 0:
                return False
        return list(self.levels) == sorted(self.levels)

    # ---- accounting -----------------------------------------------------------
    def own_cost(self, b: int) -> int:
        """Cost of executing body b once, given its module inputs: every node
        except module REFs."""
        return len(self.codes[b]) - len(self.refs(b))

    @cached_property
    def _module_costs(self) -> tuple[int, ...]:
        """Per-body cost of one execution without caching: own nodes plus a
        fresh execution of every referenced module."""
        cost = []
        for b in range(self.out + 1):
            c = self.own_cost(b)
            for t in self.refs(b):
                c += cost[t]
            cost.append(c)
        return tuple(cost)

    def cost_per_case(self) -> int:
        if not self.cached:
            return self._module_costs[self.out]
        act = self.active
        return self.own_cost(self.out) + sum(self.own_cost(k) for k in range(self.n_modules) if act[k])

    @cached_property
    def _inlined(self) -> tuple[tuple[int, int], ...]:
        """(size, depth) of each body when fully inlined; a module REF is
        replaced by the module's inlined tree and disappears itself."""
        res: list[tuple[int, int]] = []
        for b in range(self.out + 1):
            stack: list[tuple[int, int]] = []
            codes, args = self.codes[b], self.args[b]
            for i in range(len(codes) - 1, -1, -1):
                c = codes[i]
                if c == REF and args[i] >= 0:
                    stack.append(res[args[i]])
                elif c < 0:
                    stack.append((1, 0))
                else:
                    s, d = 1, 0
                    for _ in range(self.pset.functions[c].arity):
                        cs, cd = stack.pop()
                        s += cs
                        d = max(d, cd)
                    stack.append((s, d + 1))
            res.append(stack[0])
        return tuple(res)

    def expanded_size(self) -> int:
        return self._inlined[self.out][0]

    def depth(self) -> int:
        return self._inlined[self.out][1]

    def stored_size(self) -> int:
        """Nodes stored in the active modules plus the output tree (module
        REF leaves included)."""
        act = self.active
        return len(self.codes[self.out]) + sum(len(self.codes[k]) for k in range(self.n_modules) if act[k])

    def functional_size(self) -> int:
        """Stored nodes of the active bodies, REF leaves excluded (they only
        pass a value on). Never exceeds expanded_size, since every active
        module is read at least once; kept so every arm has the same check."""
        act = self.active
        return self.own_cost(self.out) + sum(self.own_cost(k) for k in range(self.n_modules) if act[k])

    def key(self) -> tuple:
        return (self.levels, self.codes, self.args, self.cached)

    def __eq__(self, other):
        return isinstance(other, GraphProgram) and self.key() == other.key()

    def __hash__(self):
        return hash(self.key())

    # ---- evaluation ---------------------------------------------------------
    def _eval_body(self, b, cols, cache):
        fns = self.pset.functions
        codes, args = self.codes[b], self.args[b]
        stack = []
        for i in range(len(codes) - 1, -1, -1):
            c = codes[i]
            if c == REF:
                a = args[i]
                if a < 0:
                    stack.append(cols[-1 - a])
                elif cache is not None:
                    stack.append(cache[a])
                else:
                    stack.append(self._eval_body(a, cols, None))
            elif c == ERC:
                stack.append(args[i])
            else:
                f = fns[c]
                ch = [stack.pop() for _ in range(f.arity)]
                stack.append(f.fn(*ch))
        return stack[0]

    def module_outputs(self, X: np.ndarray) -> dict[int, np.ndarray]:
        """Outputs of the active modules (used by analysis and A4's readout)."""
        cols = [X[:, j] for j in range(X.shape[1])]
        n = X.shape[0]
        cache: dict[int, np.ndarray] = {}
        with np.errstate(all="ignore"):
            for k in range(self.n_modules):
                if self.active[k]:
                    cache[k] = np.broadcast_to(self._eval_body(k, cols, cache), (n,))
        return cache

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        cols = [X[:, j] for j in range(X.shape[1])]
        with np.errstate(all="ignore"):
            if self.cached:
                cache: dict[int, np.ndarray] = {}
                for k in range(self.n_modules):
                    if self.active[k]:
                        cache[k] = self._eval_body(k, cols, cache)
                out = self._eval_body(self.out, cols, cache)
            else:
                out = self._eval_body(self.out, cols, None)
        return np.array(np.broadcast_to(out, (n,)), dtype=self.pset.dtype)

    # ---- display ------------------------------------------------------------
    def body_str(self, b: int) -> str:
        codes, args = self.codes[b], self.args[b]
        pos = 0

        def rec():
            nonlocal pos
            i = pos
            pos += 1
            c = codes[i]
            if c == REF:
                a = args[i]
                return f"x{-1 - a}" if a < 0 else f"M{a}"
            if c == ERC:
                return f"{args[i]:.4g}"
            f = self.pset.functions[c]
            return f"{f.name}(" + ", ".join(rec() for _ in range(f.arity)) + ")"

        return rec()

    def __str__(self):
        parts = [f"out := {self.body_str(self.out)}"]
        for k in range(self.n_modules):
            if self.active[k]:
                parts.append(f"M{k}[L{self.levels[k]}] := {self.body_str(k)}")
        return " ; ".join(parts)


class ReadoutGraph(GraphProgram):
    """A graph program whose prediction is a linear readout over the outputs
    of *all* its modules (A4, as published: a ridge model over every tree).

    The output body is a fixed placeholder (one ERC) and is never executed.
    Every module is active and executed once per evaluation (cached).

    The readout a + sum_i w_i T_i is fitted outside the program, as linear
    scaling a + b f is for every other arm, and the fit is not charged. Its
    k weights, however, do more than linear scaling's single weight, so the
    extra terms are charged and counted as nodes: 2(k - 1) per case (k - 1
    further multiplications and additions beyond a + b f), both in the cost
    and in the expanded size. With one tree A4 costs exactly what a
    linearly scaled tree costs. (Added on 3 Nov 2026 after the WP2
    independent review.)
    """

    @cached_property
    def active(self) -> tuple[bool, ...]:
        return (True,) * self.n_modules

    @property
    def readout_nodes(self) -> int:
        return 2 * (self.n_modules - 1)

    def cost_per_case(self) -> int:
        return sum(self.own_cost(k) for k in range(self.n_modules)) + self.readout_nodes

    def expanded_size(self) -> int:
        return sum(self._inlined[k][0] for k in range(self.n_modules)) + self.readout_nodes

    def depth(self) -> int:
        return max(self._inlined[k][1] for k in range(self.n_modules))

    def stored_size(self) -> int:
        return sum(len(self.codes[k]) for k in range(self.n_modules)) + self.readout_nodes

    def functional_size(self) -> int:
        return sum(self.own_cost(k) for k in range(self.n_modules)) + self.readout_nodes

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Matrix of module outputs, shape (n, n_modules)."""
        n = X.shape[0]
        cols = [X[:, j] for j in range(X.shape[1])]
        cache: dict[int, np.ndarray] = {}
        with np.errstate(all="ignore"):
            for k in range(self.n_modules):
                cache[k] = self._eval_body(k, cols, cache)
        return np.column_stack([np.broadcast_to(cache[k], (n,)) for k in range(self.n_modules)]).astype(
            self.pset.dtype)

    def __str__(self):
        return " ; ".join(f"T{k} := {self.body_str(k)}" for k in range(self.n_modules))

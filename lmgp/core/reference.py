"""Slow reference interpreter for TreeProgram, used only in tests.

It evaluates one training case at a time, top-down and recursively (a
different algorithm from the vectorised stack evaluator), and counts every
node it executes. The compute-counter tests check that its count equals
`cost_per_case()` and its outputs equal `evaluate()`.
"""
from __future__ import annotations

import numpy as np

from .program import Kind, TreeProgram


def reference_eval(prog: TreeProgram, X: np.ndarray):
    """Return (outputs, total_nodes_executed) over all rows of X.

    ADF arguments are passed call-by-need: each argument is a thunk, executed
    the first time the body reads it and memoised for the rest of the call.
    An argument that is never read is therefore never executed or counted.
    This is a different mechanism from the vectorised evaluator's static
    dead-argument masks, and must give the same outputs and counts."""
    arch = prog.arch
    counter = [0]

    def run_branch(b, row, args):
        codes, vals = prog.codes[b], prog.vals[b]

        def end(i):  # one past the subtree starting at i, without executing it
            need = 1
            while need:
                need += arch.arity[codes[i]] - 1
                i += 1
            return i

        def thunk(i):
            memo = []

            def force():
                if not memo:
                    memo.append(node(i)[0])
                return memo[0]
            return force

        def node(i):  # -> (value, next position)
            s = arch.symbols[codes[i]]
            if s.kind not in (Kind.CALL, Kind.ARG):  # call and argument nodes only pass values on
                counter[0] += 1
            if s.kind == Kind.VAR:
                return row[s.ref], i + 1
            if s.kind == Kind.ARG:
                return args[s.ref](), i + 1
            if s.kind == Kind.ERC:
                return vals[i], i + 1
            j = i + 1
            if s.kind == Kind.PRIM:
                ch = []
                for _ in range(s.arity):
                    v, j = node(j)
                    ch.append(v)
                return s.ref(*ch), j
            thunks = []
            for _ in range(s.arity):
                thunks.append(thunk(j))
                j = end(j)
            return run_branch(s.ref + 1, row, thunks), j

        return node(0)[0]

    out = [run_branch(0, X[r], ()) for r in range(X.shape[0])]
    return np.asarray(out, dtype=arch.pset.dtype), counter[0]


def reference_eval_graph(prog, X: np.ndarray):
    """Reference interpreter for GraphProgram: one case at a time, top-down
    recursion from the output tree, with a per-case memo of module outputs
    when the program is cached. Returns (outputs, total nodes executed)."""
    from .graph import ERC, REF

    counter = [0]
    fns = prog.pset.functions

    def run(b, row, memo):
        codes, args = prog.codes[b], prog.args[b]
        pos = [0]

        def node():
            i = pos[0]
            pos[0] += 1
            c = codes[i]
            if not (c == REF and args[i] >= 0):  # a module read only passes a value on
                counter[0] += 1
            if c == REF:
                a = args[i]
                if a < 0:
                    return row[-1 - a]
                if memo is None:
                    return run(a, row, None)
                if a not in memo:
                    memo[a] = run(a, row, memo)
                return memo[a]
            if c == ERC:
                return args[i]
            f = fns[c]
            ch = [node() for _ in range(f.arity)]
            return f.fn(*ch)

        return node()

    out = []
    with np.errstate(all="ignore"):
        for r in range(X.shape[0]):
            out.append(run(prog.out, X[r], {} if prog.cached else None))
    return np.asarray(out, dtype=prog.pset.dtype), counter[0]

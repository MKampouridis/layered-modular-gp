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
    """Return (outputs, total_nodes_executed) over all rows of X."""
    arch = prog.arch
    counter = [0]

    def run_branch(b, row, args):
        codes, vals = prog.codes[b], prog.vals[b]
        pos = [0]

        def node():
            i = pos[0]
            pos[0] += 1
            counter[0] += 1
            s = arch.symbols[codes[i]]
            if s.kind == Kind.VAR:
                return row[s.ref]
            if s.kind == Kind.ARG:
                return args[s.ref]
            if s.kind == Kind.ERC:
                return vals[i]
            ch = [node() for _ in range(s.arity)]
            if s.kind == Kind.PRIM:
                return s.ref(*ch)
            return run_branch(s.ref + 1, row, ch)

        return node()

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
            counter[0] += 1
            c = codes[i]
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

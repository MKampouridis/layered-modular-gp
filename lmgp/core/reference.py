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

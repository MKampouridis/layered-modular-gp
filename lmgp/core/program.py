"""Program representation shared by the tree-based arms (A1, A2).

A program is a tuple of branches. Branch 0 is the result-producing branch
(RPB); branches 1..K are automatically defined functions ADF0..ADF(K-1). A1 is
the special case K = 0. Each branch is a prefix-ordered tuple of integer
symbol codes plus a parallel tuple of constant values (used only by ERC
nodes).

Compute accounting (see `lmgp.core.counter`): one unit per node executed per
training case. The cost per case of a program is the number of node
executions in one evaluation:

* a primitive, variable, constant or ADF-argument node costs 1;
* an ADF call node costs 1, plus its argument subtrees, plus the full cost of
  the ADF body, charged afresh on every call (even with identical arguments).

ADF arguments are passed by value: each argument subtree is executed once per
call and every reference to it inside the body costs 1. This is genuine
reuse of a computed value, the benefit modularity is meant to provide, and it
is the same principle as charging a cached module output once per program
evaluation (A5). It means an ADF program can cost less than the equivalent
inlined tree.

Size is measured separately from cost. `expanded_size` is the size of the
equivalent plain tree with every ADF call inlined (the call node disappears,
each ARG reference is replaced by a copy of the argument subtree, unused
arguments vanish). The size cap uses `expanded_size`, so it means the same
thing for every arm; `cost_per_case` is used only by the compute counter.
For A1 the two coincide.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from functools import cached_property
from typing import Protocol, runtime_checkable

import numpy as np

from .primitives import PrimitiveSet


@runtime_checkable
class Program(Protocol):
    """Interface every arm's programs implement."""

    def evaluate(self, X: np.ndarray) -> np.ndarray: ...
    def cost_per_case(self) -> int: ...
    def expanded_size(self) -> int: ...
    def stored_size(self) -> int: ...
    def depth(self) -> int: ...
    def key(self) -> tuple: ...


class Kind(IntEnum):
    PRIM = 0
    CALL = 1
    VAR = 2
    ARG = 3
    ERC = 4


@dataclass(frozen=True)
class Symbol:
    name: str
    arity: int
    kind: Kind
    ref: object  # function for PRIM; ADF index for CALL; column for VAR; position for ARG


class Architecture:
    """Symbol table plus the symbols each branch may use.

    adf_arities: arity of each ADF (empty for plain tree GP).
    hierarchical: ADF k may call ADF j for j < k (Koza 1994).
    adf_sees_inputs: ADF bodies may read raw input variables (Koza's default
        is no: ADFs see only their arguments).
    """

    def __init__(self, pset: PrimitiveSet, adf_arities=(), hierarchical=True, adf_sees_inputs=False):
        self.pset = pset
        self.adf_arities = tuple(int(a) for a in adf_arities)
        self.hierarchical = hierarchical
        self.adf_sees_inputs = adf_sees_inputs
        syms: list[Symbol] = [Symbol(f.name, f.arity, Kind.PRIM, f.fn) for f in pset.functions]
        prim_codes = list(range(len(syms)))
        call_codes = []
        for k, a in enumerate(self.adf_arities):
            call_codes.append(len(syms))
            syms.append(Symbol(f"ADF{k}", a, Kind.CALL, k))
        var_codes = []
        for j in range(pset.n_vars):
            var_codes.append(len(syms))
            syms.append(Symbol(f"x{j}", 0, Kind.VAR, j))
        arg_codes = []
        for i in range(max(self.adf_arities, default=0)):
            arg_codes.append(len(syms))
            syms.append(Symbol(f"ARG{i}", 0, Kind.ARG, i))
        erc_codes = []
        if pset.erc:
            erc_codes.append(len(syms))
            syms.append(Symbol("ERC", 0, Kind.ERC, None))
        self.symbols = tuple(syms)
        self.arity = tuple(s.arity for s in syms)
        self.kind = tuple(s.kind for s in syms)
        self.erc_code = erc_codes[0] if erc_codes else None

        # Allowed functions and terminals per branch.
        self.branch_functions: list[tuple[int, ...]] = []
        self.branch_terminals: list[tuple[int, ...]] = []
        self.branch_functions.append(tuple(prim_codes + call_codes))
        self.branch_terminals.append(tuple(var_codes + erc_codes))
        for k, a in enumerate(self.adf_arities):
            calls = call_codes[:k] if hierarchical else []
            terms = arg_codes[:a] + (var_codes if adf_sees_inputs else []) + erc_codes
            self.branch_functions.append(tuple(prim_codes + calls))
            self.branch_terminals.append(tuple(terms))

    @property
    def n_branches(self) -> int:
        return 1 + len(self.adf_arities)

    def describe(self) -> dict:
        return {
            "functions": [f.name for f in self.pset.functions],
            "n_vars": self.pset.n_vars,
            "erc": self.pset.erc,
            "adf_arities": list(self.adf_arities),
            "hierarchical": self.hierarchical,
            "adf_sees_inputs": self.adf_sees_inputs,
        }


class TreeProgram:
    """Immutable multi-branch tree program."""

    __slots__ = ("arch", "codes", "vals", "__dict__")

    def __init__(self, arch: Architecture, codes, vals):
        self.arch = arch
        self.codes = tuple(tuple(c) for c in codes)
        self.vals = tuple(tuple(v) for v in vals)

    # ---- structure ---------------------------------------------------------
    @cached_property
    def _branch_costs(self) -> tuple[int, ...]:
        """Cost per case of executing each branch once (RPB at index 0)."""
        kind, syms = self.arch.kind, self.arch.symbols
        K = len(self.arch.adf_arities)
        adf_cost = [0] * K
        # ADF k may only call ADF j < k, so compute in index order.
        for k in range(K):
            c = 0
            for code in self.codes[k + 1]:
                c += 1
                if kind[code] == Kind.CALL:
                    c += adf_cost[syms[code].ref]
            adf_cost[k] = c
        rpb = 0
        for code in self.codes[0]:
            rpb += 1
            if kind[code] == Kind.CALL:
                rpb += adf_cost[syms[code].ref]
        return (rpb, *adf_cost)

    def cost_per_case(self) -> int:
        return self._branch_costs[0]

    def expanded_size(self) -> int:
        return self._inlined_size

    @cached_property
    def _inlined_size(self) -> int:
        memo: dict = {}

        def branch_size(b: int, arg_sizes: tuple[int, ...]) -> int:
            key = (b, arg_sizes)
            if key in memo:
                return memo[key]
            kind, syms, arity = self.arch.kind, self.arch.symbols, self.arch.arity
            stack: list[int] = []
            for code in reversed(self.codes[b]):
                k = kind[code]
                if k == Kind.ARG:
                    stack.append(arg_sizes[syms[code].ref])
                elif k in (Kind.VAR, Kind.ERC):
                    stack.append(1)
                else:
                    ch = tuple(stack.pop() for _ in range(arity[code]))
                    if k == Kind.PRIM:
                        stack.append(1 + sum(ch))
                    else:
                        stack.append(branch_size(syms[code].ref + 1, ch))
            memo[key] = stack[0]
            return stack[0]

        return branch_size(0, ())

    def stored_size(self) -> int:
        return sum(len(c) for c in self.codes)

    def depth(self) -> int:
        return self._depth

    @cached_property
    def _depth(self) -> int:
        """Depth of the inlined tree (as for expanded_size: call nodes
        disappear, ARG leaves are replaced by the argument subtrees). A lone
        terminal has depth 0."""
        memo: dict = {}

        def branch_depth(b: int, arg_depths: tuple[int, ...]) -> int:
            key = (b, arg_depths)
            if key in memo:
                return memo[key]
            kind, syms, arity = self.arch.kind, self.arch.symbols, self.arch.arity
            stack: list[int] = []
            for code in reversed(self.codes[b]):
                k = kind[code]
                if k == Kind.ARG:
                    stack.append(arg_depths[syms[code].ref])
                elif k in (Kind.VAR, Kind.ERC):
                    stack.append(0)
                else:
                    ch = tuple(stack.pop() for _ in range(arity[code]))
                    if k == Kind.PRIM:
                        stack.append(1 + max(ch))
                    else:
                        stack.append(branch_depth(syms[code].ref + 1, ch))
            memo[key] = stack[0]
            return stack[0]

        return branch_depth(0, ())

    def key(self) -> tuple:
        return (self.codes, self.vals)

    def __eq__(self, other):
        return isinstance(other, TreeProgram) and self.key() == other.key()

    def __hash__(self):
        return hash(self.key())

    # ---- evaluation --------------------------------------------------------
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Vectorised evaluation over the rows of X; returns shape (n,)."""
        cols = [X[:, j] for j in range(X.shape[1])]
        out = self._eval_branch(0, cols, ())
        n = X.shape[0]
        return np.array(np.broadcast_to(out, (n,)), dtype=self.arch.pset.dtype)

    def _eval_branch(self, b, cols, args):
        kind, syms, arity = self.arch.kind, self.arch.symbols, self.arch.arity
        codes, vals = self.codes[b], self.vals[b]
        stack = []
        for i in range(len(codes) - 1, -1, -1):
            code = codes[i]
            k = kind[code]
            if k == Kind.VAR:
                stack.append(cols[syms[code].ref])
            elif k == Kind.ARG:
                stack.append(args[syms[code].ref])
            elif k == Kind.ERC:
                stack.append(vals[i])
            else:
                a = arity[code]
                ch = [stack.pop() for _ in range(a)]
                if k == Kind.PRIM:
                    stack.append(syms[code].ref(*ch))
                else:
                    stack.append(self._eval_branch(syms[code].ref + 1, cols, ch))
        return stack[0]

    # ---- display -----------------------------------------------------------
    def branch_str(self, b: int) -> str:
        syms, arity = self.arch.symbols, self.arch.arity
        codes, vals = self.codes[b], self.vals[b]
        pos = 0

        def rec():
            nonlocal pos
            i = pos
            pos += 1
            s = syms[codes[i]]
            if s.kind == Kind.ERC:
                return f"{vals[i]:.4g}"
            if s.arity == 0:
                return s.name
            return f"{s.name}(" + ", ".join(rec() for _ in range(arity[codes[i]])) + ")"

        return rec()

    def __str__(self):
        parts = [self.branch_str(0)]
        for k in range(len(self.arch.adf_arities)):
            parts.append(f"ADF{k} := {self.branch_str(k + 1)}")
        return " ; ".join(parts)

    __repr__ = __str__

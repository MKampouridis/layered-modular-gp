"""A3: Cartesian GP (Miller and Thomson 2000), with the levels-back variants
used as a skip ablation (docs/wp2-design.md).

Genotype: a rows x cols grid of nodes; each node has a function gene and
max-arity connection genes; one output gene. Regression adds `n_const`
constant nodes (CGP's usual stand-in for ERCs), readable wherever the inputs
are. A node in column c may read the inputs (and constants) if c < lb, and
nodes of columns max(0, c - lb) .. c - 1. Inactive nodes are kept in the
genotype (CGP's neutral drift) but are not executed and cost nothing.

Variation: Goldman and Punch's (2013) "single active" mutation: genes are
mutated one at a time, uniformly over the genotype, until one active gene has
changed, so no evaluation is spent on a phenotypically identical offspring.
No crossover, as is standard for CGP. Selection is the shared lexicase loop
(a deviation from CGP's usual (1+4)-ES, so that only the representation
differs between arms).

The phenotype is a `GraphProgram`: each node is a module with body
f(REF, ..., REF) (cost 1 + arity per case when active), each constant a
module with body ERC (cost 1), and the output tree a single REF (cost 1).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property

import numpy as np

from ..core.graph import ERC, REF, GraphProgram, input_ref
from ..core.primitives import PrimitiveSet


@dataclass
class CGPConfig:
    rows: int = 1
    cols: int = 100
    levels_back: int = 100
    output_any: bool = True  # False: the output reads the last column only
    n_const: int = 2  # regression only
    max_size: int = 500
    max_depth: int = 17
    max_tries: int = 10
    extra: dict = field(default_factory=dict)


class CGPProgram(GraphProgram):
    """A GraphProgram that also carries its full CGP genotype."""

    __slots__ = ()

    def __init__(self, pset, levels, codes, args, genes):
        super().__init__(pset, levels, codes, args, cached=True)
        self.__dict__["genes"] = genes  # (consts, funcs, conns, output)

    def key(self):
        return ("cgp", self.genes)


class CGPArm:
    def __init__(self, pset: PrimitiveSet, cfg: CGPConfig, name: str = "a3"):
        self.pset = pset
        self.cfg = cfg
        self.name = name
        self.n_nodes = cfg.rows * cfg.cols
        self.amax = max(f.arity for f in pset.functions)
        self.nc = cfg.n_const if pset.erc else 0

    # ---- connectivity -------------------------------------------------------------
    def col(self, i):
        return i // self.cfg.rows

    @cached_property
    def _sources(self):
        """Legal sources per node: ('in', j), ('const', k) or ('node', i)."""
        c, out = self.cfg, []
        base = [("in", j) for j in range(self.pset.n_vars)] + [("const", k) for k in range(self.nc)]
        for i in range(self.n_nodes):
            ci = self.col(i)
            s = list(base) if ci < c.levels_back else []
            s += [("node", m) for m in range(self.n_nodes) if max(0, ci - c.levels_back) <= self.col(m) < ci]
            out.append(s)
        return out

    @cached_property
    def _out_sources(self):
        c = self.cfg
        if c.output_any:
            return [("in", j) for j in range(self.pset.n_vars)] + [("node", m) for m in range(self.n_nodes)]
        last = c.cols - 1
        return [("node", m) for m in range(self.n_nodes) if self.col(m) == last]

    # ---- genotype <-> program ---------------------------------------------------------
    def _rand_src(self, srcs, rng, avoid=None):
        cand = [s for s in srcs if s != avoid] or srcs
        return cand[int(rng.integers(len(cand)))]

    def random_genes(self, rng):
        lo, hi = self.pset.erc_range
        consts = tuple(float(rng.uniform(lo, hi)) for _ in range(self.nc))
        funcs = tuple(int(rng.integers(len(self.pset.functions))) for _ in range(self.n_nodes))
        conns = tuple(tuple(self._rand_src(self._sources[i], rng) for _ in range(self.amax))
                      for i in range(self.n_nodes))
        out = self._rand_src(self._out_sources, rng)
        return consts, funcs, conns, out

    def _active_nodes(self, genes):
        consts, funcs, conns, out = genes
        act = [False] * self.n_nodes
        stack = [out[1]] if out[0] == "node" else []
        while stack:
            i = stack.pop()
            if act[i]:
                continue
            act[i] = True
            for s in conns[i][: self.pset.functions[funcs[i]].arity]:
                if s[0] == "node":
                    stack.append(s[1])
        return act

    def build(self, genes) -> CGPProgram:
        consts, funcs, conns, out = genes
        nc = self.nc

        def ref(s):
            if s[0] == "in":
                return input_ref(s[1])
            if s[0] == "const":
                return s[1]
            return nc + s[1]

        levels = [0] * nc + [self.col(i) + 1 for i in range(self.n_nodes)]
        codes = [[ERC] for _ in range(nc)]
        args = [[consts[k]] for k in range(nc)]
        for i in range(self.n_nodes):
            ar = self.pset.functions[funcs[i]].arity
            codes.append([funcs[i]] + [REF] * ar)
            args.append([0] + [ref(s) for s in conns[i][:ar]])
        codes.append([REF])
        args.append([ref(out)])
        return CGPProgram(self.pset, levels, codes, args, genes)

    # ---- arm interface ------------------------------------------------------------------
    def is_valid(self, p) -> bool:
        return (p.expanded_size() <= self.cfg.max_size and p.functional_size() <= self.cfg.max_size
                and p.depth() <= self.cfg.max_depth)

    def init_population(self, n, rng):
        pop, seen, retries = [], set(), 0
        for _ in range(n):
            for t in range(200):
                p = self.build(self.random_genes(rng))
                if self.is_valid(p) and p.key() not in seen:
                    break
                retries += 1
            seen.add(p.key())
            pop.append(p)  # after 200 tries the last draw is kept even if invalid
            if not self.is_valid(p):
                raise RuntimeError("CGP initialisation could not satisfy the size cap")
        self.init_stats = {"init_retries": retries}
        return pop

    def mutate(self, p: CGPProgram, rng) -> CGPProgram:
        """Single-active mutation (Goldman and Punch 2013)."""
        consts, funcs, conns, out = p.genes
        consts, funcs, conns = list(consts), list(funcs), [list(c) for c in conns]
        act = self._active_nodes(p.genes)
        n_genes = self.nc + self.n_nodes * (1 + self.amax) + 1
        lo, hi = self.pset.erc_range
        while True:
            g = int(rng.integers(n_genes))
            if g == n_genes - 1:
                out = self._rand_src(self._out_sources, rng, avoid=out)
                break  # the output gene is always active
            if g < self.nc:
                consts[g] = float(rng.uniform(lo, hi))
                used = any(s == ("const", g) for i in range(self.n_nodes) if act[i]
                           for s in conns[i][: self.pset.functions[funcs[i]].arity]) or out == ("const", g)
                if used:
                    break
                continue
            g -= self.nc
            i, slot = divmod(g, 1 + self.amax)
            if slot == 0:
                nf = len(self.pset.functions)
                if nf > 1:
                    funcs[i] = int((funcs[i] + 1 + rng.integers(nf - 1)) % nf)
                if act[i]:
                    break
            else:
                conns[i][slot - 1] = self._rand_src(self._sources[i], rng, avoid=conns[i][slot - 1])
                if act[i] and slot - 1 < self.pset.functions[funcs[i]].arity:
                    break
        return self.build((tuple(consts), tuple(funcs), tuple(tuple(c) for c in conns), out))

    def vary(self, parents, rng):
        p1 = parents[0]
        for _ in range(self.cfg.max_tries):
            child = self.mutate(p1, rng)
            if self.is_valid(child):
                return child
        return p1

    def describe(self) -> dict:
        d = {"arm": self.name, "functions": [f.name for f in self.pset.functions],
             "n_vars": self.pset.n_vars, "erc_consts": self.nc}
        d.update({k: v for k, v in self.cfg.__dict__.items() if k != "extra"})
        return d


VARIANTS = {
    "a3": {},
    # Same 10 x 6 grid for both: with levels-back 1 every active path crosses all
    # columns, so the number of columns is the depth and must respect the depth
    # and expanded-size caps (6 levels of binary functions: at most 127 nodes).
    "a3_lb1": {"rows": 10, "cols": 6, "levels_back": 1, "output_any": False},
    "a3_lbmax": {"rows": 10, "cols": 6, "levels_back": 6, "output_any": True},
}


def make_a3(name: str, pset: PrimitiveSet, cfg: CGPConfig | None = None) -> CGPArm:
    return CGPArm(pset, replace(cfg or CGPConfig(), **VARIANTS[name]), name=name)

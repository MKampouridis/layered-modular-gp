"""Tree GP with optional automatically defined functions.

A1 (`a1_tree.py`) is this arm with no ADFs; A2 (`a2_adf.py`) adds ADFs. Both
share one implementation so that their operators and compute accounting are
identical apart from the ADF branches.

Operators (Koza 1992, 1994):
* ramped half-and-half initialisation, every branch generated with the same
  depth and method;
* subtree crossover, branch-typed: a branch is chosen with probability
  proportional to its size in the first parent, and the crossover point in the
  second parent is taken from the same branch;
* subtree mutation: a node is replaced by a fresh grow tree;
* node choice uses the 90/10 internal/leaf rule.

Offspring that break the expanded-size or expanded-depth cap are rejected and
the operator retried, up to `max_tries`; after that the first parent is copied.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.primitives import PrimitiveSet
from ..core.program import Architecture, TreeProgram
from ..core.tree_ops import generate, pick_node, replace_subtree, subtree_end


@dataclass
class TreeArmConfig:
    max_size: int = 500  # expanded and functional nodes; identical across arms
    max_depth: int = 17  # expanded depth (Koza)
    init_depth: tuple[int, int] = (2, 6)
    p_crossover: float = 0.9  # otherwise subtree mutation
    mutation_depth: int = 4  # max depth of the grow tree inserted by mutation
    p_internal: float = 0.9
    max_tries: int = 10
    adf_arities: tuple[int, ...] = ()
    hierarchical: bool = True
    adf_sees_inputs: bool = False
    extra: dict = field(default_factory=dict)


class TreeArm:
    def __init__(self, pset: PrimitiveSet, cfg: TreeArmConfig, name: str = "tree"):
        self.cfg = cfg
        self.name = name
        self.arch = Architecture(pset, cfg.adf_arities, cfg.hierarchical, cfg.adf_sees_inputs)

    # ---- validity ----------------------------------------------------------
    def is_valid(self, p: TreeProgram) -> bool:
        # The functional-size check only binds for A2: code in an ADF argument
        # the body never reads, or in an ADF that is never called, has no
        # inlined size, and without this check it bloats without limit
        # (3 Nov 2026). For A1 it equals expanded_size.
        return (p.expanded_size() <= self.cfg.max_size and p.functional_size() <= self.cfg.max_size
                and p.depth() <= self.cfg.max_depth)

    # ---- initialisation ----------------------------------------------------
    def random_program(self, depth: int, method: str, rng) -> TreeProgram:
        codes, vals = [], []
        for b in range(self.arch.n_branches):
            c, v = generate(self.arch, b, depth, method, rng)
            codes.append(c)
            vals.append(v)
        return TreeProgram(self.arch, codes, vals)

    def init_population(self, n: int, rng) -> list[TreeProgram]:
        """Ramped half-and-half over depths init_depth[0]..init_depth[1].

        Each slot tries 20 times at its (depth, method) to get a valid,
        non-duplicate program; if that fails (e.g. ADF programs whose inlined
        size breaks the cap at large depth) the depth is lowered by one and
        the method kept, down to depth 1. Only then is a duplicate accepted.
        The same rule applies to every arm; `init_stats` records how often
        the depth had to be lowered."""
        lo, hi = self.cfg.init_depth
        depths = list(range(lo, hi + 1))
        pop: list[TreeProgram] = []
        seen: set = set()
        lowered = dup = 0
        for i in range(n):
            depth = depths[i % len(depths)]
            method = "full" if (i // len(depths)) % 2 == 0 else "grow"
            chosen = None
            for d in range(depth, 0, -1):
                for _ in range(20):
                    p = self.random_program(d, method, rng)
                    if self.is_valid(p) and p.key() not in seen:
                        chosen = p
                        break
                if chosen is not None:
                    lowered += d < depth
                    break
            if chosen is None:
                dup += 1
                while chosen is None or not self.is_valid(chosen):
                    chosen = self.random_program(1, method, rng)
            seen.add(chosen.key())
            pop.append(chosen)
        self.init_stats = {"init_depth_lowered": lowered, "init_duplicates": dup}
        return pop

    # ---- variation ---------------------------------------------------------
    def _choose_branch(self, p: TreeProgram, rng) -> int:
        sizes = np.array([len(c) for c in p.codes], dtype=float)
        return int(rng.choice(len(sizes), p=sizes / sizes.sum()))

    def crossover(self, p1: TreeProgram, p2: TreeProgram, rng) -> TreeProgram:
        ar = self.arch.arity
        b = self._choose_branch(p1, rng)
        i = pick_node(p1.codes[b], ar, rng, self.cfg.p_internal)
        j = pick_node(p2.codes[b], ar, rng, self.cfg.p_internal)
        ie = subtree_end(p1.codes[b], i, ar)
        je = subtree_end(p2.codes[b], j, ar)
        c, v = replace_subtree(p1.codes[b], p1.vals[b], i, ie, p2.codes[b][j:je], p2.vals[b][j:je])
        codes = list(p1.codes)
        vals = list(p1.vals)
        codes[b], vals[b] = c, v
        return TreeProgram(self.arch, codes, vals)

    def mutate(self, p: TreeProgram, rng) -> TreeProgram:
        ar = self.arch.arity
        b = self._choose_branch(p, rng)
        i = pick_node(p.codes[b], ar, rng, self.cfg.p_internal)
        ie = subtree_end(p.codes[b], i, ar)
        d = int(rng.integers(0, self.cfg.mutation_depth + 1))
        sc, sv = generate(self.arch, b, d, "grow", rng)
        c, v = replace_subtree(p.codes[b], p.vals[b], i, ie, sc, sv)
        codes = list(p.codes)
        vals = list(p.vals)
        codes[b], vals[b] = c, v
        return TreeProgram(self.arch, codes, vals)

    def vary(self, parents, rng) -> TreeProgram:
        p1, p2 = parents[0], parents[1]
        use_xo = rng.random() < self.cfg.p_crossover
        for _ in range(self.cfg.max_tries):
            child = self.crossover(p1, p2, rng) if use_xo else self.mutate(p1, rng)
            if self.is_valid(child):
                return child
        return p1

    def describe(self) -> dict:
        d = {"arm": self.name, **self.arch.describe()}
        d.update({k: v for k, v in self.cfg.__dict__.items() if k != "extra"})
        return d

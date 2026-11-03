"""A4: modular multi-tree GP (Zhang, Chen, Xue, Banzhaf and Zhang, IEEE TEVC
2024), configured as the authors' published MMTGP settings
(EvolutionaryForest `experiment/methods/MMTGP.py`, read on 2 Nov 2026; see
docs/lit-check.md section 2.11):

* 20 trees, each of depth at most 2 (initial depth 1-2);
* tree x reads the outputs of trees max(0, x-10) .. x-1 (sliding window of
  10); trees 0-9 also read the raw inputs, trees 10-19 do not (strict);
* regression: the prediction is a ridge readout over the outputs of all 20
  trees (`ReadoutGraph`); Boolean tasks: the last tree is the output (a
  linear readout has no meaning there; our adaptation).

Operators: macro crossover (replace tree x with the partner's tree drawn
from the same block of 10, as in the published code), aligned subtree
crossover on one tree and subtree mutation on one tree. MMTGP's adaptive
operator rates, EDA terminal mutation and redundant/irrelevant-tree deletion
are not reproduced: every arm uses fixed rates set by the same tuning
procedure (docs/wp2-design.md).

Accounting is the shared GraphProgram rule: each tree is executed once per
evaluation and every read of another tree's output is one REF node.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..core.graph import ERC, REF, GraphProgram, ReadoutGraph
from ..core.primitives import PrimitiveSet
from .hier import HierArm, HierArmConfig, _subtree_end


@dataclass
class A4Config:
    n_trees: int = 20
    window: int = 10
    input_trees: int = 10  # trees 0..input_trees-1 read the raw inputs
    init_depth: tuple[int, int] = (1, 2)
    max_tree_depth: int = 2
    mutation_depth: int = 2
    p_macro: float = 0.25
    p_crossover: float = 0.65
    p_mutation: float = 0.10
    p_internal: float = 0.9
    max_size: int = 500
    max_depth: int = 17
    max_tries: int = 10
    extra: dict = field(default_factory=dict)


class A4Arm(HierArm):
    def __init__(self, pset: PrimitiveSet, cfg: A4Config, name: str = "a4"):
        self.a4 = cfg
        h = HierArmConfig(levels=cfg.n_trees, min_modules=1, max_modules=1, init_modules=(1, 1),
                          module_init_depth=cfg.init_depth, module_max_depth=cfg.max_tree_depth,
                          out_init_depth=(1, 1), out_max_depth=1, mutation_depth=cfg.mutation_depth,
                          p_internal=cfg.p_internal, max_size=cfg.max_size, max_depth=cfg.max_depth,
                          max_tries=cfg.max_tries)
        super().__init__(pset, h, name=name)
        self.readout = pset.dtype is not bool

    # tree x has level x + 1; the output "level" is n_trees + 1
    def targets(self, levels, lvl):
        c = self.a4
        if lvl == c.n_trees + 1:  # Boolean output: the last tree
            return [len(levels) - 1]
        x = lvl - 1
        t = list(self.inputs) if x < c.input_trees else []
        return t + [k for k, l in enumerate(levels) if lvl - c.window <= l < lvl]

    def _pack(self, levels, codes, args):
        cls = ReadoutGraph if self.readout else GraphProgram
        return cls(self.pset, levels, codes, args, cached=True)

    def _random_program(self, out_depth, method, rng, frozen=None):
        c = self.a4
        levels, codes, args = [], [], []
        for x in range(c.n_trees):
            d = int(rng.integers(c.init_depth[0], c.init_depth[1] + 1))
            cc, aa = self._gen(self.targets(levels, x + 1), d, method, rng)
            levels.append(x + 1)
            codes.append(cc)
            args.append(aa)
        if self.readout:
            codes.append([ERC])
            args.append([0.0])
        else:
            codes.append([REF])
            args.append([c.n_trees - 1])
        return self._pack(levels, codes, args)

    def _evolvable(self, levels):
        return list(range(len(levels)))  # trees only, never the output placeholder

    def macro_crossover(self, p1, p2, rng):
        """Replace one tree of p1 by a tree of p2 from the same block of
        `window` trees; REFs that are illegal at the new position are re-drawn."""
        c = self.a4
        levels, codes, args = self._unpack(p1)
        x = int(rng.integers(c.n_trees))
        lo = (x // c.window) * c.window
        y = int(rng.integers(lo, min(lo + c.window, c.n_trees)))
        legal = self.targets(levels, x + 1)
        lset = set(legal)
        codes[x] = list(p2.codes[y])
        args[x] = [(a if (cc != REF or a in lset) else legal[int(rng.integers(len(legal)))])
                   for cc, a in zip(p2.codes[y], p2.args[y])]
        return self._pack(levels, codes, args)

    def _finish(self, levels, codes, args):
        return self._pack(levels, codes, args)  # fixed shape: nothing to prune or re-index

    def vary(self, parents, rng):
        c = self.a4
        p1, p2 = parents[0], parents[1]
        rates = np.array([c.p_macro, c.p_crossover, c.p_mutation])
        op = int(rng.choice(3, p=rates / rates.sum()))
        for _ in range(c.max_tries):
            if op == 0:
                child = self.macro_crossover(p1, p2, rng)
            elif op == 1:
                child = self.crossover(p1, p2, rng)
            else:
                child = self.mutate(p1, rng)
            if self.is_valid(child):
                return child
        return p1

    def describe(self) -> dict:
        d = {"arm": self.name, "functions": [f.name for f in self.pset.functions],
             "n_vars": self.pset.n_vars, "erc": self.pset.erc, "readout": self.readout}
        d.update({k: v for k, v in self.a4.__dict__.items() if k != "extra"})
        return d


def make_a4(pset: PrimitiveSet, cfg: A4Config | None = None, **overrides) -> A4Arm:
    return A4Arm(pset, replace(cfg or A4Config(), **overrides))

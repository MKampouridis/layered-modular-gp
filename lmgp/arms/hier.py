"""Hierarchical modular GP: A5, A6, A6-inputs and the ablations A5-copy and
A5-frozen (design in docs/wp2-design.md).

A program is a `GraphProgram` whose modules sit in levels 1..L, plus an
output tree. What a body at level l may read (its REF targets) is set by the
connectivity rule:

* ``strict`` (A5): level 1 reads the raw inputs; level l > 1 reads only level
  l-1; the output tree reads only level L.
* ``dense`` (A6): every body reads the raw inputs and every lower level.
* ``inputs`` (A6-inputs): level l-1 plus the raw inputs; the output tree reads
  level L plus the raw inputs.

Variation (one operator per offspring, chosen by the configured rates):
subtree crossover between aligned bodies, module mutation (subtree mutation
inside one body), rewiring (re-point one REF), encapsulation (copy a subtree
of a level-l module into a new level-l module and re-point a REF one level up
to it) and deletion (remove a module, re-pointing its readers to another
module of the same level). After every operator, modules the output can no
longer reach are removed (each level keeps at least one module).

A5-copy gives every module read a private copy (`private_copies`): after
every operator, a module read by several REFs is duplicated so that each REF
owns its module, so neither computation nor genes are shared, and the cost
equals the inlined tree's. Operators are A5's. (Before 3 Nov 2026 A5-copy
only switched the cache off and kept shared genes; changed after the WP2
independent review.)
A5-frozen draws levels 1..L-1 at random once per run, shares them across the
population and never varies them; only level L and the output tree evolve.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.graph import ERC, REF, GraphProgram, input_ref
from ..core.primitives import PrimitiveSet


@dataclass
class HierArmConfig:
    levels: int = 4
    connectivity: str = "strict"  # strict (A5) | dense (A6) | inputs (A6-inputs)
    cached: bool = True  # False: A5-copy
    frozen_below: bool = False  # True: A5-frozen
    private_copies: bool = False  # True: A5-copy (every module read gets its own module)
    min_modules: int = 1
    max_modules: int = 8  # per level
    init_modules: tuple[int, int] = (2, 4)
    module_init_depth: tuple[int, int] = (1, 2)
    module_max_depth: int = 4  # stored depth of a module body
    out_init_depth: tuple[int, int] = (1, 3)
    out_max_depth: int = 8  # stored depth of the output tree
    mutation_depth: int = 2
    p_crossover: float = 0.5
    p_mutation: float = 0.2
    p_rewire: float = 0.15
    p_encapsulate: float = 0.1
    p_delete: float = 0.05
    p_internal: float = 0.9
    max_size: int = 500  # expanded nodes; identical across arms
    max_depth: int = 17  # expanded depth
    max_tries: int = 10
    extra: dict = field(default_factory=dict)


class _Arity:
    def __init__(self, pset):
        self.a = [f.arity for f in pset.functions]

    def __getitem__(self, c):
        return self.a[c] if c >= 0 else 0


def _subtree_end(codes, start, ar):
    need, j = 1, start
    while need:
        need += ar[codes[j]] - 1
        j += 1
    return j


def _body_depth(codes, ar):
    stack = []
    for c in reversed(codes):
        k = ar[c]
        if k == 0:
            stack.append(0)
        else:
            d = max(stack.pop() for _ in range(k))
            stack.append(d + 1)
    return stack[0]


class HierArm:
    def __init__(self, pset: PrimitiveSet, cfg: HierArmConfig, name: str = "a5"):
        if cfg.connectivity not in ("strict", "dense", "inputs"):
            raise ValueError(cfg.connectivity)
        if cfg.min_modules < 1:
            raise ValueError("every level needs at least one module (min_modules >= 1)")
        self.pset = pset
        self.cfg = cfg
        self.name = name
        self.ar = _Arity(pset)
        self.funcs = list(range(len(pset.functions)))
        self.inputs = [input_ref(j) for j in range(pset.n_vars)]
        self.frozen = None  # (levels, codes, args) of the shared frozen levels (A5-frozen)

    # ---- connectivity -----------------------------------------------------------
    def targets(self, levels, lvl: int) -> list[int]:
        """Legal REF targets for a body at level `lvl` (L+1 = output tree)."""
        L, mode = self.cfg.levels, self.cfg.connectivity
        if mode == "strict":
            if lvl == 1:
                return list(self.inputs)
            return [k for k, l in enumerate(levels) if l == lvl - 1]
        if mode == "dense":
            return list(self.inputs) + [k for k, l in enumerate(levels) if l < lvl]
        below = [k for k, l in enumerate(levels) if l == lvl - 1]
        return list(self.inputs) + below

    # ---- generation ---------------------------------------------------------------
    def _leaf(self, targets, rng):
        n = len(targets) + (1 if self.pset.erc else 0)
        i = int(rng.integers(n))
        if i < len(targets):
            return REF, int(targets[i])
        lo, hi = self.pset.erc_range
        return ERC, float(rng.uniform(lo, hi))

    def _gen(self, targets, depth, method, rng):
        codes, args = [], []
        n_f = len(self.funcs)
        n_t = len(targets) + (1 if self.pset.erc else 0)

        def rec(d):
            if d == 0 or (method == "grow" and rng.integers(n_f + n_t) >= n_f):
                c, a = self._leaf(targets, rng)
                codes.append(c)
                args.append(a)
                return
            f = self.funcs[int(rng.integers(n_f))]
            codes.append(f)
            args.append(0)
            for _ in range(self.ar[f]):
                rec(d - 1)

        rec(depth)
        return codes, args

    # ---- conversions ----------------------------------------------------------------
    @staticmethod
    def _unpack(p: GraphProgram):
        return [list(p.levels), [list(c) for c in p.codes], [list(a) for a in p.args]]

    def _pack(self, levels, codes, args) -> GraphProgram:
        return GraphProgram(self.pset, levels, codes, args, cached=self.cfg.cached)

    # ---- validity ----------------------------------------------------------------------
    def is_valid(self, p: GraphProgram) -> bool:
        if (p.expanded_size() > self.cfg.max_size or p.functional_size() > self.cfg.max_size
                or p.depth() > self.cfg.max_depth):
            return False
        if not p.is_well_formed():
            return False
        L = self.cfg.levels
        counts = [0] * (L + 1)
        for l in p.levels:
            if not 1 <= l <= L:
                return False
            counts[l] += 1
        if any(c < self.cfg.min_modules or c > self.cfg.max_modules for c in counts[1:]):
            return False
        for b in range(p.out + 1):
            lvl = p.levels[b] if b < p.out else L + 1
            legal = set(self.targets(p.levels, lvl))
            for c, a in zip(p.codes[b], p.args[b]):
                if c == REF and a not in legal:
                    return False
            lim = self.cfg.out_max_depth if b == p.out else self.cfg.module_max_depth
            if _body_depth(p.codes[b], self.ar) > lim:
                return False
        return True

    # ---- initialisation ---------------------------------------------------------------
    def _random_program(self, out_depth, method, rng, frozen=None):
        cfg = self.cfg
        L = cfg.levels
        levels, codes, args = [], [], []
        start = 1
        if frozen is not None:
            levels, codes, args = (list(frozen[0]), [list(c) for c in frozen[1]], [list(a) for a in frozen[2]])
            start = L
        for lvl in range(start, L + 1):
            n = int(rng.integers(cfg.init_modules[0], cfg.init_modules[1] + 1))
            n = min(max(n, cfg.min_modules), cfg.max_modules)
            for _ in range(n):
                t = self.targets(levels, lvl)
                d = int(rng.integers(cfg.module_init_depth[0], cfg.module_init_depth[1] + 1))
                c, a = self._gen(t, d, "grow" if rng.random() < 0.5 else "full", rng)
                levels.append(lvl)
                codes.append(c)
                args.append(a)
        c, a = self._gen(self.targets(levels, L + 1), out_depth, method, rng)
        codes.append(c)
        args.append(a)
        return self._finish(levels, codes, args)

    def init_population(self, n: int, rng) -> list[GraphProgram]:
        cfg = self.cfg
        if cfg.frozen_below:
            self.frozen = self._draw_frozen(rng)
        lo, hi = cfg.out_init_depth
        depths = list(range(lo, hi + 1))
        pop, seen = [], set()
        lowered = dup = 0
        for i in range(n):
            depth = depths[i % len(depths)]
            method = "full" if (i // len(depths)) % 2 == 0 else "grow"
            chosen = None
            for d in range(depth, 0, -1):
                for _ in range(20):
                    p = self._random_program(d, method, rng, self.frozen)
                    if self.is_valid(p) and p.key() not in seen:
                        chosen = p
                        break
                if chosen is not None:
                    lowered += d < depth
                    break
            if chosen is None:
                dup += 1
                while chosen is None or not self.is_valid(chosen):
                    chosen = self._random_program(1, method, rng, self.frozen)
            seen.add(chosen.key())
            pop.append(chosen)
        self.init_stats = {"init_depth_lowered": lowered, "init_duplicates": dup}
        return pop

    def _draw_frozen(self, rng):
        """Random levels 1..L-1 shared by the whole population (A5-frozen).
        Every module of the frozen levels is kept, active or not."""
        cfg = self.cfg
        levels, codes, args = [], [], []
        for lvl in range(1, cfg.levels):
            for _ in range(cfg.max_modules // 2 or 1):
                t = self.targets(levels, lvl)
                d = int(rng.integers(cfg.module_init_depth[0], cfg.module_init_depth[1] + 1))
                c, a = self._gen(t, d, "grow" if rng.random() < 0.5 else "full", rng)
                levels.append(lvl)
                codes.append(c)
                args.append(a)
        return levels, codes, args

    # ---- housekeeping --------------------------------------------------------------------
    def _finish(self, levels, codes, args) -> GraphProgram:
        """Sort modules by level (stable), drop modules the output cannot
        reach (keeping at least one per level, and all frozen modules), and
        re-index every REF."""
        L = self.cfg.levels
        if self.cfg.private_copies:
            levels, codes, args = self._unshare(levels, codes, args)
        n = len(levels)
        order = sorted(range(n), key=lambda k: levels[k])
        pos = {k: i for i, k in enumerate(order)}
        lv = [levels[k] for k in order]
        cd = [codes[k] for k in order] + [codes[n]]
        ag = [[pos[a] if (c == REF and a >= 0) else a for c, a in zip(codes[k], args[k])] for k in order]
        ag.append([pos[a] if (c == REF and a >= 0) else a for c, a in zip(codes[n], args[n])])
        # reachability from the output
        act = [False] * n
        stack = [a for c, a in zip(cd[n], ag[n]) if c == REF and a >= 0]
        while stack:
            k = stack.pop()
            if not act[k]:
                act[k] = True
                stack.extend(a for c, a in zip(cd[k], ag[k]) if c == REF and a >= 0)
        n_frozen = len(self.frozen[0]) if self.frozen is not None else 0
        keep = [act[k] or k < n_frozen for k in range(n)]
        for lvl in range(1, L + 1):
            ks = [k for k in range(n) if lv[k] == lvl]
            need = self.cfg.min_modules - sum(keep[k] for k in ks)
            for k in [k for k in ks if not keep[k]][: max(0, need)]:
                keep[k] = True
        new = {}
        for k in range(n):
            if keep[k]:
                new[k] = len(new)
        # A kept but inactive module may read a removed one: re-point it.
        out_lv, out_cd, out_ag = [], [], []
        for k in range(n):
            if keep[k]:
                out_lv.append(lv[k])
        for k in list(range(n)) + [n]:
            if k < n and not keep[k]:
                continue
            a2 = []
            for c, a in zip(cd[k], ag[k]):
                if c == REF and a >= 0:
                    if a in new:
                        a = new[a]
                    else:
                        same = [new[j] for j in range(n) if keep[j] and lv[j] == lv[a]]
                        a = same[0]
                a2.append(a)
            out_cd.append(cd[k])
            out_ag.append(a2)
        return self._pack(out_lv, out_cd, out_ag)

    @staticmethod
    def _unshare(levels, codes, args):
        """A5-copy: give every module read its own module, so nothing is
        shared, neither computation nor genes. Walking down from the output,
        a REF to a module that another REF already owns is re-pointed to a
        fresh copy of that module (whose own reads are then processed in the
        same way). The result computes the same function."""
        levels = list(levels)
        mods_c = [list(c) for c in codes[:-1]]
        mods_a = [list(a) for a in args[:-1]]
        out_c, out_a = list(codes[-1]), list(args[-1])
        owned = set()
        work = [("out", None)]
        while work:
            _, b = work.pop()
            bc, ba = (out_c, out_a) if b is None else (mods_c[b], mods_a[b])
            for j, (c, a) in enumerate(zip(bc, ba)):
                if c == REF and a >= 0:
                    if a in owned:
                        levels.append(levels[a])
                        mods_c.append(list(mods_c[a]))
                        mods_a.append(list(mods_a[a]))
                        a = len(levels) - 1
                        ba[j] = a
                    owned.add(a)
                    work.append(("mod", a))
        return levels, mods_c + [out_c], mods_a + [out_a]

    # ---- operators ----------------------------------------------------------------------
    def _evolvable(self, levels):
        """Bodies variation may touch (all, or level L and the output when frozen)."""
        n = len(levels)
        if self.cfg.frozen_below:
            return [k for k in range(n) if levels[k] == self.cfg.levels] + [n]
        return list(range(n + 1))

    def _choose_body(self, p, rng):
        cand = self._evolvable(p.levels)
        sizes = np.array([len(p.codes[b]) for b in cand], dtype=float)
        return cand[int(rng.choice(len(cand), p=sizes / sizes.sum()))]

    def _pick(self, codes, rng):
        internal = [i for i, c in enumerate(codes) if self.ar[c] > 0]
        if internal and rng.random() < self.cfg.p_internal:
            return internal[int(rng.integers(len(internal)))]
        leaves = [i for i, c in enumerate(codes) if self.ar[c] == 0]
        return leaves[int(rng.integers(len(leaves)))]

    def _level_of(self, p_levels, b):
        return p_levels[b] if b < len(p_levels) else self.cfg.levels + 1

    def crossover(self, p1: GraphProgram, p2: GraphProgram, rng) -> GraphProgram:
        levels, codes, args = self._unpack(p1)
        b = self._choose_body(p1, rng)
        lvl = self._level_of(p1.levels, b)
        # aligned partner: same level and rank within the level, else a random one of that level
        if b == p1.out:
            pb = p2.out
        else:
            rank = [k for k, l in enumerate(p1.levels) if l == lvl].index(b)
            same = [k for k, l in enumerate(p2.levels) if l == lvl]
            pb = same[rank] if rank < len(same) else same[int(rng.integers(len(same)))]
        i = self._pick(codes[b], rng)
        ie = _subtree_end(codes[b], i, self.ar)
        j = self._pick(list(p2.codes[pb]), rng)
        je = _subtree_end(p2.codes[pb], j, self.ar)
        legal = self.targets(levels, lvl)
        lset = set(legal)
        sub_c = list(p2.codes[pb][j:je])
        sub_a = []
        for c, a in zip(sub_c, p2.args[pb][j:je]):
            if c == REF:
                a = self._map_ref(p2, a, p1, lset, legal, rng)
            sub_a.append(a)
        codes[b] = codes[b][:i] + sub_c + codes[b][ie:]
        args[b] = args[b][:i] + sub_a + args[b][ie:]
        return self._finish(levels, codes, args)

    def _map_ref(self, src, a, dst, lset, legal, rng):
        """Translate a REF target of `src` into `dst`: inputs map to
        themselves, a module to the module of the same level and rank; a
        target that is not legal at the receiving position is re-drawn."""
        if a >= 0:
            lvl = src.levels[a]
            rank = [k for k, l in enumerate(src.levels) if l == lvl].index(a)
            same = [k for k, l in enumerate(dst.levels) if l == lvl]
            a = same[rank] if rank < len(same) else None
        if a is None or a not in lset:
            a = legal[int(rng.integers(len(legal)))]
        return a

    def mutate(self, p: GraphProgram, rng) -> GraphProgram:
        levels, codes, args = self._unpack(p)
        b = self._choose_body(p, rng)
        lvl = self._level_of(p.levels, b)
        i = self._pick(codes[b], rng)
        ie = _subtree_end(codes[b], i, self.ar)
        d = int(rng.integers(0, self.cfg.mutation_depth + 1))
        sc, sa = self._gen(self.targets(levels, lvl), d, "grow", rng)
        codes[b] = codes[b][:i] + sc + codes[b][ie:]
        args[b] = args[b][:i] + sa + args[b][ie:]
        return self._finish(levels, codes, args)

    def rewire(self, p: GraphProgram, rng) -> GraphProgram:
        levels, codes, args = self._unpack(p)
        cand = [(b, i) for b in self._evolvable(p.levels) for i, c in enumerate(codes[b]) if c == REF]
        if not cand:
            return self.mutate(p, rng)
        b, i = cand[int(rng.integers(len(cand)))]
        legal = [t for t in self.targets(levels, self._level_of(p.levels, b)) if t != args[b][i]]
        if not legal:
            return self.mutate(p, rng)
        args[b][i] = legal[int(rng.integers(len(legal)))]
        return self._finish(levels, codes, args)

    def encapsulate(self, p: GraphProgram, rng) -> GraphProgram:
        """Copy a subtree S (depth >= 1) of a level-l module M into a new
        level-l module N, then re-point one REF of a level-(l+1) body (the
        output tree if l = L) that reads level l to N."""
        L = self.cfg.levels
        levels, codes, args = self._unpack(p)
        mods = [k for k in self._evolvable(p.levels) if k < p.out
                and sum(1 for l in levels if l == levels[k]) < self.cfg.max_modules
                and any(self.ar[c] > 0 for c in codes[k])]
        if not mods:
            return self.mutate(p, rng)
        m = mods[int(rng.integers(len(mods)))]
        lvl = levels[m]
        internal = [i for i, c in enumerate(codes[m]) if self.ar[c] > 0]
        i = internal[int(rng.integers(len(internal)))]
        ie = _subtree_end(codes[m], i, self.ar)
        new = len(levels)
        levels.append(lvl)
        codes.insert(new, codes[m][i:ie])
        args.insert(new, args[m][i:ie])
        out = new + 1
        readers = [b for b in range(out + 1)
                   if (levels[b] if b < out else L + 1) == lvl + 1]
        sites = [(b, j) for b in readers for j, (c, a) in enumerate(zip(codes[b], args[b]))
                 if c == REF and a >= 0 and levels[a] == lvl]
        if not sites:
            sites = [(b, j) for b in readers for j, c in enumerate(codes[b]) if c == REF]
        if not sites:
            return self.mutate(p, rng)
        b, j = sites[int(rng.integers(len(sites)))]
        args[b][j] = new
        return self._finish(levels, codes, args)

    def delete(self, p: GraphProgram, rng) -> GraphProgram:
        levels, codes, args = self._unpack(p)
        cand = [k for k in self._evolvable(p.levels) if k < p.out
                and sum(1 for l in levels if l == levels[k]) > self.cfg.min_modules]
        if not cand:
            return self.mutate(p, rng)
        d = cand[int(rng.integers(len(cand)))]
        others = [k for k, l in enumerate(levels) if l == levels[d] and k != d]
        for b in range(len(codes)):
            for j, (c, a) in enumerate(zip(codes[b], args[b])):
                if c == REF and a == d:
                    args[b][j] = others[int(rng.integers(len(others)))]
        # drop d and shift indices above it
        del levels[d], codes[d], args[d]
        for b in range(len(codes)):
            args[b] = [(a - 1 if (c == REF and a > d) else a) for c, a in zip(codes[b], args[b])]
        return self._finish(levels, codes, args)

    def vary(self, parents, rng) -> GraphProgram:
        cfg = self.cfg
        p1, p2 = parents[0], parents[1]
        rates = np.array([cfg.p_crossover, cfg.p_mutation, cfg.p_rewire, cfg.p_encapsulate, cfg.p_delete])
        op = int(rng.choice(5, p=rates / rates.sum()))
        for _ in range(cfg.max_tries):
            if op == 0:
                child = self.crossover(p1, p2, rng)
            elif op == 1:
                child = self.mutate(p1, rng)
            elif op == 2:
                child = self.rewire(p1, rng)
            elif op == 3:
                child = self.encapsulate(p1, rng)
            else:
                child = self.delete(p1, rng)
            if self.is_valid(child):
                return child
        return p1

    def describe(self) -> dict:
        d = {"arm": self.name, "functions": [f.name for f in self.pset.functions],
             "n_vars": self.pset.n_vars, "erc": self.pset.erc}
        d.update({k: v for k, v in self.cfg.__dict__.items() if k != "extra"})
        return d


VARIANTS = {
    "a5": {},
    "a6": {"connectivity": "dense"},
    "a6_inputs": {"connectivity": "inputs"},
    # Private copies: the per-level module limit is lifted (the expanded-size cap,
    # identical for every arm, is what bounds the program).
    "a5_copy": {"cached": False, "private_copies": True, "max_modules": 64},
    "a5_frozen": {"frozen_below": True},
}


def make_hier(name: str, pset: PrimitiveSet, cfg: HierArmConfig | None = None) -> HierArm:
    from dataclasses import replace

    cfg = replace(cfg or HierArmConfig(), **VARIANTS[name])
    return HierArm(pset, cfg, name=name)

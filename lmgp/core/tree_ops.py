"""Low-level operations on prefix-ordered branches: generation, node choice and
subtree replacement. All randomness comes from an explicit numpy Generator."""
from __future__ import annotations

import numpy as np

from .program import Architecture, Kind


def generate(arch: Architecture, branch: int, depth: int, method: str, rng: np.random.Generator):
    """Generate a branch with the full or grow method (Koza 1992).

    Grow chooses uniformly from the union of functions and terminals at every
    node above the depth limit; full chooses only functions until the limit.
    Returns (codes, vals) lists.
    """
    funcs = arch.branch_functions[branch]
    terms = arch.branch_terminals[branch]
    both = funcs + terms
    codes: list[int] = []
    vals: list[float] = []
    lo, hi = arch.pset.erc_range

    def add_terminal():
        c = terms[rng.integers(len(terms))]
        codes.append(c)
        vals.append(float(rng.uniform(lo, hi)) if c == arch.erc_code else 0.0)

    def rec(d):
        if d == 0:
            add_terminal()
            return
        if method == "full":
            c = funcs[rng.integers(len(funcs))]
        else:
            c = both[rng.integers(len(both))]
            if arch.arity[c] == 0:
                codes.append(c)
                vals.append(float(rng.uniform(lo, hi)) if c == arch.erc_code else 0.0)
                return
        codes.append(c)
        vals.append(0.0)
        for _ in range(arch.arity[c]):
            rec(d - 1)

    rec(depth)
    return codes, vals


def subtree_end(codes, start: int, arity) -> int:
    """Index one past the end of the subtree rooted at `start`."""
    need = 1
    j = start
    while need:
        need += arity[codes[j]] - 1
        j += 1
    return j


def pick_node(codes, arity, rng: np.random.Generator, p_internal: float = 0.9) -> int:
    """Koza's 90/10 rule: an internal node with probability p_internal (if any
    exist), otherwise a leaf; uniform within the chosen class."""
    internal = [i for i, c in enumerate(codes) if arity[c] > 0]
    if internal and rng.random() < p_internal:
        return internal[rng.integers(len(internal))]
    leaves = [i for i, c in enumerate(codes) if arity[c] == 0]
    return leaves[rng.integers(len(leaves))]


def replace_subtree(codes, vals, start, end, sub_codes, sub_vals):
    return (
        list(codes[:start]) + list(sub_codes) + list(codes[end:]),
        list(vals[:start]) + list(sub_vals) + list(vals[end:]),
    )


def is_well_formed(arch: Architecture, branch: int, codes) -> bool:
    """True if `codes` is a complete prefix tree using only symbols allowed in
    `branch`."""
    allowed = set(arch.branch_functions[branch]) | set(arch.branch_terminals[branch])
    need = 1
    for i, c in enumerate(codes):
        if c not in allowed or need <= 0:
            return False
        need += arch.arity[c] - 1
    return need == 0 and len(codes) > 0


__all__ = ["generate", "subtree_end", "pick_node", "replace_subtree", "is_well_formed", "Kind"]

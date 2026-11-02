"""A1: standard tree GP with size and depth limits (main baseline)."""
from __future__ import annotations

from dataclasses import replace

from ..core.primitives import PrimitiveSet
from .tree import TreeArm, TreeArmConfig


def make_a1(pset: PrimitiveSet, cfg: TreeArmConfig | None = None, **overrides) -> TreeArm:
    cfg = replace(cfg or TreeArmConfig(), adf_arities=(), **overrides)
    return TreeArm(pset, cfg, name="a1")

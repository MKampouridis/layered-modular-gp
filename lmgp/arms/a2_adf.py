"""A2: tree GP with automatically defined functions (Koza 1994).

Default architecture: two ADFs, ADF0 with two arguments and ADF1 with three;
ADF1 may call ADF0 (hierarchical), and ADF bodies see only their arguments.
This follows the shape of Koza's even-parity architectures in GP II (a small
fixed set of hierarchical ADFs); the exact arities Koza used per problem were
not checked against the book and are set in config. The result-producing branch sees
the inputs and may call every ADF.

Every ADF call is charged in full (call node + arguments + body), even when
two calls receive identical arguments: no caching, so ADFs get no discount
that plain trees could not also get.
"""
from __future__ import annotations

from dataclasses import replace

from ..core.primitives import PrimitiveSet
from .tree import TreeArm, TreeArmConfig

DEFAULT_ADF_ARITIES = (2, 3)


def make_a2(pset: PrimitiveSet, cfg: TreeArmConfig | None = None, **overrides) -> TreeArm:
    cfg = cfg or TreeArmConfig()
    if not cfg.adf_arities and "adf_arities" not in overrides:
        overrides["adf_arities"] = DEFAULT_ADF_ARITIES
    cfg = replace(cfg, **overrides)
    return TreeArm(pset, cfg, name="a2")

"""Primitive (function and terminal) sets.

Every primitive costs one unit per training case when executed. The function
implementations are NumPy-vectorised but also accept scalars, so the same code
serves the vectorised evaluator and the scalar reference interpreter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


def _aq(a, b):
    """Analytic quotient (Ni, Drieberg and Rockett 2013): a / sqrt(1 + b^2)."""
    return a / np.sqrt(1.0 + b * b)


def _if(c, a, b):
    return np.where(c, a, b)


def _nand(a, b):
    return np.logical_not(np.logical_and(a, b))


def _nor(a, b):
    return np.logical_not(np.logical_or(a, b))


@dataclass(frozen=True)
class Function:
    name: str
    arity: int
    fn: Callable


FUNCTIONS: dict[str, Function] = {
    f.name: f
    for f in [
        Function("add", 2, np.add),
        Function("sub", 2, np.subtract),
        Function("mul", 2, np.multiply),
        Function("aq", 2, _aq),
        Function("sin", 1, np.sin),
        Function("cos", 1, np.cos),
        Function("and", 2, np.logical_and),
        Function("or", 2, np.logical_or),
        Function("nand", 2, _nand),
        Function("nor", 2, _nor),
        Function("not", 1, np.logical_not),
        Function("if", 3, _if),
    ]
}


@dataclass(frozen=True)
class PrimitiveSet:
    """Base functions, number of input variables, and whether ERCs are used."""

    functions: tuple[Function, ...]
    n_vars: int
    erc: bool = False
    erc_range: tuple[float, float] = (-1.0, 1.0)
    dtype: type = field(default=float)

    @staticmethod
    def from_names(names, n_vars, erc=False, erc_range=(-1.0, 1.0), dtype=float):
        return PrimitiveSet(tuple(FUNCTIONS[n] for n in names), n_vars, erc, tuple(erc_range), dtype)


REGRESSION = ("add", "sub", "mul", "aq", "sin", "cos")
PARITY = ("and", "or", "nand", "nor")
MULTIPLEXER = ("and", "or", "not", "if")


def regression_set(n_vars: int, erc_range=(-1.0, 1.0)) -> PrimitiveSet:
    return PrimitiveSet.from_names(REGRESSION, n_vars, erc=True, erc_range=erc_range, dtype=float)


def parity_set(n_vars: int) -> PrimitiveSet:
    return PrimitiveSet.from_names(PARITY, n_vars, dtype=bool)


def multiplexer_set(n_vars: int) -> PrimitiveSet:
    return PrimitiveSet.from_names(MULTIPLEXER, n_vars, dtype=bool)

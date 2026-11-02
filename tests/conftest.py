import numpy as np
import pytest

from lmgp.arms.a1_tree import make_a1
from lmgp.arms.a2_adf import make_a2
from lmgp.arms.tree import TreeArmConfig
from lmgp.core.primitives import multiplexer_set, parity_set, regression_set

PSETS = {
    "regression": lambda: regression_set(3),
    "parity": lambda: parity_set(4),
    "mux": lambda: multiplexer_set(6),
}


def arms():
    out = []
    for pname, mk in PSETS.items():
        out.append((f"a1-{pname}", make_a1(mk(), TreeArmConfig(max_size=200, max_depth=10))))
        out.append((f"a2-{pname}", make_a2(mk(), TreeArmConfig(max_size=200, max_depth=10))))
        out.append((f"a2flat-{pname}", make_a2(mk(), TreeArmConfig(max_size=200, max_depth=10),
                                                adf_arities=(1, 2, 3), hierarchical=False,
                                                adf_sees_inputs=True)))
    return out


ARMS = arms()


@pytest.fixture(params=ARMS, ids=[a[0] for a in ARMS])
def arm(request):
    return request.param[1]


def data_for(arm, n=37, seed=0):
    rng = np.random.default_rng(seed)
    nv = arm.arch.pset.n_vars
    if arm.arch.pset.dtype is bool:
        return rng.random((n, nv)) < 0.5
    return rng.uniform(-3, 3, size=(n, nv))

import numpy as np
from mtsgamma.solver import mts_gamma_C4


def test_solver_returns_cycle():
    coords = np.random.rand(20, 2) * 100
    route, length = mts_gamma_C4(coords)
    assert route.shape[0] == coords.shape[0]
    assert np.isfinite(length)

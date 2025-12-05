import numpy as np
from mtsgamma.refine import refine_c4, tour_length


def test_refine_reduces_length():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
    order = np.array([0, 2, 1, 3, 0], dtype=np.int32)
    initial = tour_length(order, coords)
    new_route, new_len = refine_c4(order, coords)
    assert new_len <= initial + 1e-9
    assert new_route.shape[0] == order.shape[0]

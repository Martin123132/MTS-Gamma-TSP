import numpy as np
from mtsgamma.field import build_field


def test_build_field_shape():
    coords = np.array([[10.0, 10.0], [20.0, 20.0]])
    field = build_field(coords, grid=64, iter_gamma=5, smooth=1.0)
    assert field.shape == (64, 64)
    assert np.isfinite(field).all()

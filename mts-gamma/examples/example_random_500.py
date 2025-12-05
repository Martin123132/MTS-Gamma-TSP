import numpy as np

from mtsgamma import mts_gamma_C4, christofides_route, tour_length
from mtsgamma.field import DEFAULT_GRID

if __name__ == "__main__":
    coords = np.random.rand(500, 2) * (DEFAULT_GRID - 4)
    c_route = christofides_route(coords)
    c_len = tour_length(c_route, coords)
    route, m_len = mts_gamma_C4(coords)
    print("Christofides:", c_len)
    print("MTS–Gamma C4:", m_len)
    print("Improvement %:", (c_len - m_len) / c_len * 100)

from mtsgamma import load_embedded, mts_gamma_C4, christofides_route, tour_length

if __name__ == "__main__":
    coords = load_embedded("berlin52")
    c_route = christofides_route(coords)
    c_len = tour_length(c_route, coords)
    route, m_len = mts_gamma_C4(coords)
    print("Christofides:", c_len)
    print("MTS–Gamma C4:", m_len)
    print("Improvement %:", (c_len - m_len) / c_len * 100)

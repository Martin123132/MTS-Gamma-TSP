"""Route refinement routines (2.5-opt, 3-opt, C4 pipeline).

Numba-accelerated versions mirror the fastest C4 implementation from the
source fragments. Pure-Python fallbacks are provided for environments
without Numba.
"""
from __future__ import annotations

import numpy as np

try:  # pragma: no cover - optional dependency
    import numba as nb
except Exception:  # pragma: no cover
    nb = None


def _dist_py(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _tour_length_py(order: np.ndarray, coords: np.ndarray) -> float:
    return sum(_dist_py(coords[order[i]], coords[order[i + 1]]) for i in range(len(order) - 1))


def _two_point_five_opt_py(route: np.ndarray, coords: np.ndarray) -> np.ndarray:
    route = route.tolist()
    improved = True
    n = len(route)
    while improved:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 2, n - 1):
                A, B = route[i - 1], route[i]
                C, D = route[j], route[j + 1]
                old = _dist_py(coords[A], coords[B]) + _dist_py(coords[C], coords[D])
                new = _dist_py(coords[A], coords[C]) + _dist_py(coords[B], coords[D])
                if new < old:
                    route[i : j + 1] = reversed(route[i : j + 1])
                    improved = True
                    continue
    return np.array(route, dtype=np.int32)


def _three_opt_py(route: np.ndarray, coords: np.ndarray) -> np.ndarray:
    route = route.tolist()
    n = len(route)
    improved = True
    while improved:
        improved = False
        for i in range(0, n - 3):
            for j in range(i + 2, n - 2):
                for k in range(j + 2, n - 1):
                    old = (
                        _dist_py(coords[route[i]], coords[route[i + 1]])
                        + _dist_py(coords[route[j]], coords[route[j + 1]])
                        + _dist_py(coords[route[k]], coords[route[k + 1]])
                    )

                    # Option 1
                    r1 = route.copy()
                    r1[i + 1 : j + 1] = reversed(r1[i + 1 : j + 1])
                    c1 = (
                        _dist_py(coords[r1[i]], coords[r1[i + 1]])
                        + _dist_py(coords[r1[j]], coords[r1[j + 1]])
                        + _dist_py(coords[r1[k]], coords[r1[k + 1]])
                    )

                    # Option 2
                    r2 = route.copy()
                    r2[j + 1 : k + 1] = reversed(r2[j + 1 : k + 1])
                    c2 = (
                        _dist_py(coords[r2[i]], coords[r2[i + 1]])
                        + _dist_py(coords[r2[j]], coords[r2[j + 1]])
                        + _dist_py(coords[r2[k]], coords[r2[k + 1]])
                    )

                    # Option 3
                    r3 = route.copy()
                    r3[i + 1 : j + 1] = reversed(r3[i + 1 : j + 1])
                    r3[j + 1 : k + 1] = reversed(r3[j + 1 : k + 1])
                    c3 = (
                        _dist_py(coords[r3[i]], coords[r3[i + 1]])
                        + _dist_py(coords[r3[j]], coords[r3[j + 1]])
                        + _dist_py(coords[r3[k]], coords[r3[k + 1]])
                    )

                    best = min(c1, c2, c3)
                    if best < old - 1e-12:
                        if best == c1:
                            route = r1
                        elif best == c2:
                            route = r2
                        else:
                            route = r3
                        improved = True
    return np.array(route, dtype=np.int32)


if nb:

    @nb.njit(fastmath=True)
    def dist(a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    @nb.njit(fastmath=True)
    def tour_length(order, coords):
        s = 0.0
        for i in range(len(order) - 1):
            s += dist(coords[order[i]], coords[order[i + 1]])
        return s

    @nb.njit(fastmath=True)
    def two_point_five_opt(route, coords):
        n = len(route)
        improved = True
        while improved:
            improved = False
            for i in range(1, n - 2):
                A = route[i - 1]
                B = route[i]
                for j in range(i + 2, n - 1):
                    C = route[j]
                    D = route[j + 1]
                    old = dist(coords[A], coords[B]) + dist(coords[C], coords[D])
                    new = dist(coords[A], coords[C]) + dist(coords[B], coords[D])
                    if new < old:
                        l = j - i + 1
                        tmp = route[i : j + 1].copy()
                        for k in range(l):
                            route[i + k] = tmp[l - 1 - k]
                        improved = True
        return route

    @nb.njit(fastmath=True)
    def three_opt(route, coords):
        n = len(route)
        improved = True
        while improved:
            improved = False
            for i in range(0, n - 3):
                A = route[i]
                B = route[i + 1]
                for j in range(i + 2, n - 2):
                    C = route[j]
                    D = route[j + 1]
                    for k in range(j + 2, n - 1):
                        E = route[k]
                        F = route[k + 1]
                        old = dist(coords[A], coords[B]) + dist(coords[C], coords[D]) + dist(coords[E], coords[F])

                        new1 = dist(coords[A], coords[C]) + dist(coords[B], coords[D]) + dist(coords[E], coords[F])
                        if new1 < old:
                            tmp = route[i + 1 : j + 1].copy()
                            ln = j - i
                            for q in range(ln):
                                route[i + 1 + q] = tmp[ln - 1 - q]
                            improved = True
                            continue

                        new2 = dist(coords[A], coords[B]) + dist(coords[C], coords[E]) + dist(coords[D], coords[F])
                        if new2 < old:
                            tmp = route[j + 1 : k + 1].copy()
                            ln = k - j
                            for q in range(ln):
                                route[j + 1 + q] = tmp[ln - 1 - q]
                            improved = True
                            continue

                        new3 = dist(coords[A], coords[C]) + dist(coords[B], coords[E]) + dist(coords[D], coords[F])
                        if new3 < old:
                            tmp = route[i + 1 : j + 1].copy()
                            ln = j - i
                            for q in range(ln):
                                route[i + 1 + q] = tmp[ln - 1 - q]
                            tmp2 = route[j + 1 : k + 1].copy()
                            ln2 = k - j
                            for q in range(ln2):
                                route[j + 1 + q] = tmp2[ln2 - 1 - q]
                            improved = True
                            continue
        return route

    def refine_c4(order: np.ndarray, coords: np.ndarray) -> tuple[np.ndarray, float]:
        route = two_point_five_opt(order.astype(np.int32), coords)
        route = three_opt(route.astype(np.int32), coords)
        route = two_point_five_opt(route.astype(np.int32), coords)
        return route, float(tour_length(route, coords))

else:
    dist = _dist_py
    tour_length = _tour_length_py
    two_point_five_opt = _two_point_five_opt_py
    three_opt = _three_opt_py

    def refine_c4(order: np.ndarray, coords: np.ndarray) -> tuple[np.ndarray, float]:
        route = two_point_five_opt(order.astype(np.int32), coords)
        route = three_opt(route.astype(np.int32), coords)
        route = two_point_five_opt(route.astype(np.int32), coords)
        return route, float(tour_length(route, coords))

__all__ = [
    "dist",
    "tour_length",
    "two_point_five_opt",
    "three_opt",
    "refine_c4",
]

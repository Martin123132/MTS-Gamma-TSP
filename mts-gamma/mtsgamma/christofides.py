"""Christofides baseline implementation (NetworkX)."""
from __future__ import annotations

import numpy as np
import networkx as nx
from networkx.algorithms import approximation as approx

from .refine import dist


def christofides_route(coords: np.ndarray) -> np.ndarray:
    """Return a Christofides tour (cycle=True) for Euclidean coords."""

    n = len(coords)
    G = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(i, j, weight=float(dist(coords[i], coords[j])))
    route = approx.christofides(G)
    return np.array(route, dtype=np.int32)

__all__ = ["christofides_route"]

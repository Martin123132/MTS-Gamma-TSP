"""Stability testing for MTS–Gamma across seeds and sizes."""
from __future__ import annotations

import csv
import time
from typing import Iterable

import numpy as np

from .solver import mts_gamma_C4
from .christofides import christofides_route
from .refine import tour_length
from .field import DEFAULT_GRID


DEFAULT_SIZES = [100, 300, 500, 1000]
DEFAULT_SEEDS = 20


def run_stability_tests(
    sizes: Iterable[int] = DEFAULT_SIZES,
    seeds: int = DEFAULT_SEEDS,
    grid: int = DEFAULT_GRID,
    csv_path: str = "mts_gamma_stability.csv",
) -> None:
    """Execute multi-seed comparisons and write CSV."""

    rows = [["N", "seed", "christofides", "mts_gamma", "improvement_pct", "runtime_sec"]]

    for N in sizes:
        for seed in range(seeds):
            np.random.seed(seed)
            coords = np.random.rand(N, 2) * (grid - 4)

            start = time.time()
            c_route = christofides_route(coords)
            c_len = tour_length(c_route, coords)
            c_time = time.time() - start

            start = time.time()
            _, m_len = mts_gamma_C4(coords)
            m_time = time.time() - start

            imp = (c_len - m_len) / c_len * 100.0
            rows.append([N, seed, c_len, m_len, imp, c_time + m_time])

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)

__all__ = ["run_stability_tests", "DEFAULT_SIZES", "DEFAULT_SEEDS"]

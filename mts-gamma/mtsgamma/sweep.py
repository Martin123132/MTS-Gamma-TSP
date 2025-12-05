"""Parameter sweep engine for MTS–Gamma."""
from __future__ import annotations

import csv
import itertools
import time
from typing import Iterable

import numpy as np

from .solver import SolverParams, mts_gamma_C4
from .christofides import christofides_route
from .refine import tour_length


def parameter_sweep(
    coords: np.ndarray,
    gammas: Iterable[float],
    smooth_values: Iterable[float],
    flow_steps: Iterable[int],
    step_sizes: Iterable[float],
    csv_path: str,
) -> None:
    """Run a grid search over the provided parameter ranges and log to CSV."""

    rows = [["gamma", "smooth", "flow_steps", "step_size", "mts_len", "improvement_pct", "runtime_sec"]]
    base_route = christofides_route(coords)
    base_len = tour_length(base_route, coords)

    for gamma, smooth, fs, step in itertools.product(gammas, smooth_values, flow_steps, step_sizes):
        params = SolverParams(gamma=gamma, smooth=smooth, flow_steps=fs, step_size=step)
        start = time.time()
        _, mts_len = mts_gamma_C4(coords, params=params)
        runtime = time.time() - start
        imp = (base_len - mts_len) / base_len * 100.0
        rows.append([gamma, smooth, fs, step, mts_len, imp, runtime])

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)

__all__ = ["parameter_sweep"]

"""End-to-end MTS–Gamma solver with C4 refinement."""
from __future__ import annotations

import time
import numpy as np

from .field import build_field, DEFAULT_GRID, DEFAULT_GAMMA, DEFAULT_ITER_GAMMA, DEFAULT_SMOOTH
from .flow import gradient_flow, DEFAULT_FLOW_STEPS, DEFAULT_STEP_SIZE
from .refine import refine_c4, tour_length
from .christofides import christofides_route


class SolverParams:
    def __init__(
        self,
        grid: int = DEFAULT_GRID,
        gamma: float = DEFAULT_GAMMA,
        iter_gamma: int = DEFAULT_ITER_GAMMA,
        smooth: float = DEFAULT_SMOOTH,
        flow_steps: int = DEFAULT_FLOW_STEPS,
        step_size: float = DEFAULT_STEP_SIZE,
    ) -> None:
        self.grid = grid
        self.gamma = gamma
        self.iter_gamma = iter_gamma
        self.smooth = smooth
        self.flow_steps = flow_steps
        self.step_size = step_size


def mts_gamma_C4(coords: np.ndarray, params: SolverParams | None = None) -> tuple[np.ndarray, float]:
    """Run the full MTS–Gamma C4 pipeline and return the refined tour and length."""

    p = params or SolverParams()
    field = build_field(
        coords,
        grid=p.grid,
        gamma=p.gamma,
        iter_gamma=p.iter_gamma,
        smooth=p.smooth,
    )
    order = gradient_flow(field, coords, flow_steps=p.flow_steps, step_size=p.step_size, grid=p.grid)
    return refine_c4(order.astype(np.int32), coords)


def run_test(N: int = 500, seed: int | None = None, params: SolverParams | None = None) -> dict[str, float]:
    """Generate random coordinates and compare Christofides vs MTS–Gamma."""

    if seed is not None:
        np.random.seed(seed)
    coords = np.random.rand(N, 2) * (DEFAULT_GRID - 4)

    start = time.time()
    c_route = christofides_route(coords)
    christofides_len = tour_length(c_route.astype(np.int32), coords)
    c_time = time.time() - start

    start = time.time()
    mts_route, mts_len = mts_gamma_C4(coords, params=params)
    m_time = time.time() - start

    return {
        "christofides": christofides_len,
        "mts_gamma": mts_len,
        "improvement_pct": (christofides_len - mts_len) / christofides_len * 100.0,
        "christofides_time": c_time,
        "mts_time": m_time,
    }

__all__ = ["mts_gamma_C4", "run_test", "SolverParams"]

"""Gradient flow ordering used by MTS–Gamma."""
from __future__ import annotations

import math
import numpy as np

from .field import DEFAULT_GRID

DEFAULT_FLOW_STEPS = 450
DEFAULT_STEP_SIZE = 1.2


def gradient_flow(
    field: np.ndarray,
    coords: np.ndarray,
    flow_steps: int = DEFAULT_FLOW_STEPS,
    step_size: float = DEFAULT_STEP_SIZE,
    grid: int = DEFAULT_GRID,
) -> np.ndarray:
    """Compute ordering by following gradient ascent on the field.

    Each point is advanced ``flow_steps`` steps along the gradient with
    boundary clamping after every update. The final field values are
    used to rank cities in descending order.
    """

    grad_y, grad_x = np.gradient(field)
    flow_val = np.zeros(len(coords), dtype=np.float64)

    for i, (x0, y0) in enumerate(coords):
        x = float(np.clip(x0, 1, grid - 2))
        y = float(np.clip(y0, 1, grid - 2))

        for _ in range(flow_steps):
            xi = int(x)
            yi = int(y)
            gx = grad_x[yi, xi]
            gy = grad_y[yi, xi]
            mag = math.hypot(gx, gy)
            if mag < 1e-9:
                break
            x += step_size * gx / mag
            y += step_size * gy / mag
            x = float(np.clip(x, 1, grid - 2))
            y = float(np.clip(y, 1, grid - 2))

        flow_val[i] = field[int(y), int(x)]

    return np.argsort(-flow_val)

__all__ = ["gradient_flow", "DEFAULT_FLOW_STEPS", "DEFAULT_STEP_SIZE"]

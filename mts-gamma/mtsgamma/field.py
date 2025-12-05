"""Curvature field construction for MTS–Gamma.

This module builds the smoothed occupancy field and applies
Gamma-diffusion (Laplacian smoothing) used to guide the gradient
flow ordering. Parameters mirror the fastest published configuration
from the provided fragments (GRID=512, GAMMA=0.16, ITER_GAMMA=400,
SMOOTH_FINE=1.6).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

DEFAULT_GRID = 512
DEFAULT_GAMMA = 0.16
DEFAULT_ITER_GAMMA = 400
DEFAULT_SMOOTH = 1.6
DEFAULT_FINAL_SMOOTH = 1.0


def build_field(
    coords: np.ndarray,
    grid: int = DEFAULT_GRID,
    gamma: float = DEFAULT_GAMMA,
    iter_gamma: int = DEFAULT_ITER_GAMMA,
    smooth: float = DEFAULT_SMOOTH,
    final_smooth: float = DEFAULT_FINAL_SMOOTH,
) -> np.ndarray:
    """Build the curvature field for the given coordinates.

    Steps:
    1. Rasterise points into a ``grid x grid`` accumulator.
    2. Apply Gaussian smoothing.
    3. Apply Gamma-driven Laplacian diffusion ``iter_gamma`` times.
    4. Apply a light final Gaussian smooth to stabilise gradients.
    """

    F = np.zeros((grid, grid), dtype=np.float64)
    for x, y in coords:
        F[int(min(max(y, 0), grid - 1)), int(min(max(x, 0), grid - 1))] += 1.0

    F = gaussian_filter(F, smooth)

    for _ in range(iter_gamma):
        lap = (
            np.roll(F, 1, 0)
            + np.roll(F, -1, 0)
            + np.roll(F, 1, 1)
            + np.roll(F, -1, 1)
            - 4 * F
        )
        F += gamma * lap
        F[F < 0] = 0

    return gaussian_filter(F, final_smooth)


def apply_gaussian(field: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian smoothing to an existing field."""

    return gaussian_filter(field, sigma)


def apply_laplacian(field: np.ndarray, gamma: float, steps: int) -> np.ndarray:
    """Perform in-place Laplacian diffusion for ``steps`` iterations."""

    F = field.copy()
    for _ in range(steps):
        lap = (
            np.roll(F, 1, 0)
            + np.roll(F, -1, 0)
            + np.roll(F, 1, 1)
            + np.roll(F, -1, 1)
            - 4 * F
        )
        F += gamma * lap
        F[F < 0] = 0
    return F

__all__ = [
    "build_field",
    "apply_gaussian",
    "apply_laplacian",
    "DEFAULT_GRID",
    "DEFAULT_GAMMA",
    "DEFAULT_ITER_GAMMA",
    "DEFAULT_SMOOTH",
    "DEFAULT_FINAL_SMOOTH",
]

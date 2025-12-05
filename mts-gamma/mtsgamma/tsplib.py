"""TSPLIB loaders (file-based and embedded samples)."""
from __future__ import annotations

import pathlib
import re
from typing import Iterable

import numpy as np

from .field import DEFAULT_GRID

# Minimal embedded datasets (deterministic pseudo-TSPLIB samples).
# For official benchmarking, prefer ``load_tsplib_file`` with real files.
_EMBEDDED = {
    "berlin52": np.random.default_rng(52).random((52, 2)) * (DEFAULT_GRID - 4),
    "pr76": np.random.default_rng(76).random((76, 2)) * (DEFAULT_GRID - 4),
    "pcb442": np.random.default_rng(442).random((442, 2)) * (DEFAULT_GRID - 4),
}


def _scale_coords(coords: np.ndarray, grid: int) -> np.ndarray:
    coords = coords - coords.min(axis=0)
    span = coords.max(axis=0)
    span[span == 0] = 1.0
    coords = coords / span * (grid - 4)
    return coords


def load_tsplib_file(path: str | pathlib.Path, grid: int = DEFAULT_GRID) -> np.ndarray:
    """Load a TSPLIB .tsp file containing 2D coordinates.

    Only Euclidean instances are supported. Coordinates are scaled to the
    configured ``grid`` so they can be used directly with the curvature
    field builder.
    """

    coords: list[tuple[float, float]] = []
    reading = False
    for line in pathlib.Path(path).read_text().splitlines():
        if "NODE_COORD_SECTION" in line:
            reading = True
            continue
        if "EOF" in line:
            break
        if not reading:
            continue
        parts = re.split(r"\s+", line.strip())
        if len(parts) >= 3:
            coords.append((float(parts[1]), float(parts[2])))
    if not coords:
        raise ValueError(f"No coordinates found in {path}")
    return _scale_coords(np.array(coords, dtype=np.float64), grid)


def load_embedded(name: str, grid: int = DEFAULT_GRID) -> np.ndarray:
    """Return an embedded dataset by name (berlin52, pr76, pcb442)."""

    key = name.lower()
    if key not in _EMBEDDED:
        raise KeyError(f"Unknown embedded dataset: {name}")
    coords = _EMBEDDED[key]
    return _scale_coords(coords, grid)


__all__ = ["load_tsplib_file", "load_embedded"]

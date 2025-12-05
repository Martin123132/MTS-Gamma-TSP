"""CUDA-ready scaffolding for future GPU acceleration.

This module only contains placeholders describing how curvature field
construction, gradient flow, and refinement could be mapped to CUDA.
"""
from __future__ import annotations

import numpy as np


def curvature_field_cuda_stub(coords: np.ndarray) -> None:
    """Placeholder for CUDA curvature field builder."""

    raise NotImplementedError("CUDA kernel pending (see Glyn's roadmap)")


def gradient_flow_cuda_stub(field: np.ndarray, coords: np.ndarray) -> None:
    """Placeholder for CUDA gradient flow."""

    raise NotImplementedError("CUDA kernel pending (see Glyn's roadmap)")


def refinement_cuda_stub(route: np.ndarray, coords: np.ndarray) -> None:
    """Placeholder for mapping C4 refinements to block/grid structure."""

    raise NotImplementedError("CUDA kernel pending (see Glyn's roadmap)")

__all__ = [
    "curvature_field_cuda_stub",
    "gradient_flow_cuda_stub",
    "refinement_cuda_stub",
]

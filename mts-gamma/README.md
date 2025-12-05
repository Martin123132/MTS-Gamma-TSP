# MTS–Gamma TSP Solver

Reconstruction of the curvature-flow based MTS–Γ heuristic using the fastest
C4 refinement pipeline. The solver builds a curvature field over a raster
grid, orders cities via gradient flow, then applies Numba-accelerated 2.5-opt
and 3-opt (C4) refinement.

## Features
- Curvature field with Gaussian smoothing and Gamma Laplacian diffusion
- Gradient-flow ordering with boundary clamping
- Fast Numba-accelerated C4 refinement (2.5-opt + 3-opt cycles)
- Apples-to-apples Christofides baseline (NetworkX)
- TSPLIB file loader plus embedded demo datasets
- Parameter sweep and stability testing utilities
- CLI and ready-to-run notebooks
- CUDA scaffolding for future GPU work

## Installation
```
pip install -r requirements.txt
pip install -e .
```

## Quickstart
Run on a random 500-city instance:
```
python -m cli.mtsgamma_cli solve --n 500
```

Run against a TSPLIB file:
```
python -m cli.mtsgamma_cli solve --file berlin52.tsp
```

Run parameter sweep:
```
python -m cli.mtsgamma_cli sweep --n 200
```

Stability tests:
```
python -m cli.mtsgamma_cli stability --seeds 20
```

## API usage
```python
import numpy as np
from mtsgamma import mts_gamma_C4, christofides_route, tour_length

coords = np.random.rand(500, 2) * 508
c_route = christofides_route(coords)
c_len = tour_length(c_route, coords)
route, m_len = mts_gamma_C4(coords)
print("Improvement %:", (c_len - m_len) / c_len * 100)
```

## Notebooks
- `notebooks/C4_demo.ipynb` – random N=500 and N=1000 runs
- `notebooks/TSPLIB_tests.ipynb` – berlin52, pr76, pcb442 demos
- `notebooks/parameter_sweep.ipynb` – grid sweep logging to CSV
- `notebooks/stability_tests.ipynb` – multi-seed stability report

## GPU roadmap
`mtsgamma/gpu.py` contains CUDA placeholders for curvature field, gradient
flow, and refinement kernels to be implemented later.

## License
MIT

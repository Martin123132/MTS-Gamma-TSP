"""Command-line interface for the MTS–Gamma solver."""
from __future__ import annotations

import argparse
import sys

import numpy as np

from mtsgamma import (
    SolverParams,
    christofides_route,
    load_embedded,
    load_tsplib_file,
    mts_gamma_C4,
    parameter_sweep,
    run_stability_tests,
    tour_length,
)
from mtsgamma.field import DEFAULT_GRID


def _load_coords(args: argparse.Namespace) -> np.ndarray:
    if args.file:
        return load_tsplib_file(args.file, grid=DEFAULT_GRID)
    if args.dataset:
        return load_embedded(args.dataset, grid=DEFAULT_GRID)
    return np.random.rand(args.n, 2) * (DEFAULT_GRID - 4)


def cmd_solve(args: argparse.Namespace) -> None:
    coords = _load_coords(args)
    params = SolverParams()
    route = christofides_route(coords)
    c_len = tour_length(route, coords)
    mts_route, mts_len = mts_gamma_C4(coords, params=params)
    print("Christofides length:", c_len)
    print("MTS–Gamma C4 length:", mts_len)
    print("Improvement %:", (c_len - mts_len) / c_len * 100)


def cmd_sweep(args: argparse.Namespace) -> None:
    coords = _load_coords(args)
    gammas = [float(x) for x in args.gamma.split(",")]
    smooth = [float(x) for x in args.smooth.split(",")]
    flow_steps = [int(x) for x in args.flow_steps.split(",")]
    step_sizes = [float(x) for x in args.step_sizes.split(",")]
    parameter_sweep(coords, gammas, smooth, flow_steps, step_sizes, args.output)
    print("Saved sweep results to", args.output)


def cmd_stability(args: argparse.Namespace) -> None:
    run_stability_tests(seeds=args.seeds, csv_path=args.output)
    print("Saved stability results to", args.output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mtsgamma")
    sub = parser.add_subparsers(dest="command", required=True)

    p_solve = sub.add_parser("solve", help="Run solver on random or TSPLIB data")
    p_solve.add_argument("--n", type=int, default=500, help="Number of random cities")
    p_solve.add_argument("--file", type=str, help="TSPLIB .tsp file")
    p_solve.add_argument("--dataset", type=str, help="Embedded dataset name")
    p_solve.set_defaults(func=cmd_solve)

    p_sweep = sub.add_parser("sweep", help="Parameter sweep")
    p_sweep.add_argument("--gamma", default="0.12,0.16,0.18")
    p_sweep.add_argument("--smooth", default="1.2,1.6,2.0")
    p_sweep.add_argument("--flow-steps", dest="flow_steps", default="300,450,600")
    p_sweep.add_argument("--step-sizes", dest="step_sizes", default="0.8,1.2,1.6")
    p_sweep.add_argument("--output", default="mts_gamma_sweep.csv")
    p_sweep.add_argument("--n", type=int, default=200)
    p_sweep.set_defaults(func=cmd_sweep)

    p_stab = sub.add_parser("stability", help="Multi-seed stability testing")
    p_stab.add_argument("--seeds", type=int, default=20)
    p_stab.add_argument("--output", default="mts_gamma_stability.csv")
    p_stab.set_defaults(func=cmd_stability)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

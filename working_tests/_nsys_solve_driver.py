#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-only
"""Tiny driver for nsys profiling of warm mixed-precision campaign solves.

Usage::

    python working_tests/_nsys_solve_driver.py fp64  2000x1000
    python working_tests/_nsys_solve_driver.py fp32  2000x1000
    python working_tests/_nsys_solve_driver.py mixed 2000x1000

The driver builds one case, performs one cold solve, and then performs a
configurable number of warm solves. It intentionally records no Python-level
timing because the output of interest is the nsys kernel trace.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def configure_import_path(*, repo_root: Path) -> None:
    """Make the repository importable when this file is run directly."""
    repo_root_text = str(repo_root)
    if repo_root_text not in sys.path:
        sys.path.insert(0, repo_root_text)


def configure_precision(*, mode: str) -> None:
    """Pin the model precision before importing DarcyWarp modules."""
    if mode not in {"fp64", "fp32", "mixed"}:
        raise ValueError("mode must be one of: fp64, fp32, mixed")
    os.environ["DARCY_FLOAT"] = "float64" if mode == "fp64" else "float32"


def run_profile_case(
    *,
    mode: str,
    label: str,
    repetitions: int,
    grid_cases: dict[str, tuple[int, int]],
    dx: float,
    thickness: float,
    device: str,
    t_seed: int,
) -> None:
    """Build the requested case and execute cold plus warm solves."""
    import numpy as np
    import warp as wp

    from DARCY_WARP_PACKAGE.model_builder import (
        _build_dem,
        _build_domain,
        build_truth_inputs,
        make_ugly_T_field,
    )
    from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver

    if label not in grid_cases:
        raise ValueError(f"unknown grid label {label!r}; choose from {sorted(grid_cases)}")
    if int(repetitions) < 1:
        raise ValueError("repetitions must be at least 1")

    nx, ny = grid_cases[label]
    wp.init()
    domain = _build_domain(nx=nx, ny=ny)
    dem = _build_dem(domain)
    t_field = make_ugly_T_field(nx=nx, ny=ny, domain=domain, seed=int(t_seed))
    recharge_field = np.full_like(domain, 1.0e-4, dtype=np.float64)

    with WarpDarcySolver(
        nx=nx,
        ny=ny,
        dx=dx,
        device=device,
        use_ghb=True,
        solver_type="pcg",
        aq_thickness=thickness,
    ) as solver:
        solver.build_from_truth_inputs(
            T_truth=t_field,
            R_truth=recharge_field,
            width=dx,
        )
        solver.build_hierarchy(
            max_levels=6,
            min_coarse_n=4,
            min_coarse_cells=500,
        )

        if mode == "mixed":
            from DARCY_WARP_PACKAGE.solvers.mixed_precision import (
                MixedPrecisionDefectCorrectionSession,
            )

            (_, _, _, _, bc_values_f64, _, gh_head_f64, _) = build_truth_inputs(
                nx=nx,
                ny=ny,
                dx=dx,
                T_truth=t_field,
                R_truth=recharge_field,
                use_ghb=True,
                width=dx,
            )
            session = MixedPrecisionDefectCorrectionSession(
                solver,
                bc_values_f64=bc_values_f64,
                gh_head_f64=gh_head_f64,
                R_f64=recharge_field,
                max_levels=6,
            )

            def solve_once():
                return session.solve(
                    dem,
                    inner_kcycles=5,
                    max_outer=40,
                    rel_tol=5.0e-7,
                    abs_tol_min=5.0e-7,
                )

        else:
            max_cycles = 20 if mode == "fp32" else 200

            def solve_once():
                return solver.solve_multigrid_kcycle(
                    max_cycles=max_cycles,
                    nu_pre=2,
                    nu_post=2,
                    nu_coarse=2,
                    omega=0.7,
                    rel_tol=5.0e-7,
                    abs_tol_min=5.0e-7,
                    initial_head=dem,
                    return_info=True,
                    max_levels=6,
                    check_every_no=5,
                )

        solve_once()
        for _ in range(int(repetitions)):
            solve_once()
        wp.synchronize_device(device)
    print(f"driver done {mode} {label}")


def main(*, argv: list[str]) -> None:
    """Parse the intentionally minimal profiler command line."""
    if len(argv) not in {3, 4}:
        raise SystemExit("usage: _nsys_solve_driver.py MODE GRID [REPETITIONS]")

    mode = str(argv[1]).strip().lower()
    label = str(argv[2]).strip()
    repetitions = int(argv[3]) if len(argv) == 4 else 5
    repo_root = Path(__file__).resolve().parents[1]
    grid_cases = {
        "100x100": (100, 100),
        "1000x1001": (1000, 1001),
        "2000x1000": (2000, 1000),
    }
    dx = 100.0
    thickness = 300.0
    device = "cuda:0"
    t_seed = 123

    configure_import_path(repo_root=repo_root)
    configure_precision(mode=mode)
    run_profile_case(
        mode=mode,
        label=label,
        repetitions=repetitions,
        grid_cases=grid_cases,
        dx=dx,
        thickness=thickness,
        device=device,
        t_seed=t_seed,
    )


if __name__ == "__main__":
    main(argv=sys.argv)

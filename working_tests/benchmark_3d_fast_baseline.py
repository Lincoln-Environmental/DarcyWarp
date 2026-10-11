# SPDX-License-Identifier: AGPL-3.0-only
"""Performance baseline for the fast 3D execution path.

Measures, for representative confined 3D cases on ``implementation="fast"``:

- operator/session construction time (host face build + session create),
- first-solve wall time (hierarchy build + CUDA graph capture),
- second-solve wall time (workspace + graph reuse),
- per-solve cycle counts and terminal residual,
- GPU memory in use (via nvidia-smi, if available).

Usage:
    DARCY_FLOAT=float64 python working_tests/benchmark_3d_fast_baseline.py [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from typing import Any

import numpy as np


def _gpu_memory_mib() -> float | None:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def _build_case(nz: int, ny: int, nx: int, dx: float, dz: float, seed: int = 7):
    rng = np.random.default_rng(seed)
    shape = (nz, ny, nx)
    kx = np.full(shape, 10.0)
    # Mild heterogeneity so the operator is not trivially uniform.
    kx *= np.exp(rng.normal(0.0, 0.25, size=shape))
    ky = kx.copy()
    kz = 0.1 * kx
    active = np.ones(shape, dtype=np.int32)
    bc_mask = np.zeros(shape, dtype=np.int32)
    bc_values = np.zeros(shape)
    bc_mask[:, :, 0] = 1
    bc_values[:, :, 0] = 20.0
    bc_mask[:, :, -1] = 1
    bc_values[:, :, -1] = 10.0
    rhs = np.zeros(shape)
    rhs[0] = 1.0e-4 * dx * dx  # volumetric recharge on the top layer
    initial_head = np.full(shape, 15.0)
    return kx, ky, kz, active, bc_mask, bc_values, rhs, initial_head


def _run_case(label: str, nz: int, ny: int, nx: int, dx: float, dz: float,
              transient: bool, device: str) -> dict[str, Any]:
    from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D

    kx, ky, kz, active, bc_mask, bc_values, rhs, initial_head = _build_case(nz, ny, nx, dx, dz)

    t0 = time.perf_counter()
    solver = WarpDarcySolver3D(nx=nx, ny=ny, nz=nz, dx=dx, dy=dx, dz=dz,
                               device=device, solver="kcycle", implementation="fast")
    solver.build_from_K_fields(kx, ky, kz, active, bc_mask, bc_values, rhs, initial_head)
    t_build = time.perf_counter() - t0

    solve_kwargs: dict[str, Any] = {}
    if transient:
        solve_kwargs.update(transient=True, storage_coeff=1.0e-4, dt=1.0,
                            head_prev=initial_head)

    mem_before = _gpu_memory_mib()
    t0 = time.perf_counter()
    head1, info1 = solver.solve(**solve_kwargs)
    t_solve1 = time.perf_counter() - t0
    mem_after_first = _gpu_memory_mib()

    t0 = time.perf_counter()
    head2, info2 = solver.solve(**solve_kwargs)
    t_solve2 = time.perf_counter() - t0
    mem_after_second = _gpu_memory_mib()

    result = {
        "label": label,
        "shape": [nz, ny, nx],
        "n_cells": nz * ny * nx,
        "transient": transient,
        "build_s": t_build,
        "solve1_s": t_solve1,
        "solve2_s": t_solve2,
        "cycles1": info1.get("n_cycles_used"),
        "cycles2": info2.get("n_cycles_used"),
        "converged1": info1.get("converged"),
        "converged2": info2.get("converged"),
        "r_rms_end1": info1.get("r_rms_end"),
        "r_rms_end2": info2.get("r_rms_end"),
        "n_levels": info1.get("n_levels"),
        "workspace_reused2": info2.get("workspace_reused"),
        "cuda_graph_replayed2": info2.get("cuda_graph_replayed"),
        "gpu_mem_mib_before": mem_before,
        "gpu_mem_mib_after_first": mem_after_first,
        "gpu_mem_mib_after_second": mem_after_second,
        "head_max_diff_repeat": float(np.max(np.abs(head1 - head2))),
    }
    solver.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--small-only", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("DARCY_FLOAT", "float64")

    cases = [
        ("steady_4x64x64", 4, 64, 64, 25.0, 10.0, False),
        ("transient_4x64x64", 4, 64, 64, 25.0, 10.0, True),
    ]
    if not args.small_only:
        cases += [
            ("steady_6x256x256", 6, 256, 256, 25.0, 10.0, False),
            ("transient_6x256x256", 6, 256, 256, 25.0, 10.0, True),
        ]

    results = []
    for label, nz, ny, nx, dx, dz, transient in cases:
        row = _run_case(label, nz, ny, nx, dx, dz, transient, args.device)
        results.append(row)
        print(json.dumps(row, indent=2))

    payload = {"cases": results, "device": args.device}
    out = args.out
    if out:
        with open(out, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

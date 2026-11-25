"""Measure native 3D DRN/RIV construction, solve and update costs on CUDA."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import warp as wp

from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D


def build_case_fields(*, shape: tuple[int, int, int], boundary_case: str) -> dict:
    """Create reproducible homogeneous confined fields and package layouts.

    :param shape: Cell counts in (nz, ny, nx) order.
    :param boundary_case: One of ``none``, ``sparse_drn``, ``distributed_riv`` or ``overlap``.
    :return: Keyword fields accepted by :meth:`WarpDarcySolver3D.build_from_K_fields`.
    :raises ValueError: If ``boundary_case`` is unsupported.
    """
    nz, ny, nx = shape
    active = np.ones(shape, dtype=np.int32)
    fixed = np.zeros(shape, dtype=np.int32)
    fixed[:, :, 0] = 1
    fixed[:, :, -1] = 1
    bc_values = np.full(shape, 50.0)
    bc_values[fixed == 0] = 0.0
    source = np.zeros(shape)
    source[fixed == 0] = 0.01
    fields = dict(
        kx_field=np.full(shape, 100.0),
        ky_field=np.full(shape, 80.0),
        kz_field=np.full(shape, 20.0),
        active=active,
        bc_mask=fixed,
        bc_values=bc_values,
        rhs=source,
        initial_head=np.full(shape, 50.0),
    )
    package_mask = np.zeros(shape, dtype=np.int32)
    package_mask[0, :, 1:-1] = 1
    # These paired-overhead cases keep package fields active while ensuring
    # exact zero exchange at all heads reached in the solve. Thus their
    # accepted solution is physically identical to the no-package baseline.
    neutral_level = 1.0e6
    if boundary_case == "sparse_drn":
        package_mask[0, ::10, 1:-1:10] = 1
        fields.update(drn_conductance=100.0, drn_elevation=neutral_level, drn_mask=package_mask)
    elif boundary_case == "distributed_riv":
        fields.update(riv_conductance=10.0, riv_stage=neutral_level, riv_bottom=neutral_level,
            riv_mask=package_mask)
    elif boundary_case == "overlap":
        fields.update(drn_conductance=50.0, drn_elevation=neutral_level, drn_mask=package_mask,
            riv_conductance=10.0, riv_stage=neutral_level, riv_bottom=neutral_level,
            riv_mask=package_mask)
    elif boundary_case != "none":
        raise ValueError(f"Unsupported boundary case: {boundary_case}")
    return fields


def benchmark_one(*, shape: tuple[int, int, int], boundary_case: str,
    repeats: int, device: str) -> dict:
    """Benchmark one fixed grid and boundary pattern on the selected device.

    :param shape: Cell counts in (nz, ny, nx) order.
    :param boundary_case: Boundary configuration passed to :func:`build_case_fields`.
    :param repeats: Number of timed solve and update/re-solve repetitions.
    :param device: Warp device identifier, expected to be CUDA for this benchmark.
    :return: JSON-serialisable timings, convergence results and solver counts.
    """
    field_preparation_start = perf_counter()
    fields = build_case_fields(shape=shape, boundary_case=boundary_case)
    field_preparation_seconds = perf_counter() - field_preparation_start
    build_start = perf_counter()
    model = WarpDarcySolver3D(nx=shape[2], ny=shape[1], nz=shape[0], dx=100.0,
        dy=100.0, dz=20.0, device=device, implementation="fast",
        formulation="mf6_6_6_2_npf")
    model.build_from_K_fields(**fields)
    wp.synchronize_device(device=device)
    build_seconds = perf_counter() - build_start
    warmup_start = perf_counter()
    accepted_head, last_info = model.solve(npf_preconditioner="kcycle", max_outer=40,
        max_cycles=400, krylov_restart=30)
    wp.synchronize_device(device=device)
    warmup_seconds = perf_counter() - warmup_start
    solve_rows = []
    solver_rows = []
    public_rows = []
    reference_setup_rows = []
    update_rows = []
    workspace_setup_rows = []
    reference_head = accepted_head.copy()
    package_fields = any(name in fields for name in ("drn_conductance", "riv_conductance"))
    for repeat in range(repeats):
        if repeat > 0 and "riv_conductance" in fields:
            update_start = perf_counter()
            neutral_river_level = 1.0e6 + 0.005 * repeat
            model.update_riv(riv_conductance=10.0, riv_stage=neutral_river_level,
                riv_bottom=neutral_river_level, riv_mask=fields["riv_mask"])
            wp.synchronize_device(device=device)
            update_seconds = perf_counter() - update_start
        elif repeat > 0 and "drn_conductance" in fields:
            update_start = perf_counter()
            model.update_drn(drn_conductance=100.0 if boundary_case == "sparse_drn" else 50.0,
                drn_elevation=1.0e6 + 0.005 * repeat, drn_mask=fields["drn_mask"])
            wp.synchronize_device(device=device)
            update_seconds = perf_counter() - update_start
        else:
            update_seconds = 0.0
        solve_start = perf_counter()
        accepted_head, last_info = model.solve(npf_preconditioner="kcycle", max_outer=40,
            max_cycles=400, krylov_restart=30)
        wp.synchronize_device(device=device)
        solve_seconds = perf_counter() - solve_start
        solve_rows.append(solve_seconds)
        solver_rows.append(float(last_info.get("seconds", 0.0)))
        public_rows.append(float(last_info.get("public_total_seconds", solve_seconds)))
        reference_setup_rows.append(float(last_info.get("reference_setup_seconds", 0.0)))
        update_rows.append(update_seconds)
        workspace_setup_rows.append(float(last_info.get("setup_seconds", 0.0)))
        if not last_info["converged"]:
            break
    result = dict(
        shape=list(shape),
        active_cells=int(np.prod(shape)),
        boundary_case=boundary_case,
        package_fields=package_fields,
        zero_exchange_solution_equivalent_to_no_package=True,
        field_preparation_seconds=field_preparation_seconds,
        build_seconds=build_seconds,
        warmup_seconds=warmup_seconds,
        solve_seconds=solve_rows,
        solver_seconds=solver_rows,
        public_total_seconds=public_rows,
        reference_setup_seconds=reference_setup_rows,
        update_api_seconds=update_rows,
        workspace_setup_seconds=workspace_setup_rows,
        converged=bool(last_info and last_info["converged"]),
        outer_iterations=None if last_info is None else last_info["outer_iterations"],
        krylov_iterations=None if last_info is None else last_info["krylov_iterations"],
        multigrid_cycles=None if last_info is None else last_info.get("preconditioner_cycles"),
        workspace_reused=None if last_info is None else last_info.get("workspace_reused"),
        device_final_metrics=None if last_info is None else last_info.get("device_final_metrics"),
    )
    model.close()
    return result, accepted_head


if __name__ == "__main__":
    device = "cuda:0"
    grid_shapes = [(3, 200, 200), (5, 450, 450)]
    boundary_cases = ["none", "sparse_drn", "distributed_riv", "overlap"]
    repeats = 3
    output_path = Path("benchmark_results/drn_riv_3d/cuda_runtime.json")

    wp.init()
    if not wp.is_cuda_available():
        raise RuntimeError("CUDA is required for the 3D DRN/RIV performance benchmark")
    rows = []
    for shape in grid_shapes:
        baseline_head = None
        for boundary_case in boundary_cases:
            result, accepted_head = benchmark_one(shape=shape, boundary_case=boundary_case,
                repeats=repeats, device=device)
            if boundary_case == "none":
                baseline_head = accepted_head
                result["maximum_head_difference_from_no_package_m"] = 0.0
            else:
                result["maximum_head_difference_from_no_package_m"] = float(
                    np.max(np.abs(accepted_head - baseline_head)))
            rows.append(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2) + "\n")
    print(output_path)

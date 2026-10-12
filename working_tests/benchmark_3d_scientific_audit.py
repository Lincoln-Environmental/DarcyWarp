"""Repeatable fast-3D scientific-audit benchmark (FP64, CUDA)."""
from pathlib import Path
import argparse
import json
import time

import numpy as np
import warp as wp

from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D


def run_case(*, shape, dx, dz, conductivity, recharge, storage, sy, dt,
             boundary_heads, initial_value, unconfined, transient, layered,
             contrast, device, controls, repeats):
    nz, ny, nx = shape
    k = np.full(shape, conductivity)
    if layered:
        k *= np.geomspace(1.0, contrast, nz)[:, None, None]
    active = np.ones(shape, dtype=np.int32)
    bc = np.zeros(shape, dtype=np.int32)
    bc[:, :, 0] = bc[:, :, -1] = 1
    values = np.zeros(shape)
    values[:, :, 0], values[:, :, -1] = boundary_heads
    rhs = np.zeros(shape)
    rhs[0] = recharge * dx * dx
    h0 = np.full(shape, initial_value)
    bottoms = np.broadcast_to((nz - 1 - np.arange(nz))[:, None, None] * dz, shape).copy()
    options = dict(controls)
    if not unconfined:
        for name in ("unconfined_head_tol", "final_head_residual_tol", "unconfined_max_picard_iter"):
            options.pop(name)
    if unconfined:
        options.update(unconfined=True, zbot_field=bottoms, sy=sy, ss=storage,
                       chebyshev_enabled=False)
    if transient:
        options.update(transient=True, storage_coeff=storage, dt=dt, head_prev=h0)
    with WarpDarcySolver3D(nx=nx, ny=ny, nz=nz, dx=dx, dy=dx, dz=dz,
                          device=device, implementation="fast") as solver:
        solver.build_from_K_fields(kx_field=k, ky_field=k, kz_field=k,
                                   active=active, bc_mask=bc, bc_values=values,
                                   rhs=rhs, initial_head=h0)
        rows = []
        for repeat in range(repeats):
            key = "workspace_unconfined" if unconfined else "workspace"
            before_workspace = solver._session._fast_workspace_holder.get(key)
            graph_present_before = before_workspace is not None and before_workspace.get("cycle_graph") not in (None, False)
            wp.synchronize_device(device=device)
            start = time.perf_counter()
            _, info = solver.solve(**options)
            wp.synchronize_device(device=device)
            rows.append(dict(seconds=time.perf_counter() - start, repeat=repeat,
                             cycles=info.get("n_cycles_used"),
                             total_inner_cycles=sum(row["inner_iterations"] for row in info.get("outer_history", [])) if unconfined else info.get("n_cycles_used"),
                             outer=info.get("outer_iterations"),
                             converged=info["converged"],
                             graph_reused=info.get("cuda_graph_reused"),
                             workspace_reused=info.get("workspace_reused"),
                             graph_present_before_solve=graph_present_before,
                             graph_capture_failed=info.get("cuda_graph_capture_failed"),
                             physical_head_residual=info.get("final_head_residual_rms", info.get("final_physical_head_residual_rms")),
                             residual=info.get("final_physical_residual_max",
                                               info["budget"]["cell_residual_max"]),
                             relative_budget=info["budget"]["relative_imbalance"],
                             mempool_used=wp.get_mempool_used_mem_current(device=device),
                             mempool_high=wp.get_mempool_used_mem_high(device=device)))
    return rows


def run_benchmarks(*, cases, settings, output):
    wp.init()
    results = {}
    for name, options in cases.items():
        rows = run_case(**settings, **options)
        results[name] = rows
        print(name, rows, flush=True)
    output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--nz", type=int, default=4)
    args = parser.parse_args()
    layer_thickness = 10.0
    settings = dict(shape=(args.nz, args.ny, args.nx), dx=25.0, dz=layer_thickness, conductivity=10.0,
                    recharge=1e-4, storage=1e-4, sy=0.15, dt=7.0,
                    boundary_heads=(args.nz * layer_thickness - 5.0, args.nz * layer_thickness - 8.0),
                    initial_value=args.nz * layer_thickness - 6.0,
                    contrast=1e-6, device="cuda:0", repeats=4,
                    controls=dict(max_cycles=300, rel_tol=1e-9, abs_tol_min=1e-12,
                                  check_every_no=5, unconfined_head_tol=1e-6,
                                  final_head_residual_tol=1e-6,
                                  unconfined_max_picard_iter=200))
    cases = dict(confined_steady=dict(unconfined=False, transient=False, layered=False),
                 confined_transient=dict(unconfined=False, transient=True, layered=False),
                 unconfined_steady=dict(unconfined=True, transient=False, layered=False),
                 unconfined_transient=dict(unconfined=True, transient=True, layered=False),
                 layered_unconfined_transient=dict(unconfined=True, transient=True, layered=True))
    run_benchmarks(cases=cases, settings=settings, output=args.out)

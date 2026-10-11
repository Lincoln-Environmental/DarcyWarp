# SPDX-License-Identifier: AGPL-3.0-only
"""FP64 face-array K-cycle for steady and transient 3D flow.

This backend keeps six face conductances explicit, uses block-partial
reductions, keeps K-cycle alpha calculations on device, and graph-captures a
fixed cycle on CUDA when Warp capture is available. Backward-Euler storage is
assembled with the same helper as the classic backend and carried through the
semi-coarsened hierarchy. This module is selected through
``implementation="fast"``.

Structure:

- ``solve_fast_confined_kcycle_3d`` — steady / backward-Euler confined solves
  with a session-cached workspace (graph replay across identical calls).
- ``solve_fast_unconfined_kcycle_3d`` — unconfined Picard driver (shared
  ``_picard_unconfined_7point_3d``) whose inner linear solves run on a
  persistent fast workspace refreshed IN PLACE each outer iteration (same
  device buffers, so the captured CUDA graph stays valid).  The corrected
  unconfined conductance and the volume-conservative secant storage live in
  ``solvers_3d`` and are shared with the classic backends.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from DARCY_WARP_PACKAGE.config import NP_FLOAT, WP_FLOAT
from DARCY_WARP_PACKAGE.kernels_3d import (
    add_correction_3d_kernel,
    copy_field_3d_kernel,
    dh_change_reduce_3d_kernel,
    prolong_bilinear_axes_3d_kernel,
    restrict_blockavg_axes_3d_kernel,
    zero_scalar_kernel,
)
from DARCY_WARP_PACKAGE.solvers.face_kernels_3d_f64 import (
    BLOCK_SIZE_3D,
    apply_a_dot_partials_3d_f64_kernel,
    axpy_active_device_scalar_3d_kernel,
    combine_partials_3d_f64_kernel,
    compute_safe_alpha_3d_kernel,
    dot_partials_3d_f64_kernel,
    face_jacobi_3d_f64_kernel,
    head_residual_partials_3d_f64_kernel,
    residual_partials_3d_f64_kernel,
)
from DARCY_WARP_PACKAGE.solvers_3d import (
    _choose_horizontal_coarsening,
    _coarsen_bc_mask_axes,
    _coarsen_faces_consistent_axes,
    _coarsen_max_edge_axes,
    _masked_block_sum_axes,
    _picard_unconfined_7point_3d,
    _prepare_7point_transient_terms,
    _resolve_isolated_free_cells,
    build_7point_face_conductance_unconfined_from_sat,
)
from DARCY_WARP_PACKAGE.warped_darcy import _chebyshev_relaxation_sequence


def _sanitise_faces_and_diagonal(
    *,
    faces: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    active: np.ndarray,
    storage: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """Apply the classic active-neighbour contract and form the FP64 diagonal."""
    tx_p, tx_m, ty_p, ty_m, tz_p, tz_m = (np.maximum(np.asarray(value, dtype=np.float64), 0.0).copy() for value in faces)
    active_b = np.asarray(active, dtype=np.int32) != 0
    tx_p[:, :, -1] = 0.0
    tx_m[:, :, 0] = 0.0
    ty_p[:, -1, :] = 0.0
    ty_m[:, 0, :] = 0.0
    tz_p[-1, :, :] = 0.0
    tz_m[0, :, :] = 0.0
    if tx_p.shape[2] > 1:
        tx_p[:, :, :-1][~active_b[:, :, 1:]] = 0.0
        tx_m[:, :, 1:][~active_b[:, :, :-1]] = 0.0
    if tx_p.shape[1] > 1:
        ty_p[:, :-1, :][~active_b[:, 1:, :]] = 0.0
        ty_m[:, 1:, :][~active_b[:, :-1, :]] = 0.0
    if tx_p.shape[0] > 1:
        tz_p[:-1, :, :][~active_b[1:, :, :]] = 0.0
        tz_m[1:, :, :][~active_b[:-1, :, :]] = 0.0
    cleaned = (tx_p, tx_m, ty_p, ty_m, tz_p, tz_m)
    diagonal = np.maximum(np.asarray(storage, dtype=np.float64), 0.0)
    diagonal = diagonal + sum(cleaned)
    diagonal[(active_b) & (diagonal < 1.0e-12)] = 1.0
    diagonal[~active_b] = 1.0
    return cleaned, diagonal


def _build_levels(
    *,
    faces: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    active: np.ndarray,
    bc_mask: np.ndarray,
    bc_values: np.ndarray,
    rhs: np.ndarray,
    storage_diag: np.ndarray,
    max_levels: int,
    min_coarse_n: int,
) -> list[dict[str, Any]]:
    levels = [{
        "faces": tuple(np.asarray(value, dtype=np.float64) for value in faces),
        "active": np.asarray(active, dtype=np.int32),
        "bc_mask": np.asarray(bc_mask, dtype=np.int32),
        "bc_values": np.asarray(bc_values, dtype=np.float64),
        "rhs": np.asarray(rhs, dtype=np.float64),
        "storage": np.asarray(storage_diag, dtype=np.float64),
        "coarsen_y": 1,
        "coarsen_x": 1,
    }]
    for _ in range(1, int(max_levels)):
        previous = levels[-1]
        _, ny, nx = previous["active"].shape
        coarsen_y, coarsen_x = _choose_horizontal_coarsening(
            ny=int(ny), nx=int(nx), min_coarse_n=int(min_coarse_n)
        )
        if coarsen_y == 1 and coarsen_x == 1:
            break
        active_c = _coarsen_max_edge_axes(previous["active"], coarsen_y=coarsen_y, coarsen_x=coarsen_x)
        bc_mask_c = _coarsen_bc_mask_axes(
            previous["bc_mask"], previous["active"], coarsen_y=coarsen_y, coarsen_x=coarsen_x
        )
        free_f = (previous["active"] != 0) & (previous["bc_mask"] == 0)
        face_c, springs_c = _coarsen_faces_consistent_axes(
            previous["faces"], previous["active"], previous["bc_mask"],
            coarsen_y=coarsen_y, coarsen_x=coarsen_x,
        )
        # Galerkin Dirichlet springs ride on the coarse storage diagonal slot
        # (diagonal-only, non-negative); see solvers_3d._coarsen_faces_consistent_axes.
        storage_c = _masked_block_sum_axes(
            previous["storage"], free_f.astype(np.float64), coarsen_y=coarsen_y, coarsen_x=coarsen_x
        ) / float(coarsen_y * coarsen_x) + springs_c
        storage_c[(active_c == 0) | (bc_mask_c != 0)] = 0.0
        levels.append({
            "faces": face_c,
            "active": active_c,
            "bc_mask": bc_mask_c,
            "bc_values": np.zeros_like(storage_c),
            "rhs": np.zeros_like(storage_c),
            "storage": storage_c,
            "coarsen_y": 1,
            "coarsen_x": 1,
        })
        previous["coarsen_y"] = int(coarsen_y)
        previous["coarsen_x"] = int(coarsen_x)
    return levels


def _upload_levels(
    host_levels: list[dict[str, Any]],
    device: str,
) -> list[dict[str, Any]]:
    """Upload host levels to fresh device arrays (first build)."""
    levels: list[dict[str, Any]] = []
    for host in host_levels:
        cleaned_faces, diagonal = _sanitise_faces_and_diagonal(
            faces=host["faces"], active=host["active"], storage=host["storage"]
        )
        nz, ny, nx = host["active"].shape
        n_cells = int(nz * ny * nx)
        n_partials = (n_cells + BLOCK_SIZE_3D - 1) // BLOCK_SIZE_3D
        level = {
            "nx": int(nx), "ny": int(ny), "nz": int(nz), "dim": (int(nz), int(ny), int(nx)),
            "coarsen_y": int(host["coarsen_y"]), "coarsen_x": int(host["coarsen_x"]),
            "faces": tuple(wp.array(value, dtype=wp.float64, device=device) for value in cleaned_faces),
            "storage": wp.array(np.maximum(host["storage"], 0.0), dtype=wp.float64, device=device),
            "diag": wp.array(diagonal, dtype=wp.float64, device=device),
            "active": wp.array(host["active"], dtype=wp.int32, device=device),
            "bc_mask": wp.array(host["bc_mask"], dtype=wp.int32, device=device),
            "bc_values": wp.array(host["bc_values"], dtype=wp.float64, device=device),
            "rhs": wp.array(host["rhs"], dtype=wp.float64, device=device),
            "x": wp.zeros((nz, ny, nx), dtype=wp.float64, device=device), "work": wp.zeros((nz, ny, nx), dtype=wp.float64, device=device),
            "residual": wp.zeros((nz, ny, nx), dtype=wp.float64, device=device), "correction": wp.zeros((nz, ny, nx), dtype=wp.float64, device=device),
            "z1": wp.zeros((nz, ny, nx), dtype=wp.float64, device=device), "partials_a": wp.zeros(n_partials, dtype=wp.float64, device=device),
            "partials_b": wp.zeros(n_partials, dtype=wp.float64, device=device), "num": wp.zeros(1, dtype=wp.float64, device=device),
            "den": wp.zeros(1, dtype=wp.float64, device=device), "alpha": wp.zeros(1, dtype=wp.float64, device=device), "dh_max": wp.zeros(1, dtype=wp.float64, device=device),
            "n_partials": int(n_partials), "n_cells": n_cells,
        }
        levels.append(level)
    return levels


def _refresh_levels_in_place(
    levels: list[dict[str, Any]],
    *,
    faces: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    active: np.ndarray,
    bc_mask: np.ndarray,
    bc_values: np.ndarray,
    rhs: np.ndarray,
    storage_diag: np.ndarray,
    max_levels: int,
    min_coarse_n: int,
) -> None:
    """Refresh operator data in existing device buffers (no reallocation).

    Buffer identity is preserved, so a CUDA graph captured on the level
    buffers remains valid across refreshes.  Grid shape, active topology and
    boundary topology must be unchanged (the unconfined Picard loop only
    changes conductance/storage/rhs VALUES).
    """
    host_levels = _build_levels(
        faces=faces, active=active, bc_mask=bc_mask, bc_values=bc_values,
        rhs=rhs, storage_diag=storage_diag,
        max_levels=max_levels, min_coarse_n=min_coarse_n,
    )
    if len(host_levels) != len(levels):
        raise RuntimeError("fast 3D refresh changed the level count; rebuild the workspace.")
    for level, host in zip(levels, host_levels):
        nz, ny, nx = host["active"].shape
        if (level["nz"], level["ny"], level["nx"]) != (int(nz), int(ny), int(nx)):
            raise RuntimeError("fast 3D refresh changed a level shape; rebuild the workspace.")
        cleaned_faces, diagonal = _sanitise_faces_and_diagonal(
            faces=host["faces"], active=host["active"], storage=host["storage"]
        )
        for device_arr, value in zip(level["faces"], cleaned_faces):
            device_arr.assign(value)
        level["storage"].assign(np.maximum(host["storage"], 0.0))
        level["diag"].assign(diagonal)
        level["rhs"].assign(host["rhs"])
        level["active"].assign(host["active"])


def _run_kcycles(
    *,
    levels: list[dict[str, Any]],
    workspace_record: dict[str, Any],
    x0: np.ndarray,
    n_free: int,
    device: str,
    smoother: str,
    pre_omegas: tuple[float, ...],
    post_omegas: tuple[float, ...],
    coarse_omegas: tuple[float, ...],
    max_cycles: int,
    rel_tol: float,
    abs_tol_min: float,
    dh_rms_tol: float | None,
    dh_max_tol: float | None,
    check_every_no: int,
    transient: bool,
    dt_used: float,
    isolated: np.ndarray | None = None,
    n_isolated: int = 0,
    restore_values: np.ndarray | None = None,
    return_info: bool = True,
):
    """Run the K-cycle iteration on device levels; capture/replay CUDA graph.

    ``restore_values`` supplies the heads reported for excluded singular
    cells (typically the pre-masking initial head); defaults to ``x0``.
    """
    if restore_values is None:
        restore_values = x0
    shape = x0.shape
    fine = levels[0]
    fine["x"].assign(x0)
    if "x_prev" not in fine:
        fine["x_prev"] = wp.zeros(shape, dtype=wp.float64, device=device)
    fine["x_prev"].assign(x0)

    def residual_into(level: dict[str, Any], x: Any, out: Any) -> None:
        level["partials_a"].fill_(0.0)
        wp.launch(
            residual_partials_3d_f64_kernel, dim=level["n_cells"],
            inputs=[x, level["rhs"], *level["faces"], level["storage"], level["active"], level["bc_mask"],
                    out, level["partials_a"], level["nx"], level["ny"], level["nz"], BLOCK_SIZE_3D], device=device,
        )
        wp.launch(combine_partials_3d_f64_kernel, dim=1, inputs=[level["partials_a"], level["num"], level["n_partials"]], device=device)

    def residual_norm(level: dict[str, Any], x: Any, out: Any) -> float:
        residual_into(level, x, out)
        return float(level["num"].numpy()[0])

    def smooth(level: dict[str, Any], omegas: tuple[float, ...]) -> None:
        x_in, x_out = level["x"], level["work"]
        for step in omegas:
            wp.launch(face_jacobi_3d_f64_kernel, dim=level["dim"],
                      inputs=[level["rhs"], x_in, *level["faces"], level["diag"], level["active"], level["bc_mask"],
                              level["bc_values"], float(step), level["nx"], level["ny"], level["nz"], x_out], device=device)
            x_in, x_out = x_out, x_in
        if x_in is not level["x"]:
            wp.launch(copy_field_3d_kernel, dim=level["dim"], inputs=[x_in, level["x"], level["nx"], level["ny"], level["nz"]], device=device)

    def kcycle(level_id: int) -> None:
        level = levels[level_id]
        smooth(level, pre_omegas)
        residual_into(level, level["x"], level["residual"])
        if level_id == len(levels) - 1:
            smooth(level, coarse_omegas)
            return
        coarse = levels[level_id + 1]
        wp.launch(restrict_blockavg_axes_3d_kernel, dim=coarse["dim"],
                  inputs=[level["residual"], level["active"], level["bc_mask"], coarse["rhs"],
                          level["nx"], level["ny"], level["nz"], coarse["nx"], coarse["ny"], coarse["nz"],
                          level["coarsen_y"], level["coarsen_x"]], device=device)
        coarse["x"].fill_(0.0)
        kcycle(level_id + 1)
        wp.launch(copy_field_3d_kernel, dim=coarse["dim"], inputs=[coarse["x"], coarse["z1"], coarse["nx"], coarse["ny"], coarse["nz"]], device=device)
        residual_into(coarse, coarse["z1"], coarse["residual"])
        wp.launch(copy_field_3d_kernel, dim=coarse["dim"], inputs=[coarse["residual"], coarse["rhs"], coarse["nx"], coarse["ny"], coarse["nz"]], device=device)
        coarse["x"].fill_(0.0)
        kcycle(level_id + 1)
        coarse["partials_a"].fill_(0.0)
        wp.launch(dot_partials_3d_f64_kernel, dim=coarse["n_cells"],
                  inputs=[coarse["rhs"], coarse["x"], coarse["active"], coarse["bc_mask"], coarse["partials_a"],
                          coarse["nx"], coarse["ny"], coarse["nz"], BLOCK_SIZE_3D], device=device)
        wp.launch(combine_partials_3d_f64_kernel, dim=1, inputs=[coarse["partials_a"], coarse["num"], coarse["n_partials"]], device=device)
        coarse["partials_b"].fill_(0.0)
        wp.launch(apply_a_dot_partials_3d_f64_kernel, dim=coarse["n_cells"],
                  inputs=[coarse["x"], *coarse["faces"], coarse["diag"], coarse["active"], coarse["bc_mask"], coarse["partials_b"],
                          coarse["nx"], coarse["ny"], coarse["nz"], BLOCK_SIZE_3D], device=device)
        wp.launch(combine_partials_3d_f64_kernel, dim=1, inputs=[coarse["partials_b"], coarse["den"], coarse["n_partials"]], device=device)
        wp.launch(compute_safe_alpha_3d_kernel, dim=1, inputs=[coarse["num"], coarse["den"], coarse["alpha"]], device=device)
        wp.launch(axpy_active_device_scalar_3d_kernel, dim=coarse["dim"],
                  inputs=[coarse["z1"], coarse["x"], coarse["active"], coarse["bc_mask"], coarse["alpha"], coarse["nx"], coarse["ny"], coarse["nz"]], device=device)
        wp.launch(prolong_bilinear_axes_3d_kernel, dim=level["dim"],
                  inputs=[coarse["z1"], level["correction"], level["nx"], level["ny"], level["nz"], coarse["nx"], coarse["ny"], coarse["nz"],
                          level["coarsen_y"], level["coarsen_x"]], device=device)
        wp.launch(add_correction_3d_kernel, dim=level["dim"],
                  inputs=[level["x"], level["correction"], level["active"], level["bc_mask"], level["bc_values"], level["nx"], level["ny"], level["nz"]], device=device)
        smooth(level, post_omegas)

    if n_free == 0:
        head = fine["x"].numpy()
        if n_isolated and isolated is not None:
            head[isolated] = restore_values[isolated]
        info = {
            "solver_type": "kcycle_7point_3d",
            "implementation": "fast",
            "converged": True,
            "n_cycles_used": 0,
            "n_isolated_cells_excluded": n_isolated,
            "transient": bool(transient),
            "transient_formulation": "confined" if bool(transient) else "steady",
            "dt": float(dt_used) if bool(transient) else float("nan"),
        }
        return (head, info) if return_info else head
    rtr0 = residual_norm(fine, fine["x"], fine["residual"])
    rms0 = float(np.sqrt(max(rtr0, 0.0) / n_free))
    tolerance = float(max(abs_tol_min, rel_tol * rms0))
    graph = (
        workspace_record.get("cycle_graph")
        if str(device).startswith("cuda")
        else None
    )
    graph_reused = graph is not None and graph is not False
    graph_built = False
    graph_failed = False
    cycles = 0
    converged = rms0 <= tolerance
    dh_rms = float("nan")
    dh_max = float("nan")
    for cycle in range(int(max_cycles)):
        if converged:
            break
        cycles = cycle + 1
        if str(device).startswith("cuda") and graph is not False:
            if graph is None:
                try:
                    with wp.ScopedCapture(device=device) as capture:
                        kcycle(0)
                    graph = capture.graph
                    graph_built = graph is not None
                    workspace_record["cycle_graph"] = graph
                    if graph is not None:
                        wp.capture_launch(graph)
                except Exception:
                    graph = False
                    workspace_record["cycle_graph"] = False
                    graph_failed = True
                    kcycle(0)
            else:
                wp.capture_launch(graph)
        else:
            kcycle(0)
        if cycle % int(check_every_no) != int(check_every_no) - 1 and cycle + 1 != int(max_cycles):
            continue
        rtr = residual_norm(fine, fine["x"], fine["residual"])
        rms = float(np.sqrt(max(rtr, 0.0) / n_free))
        wp.launch(zero_scalar_kernel, dim=1, inputs=[fine["num"]], device=device)
        wp.launch(zero_scalar_kernel, dim=1, inputs=[fine["dh_max"]], device=device)
        wp.launch(dh_change_reduce_3d_kernel, dim=fine["dim"],
                  inputs=[fine["x"], fine["x_prev"], fine["active"], fine["bc_mask"], fine["num"], fine["dh_max"], fine["nx"], fine["ny"], fine["nz"]], device=device)
        dh_rms = float(np.sqrt(max(float(fine["num"].numpy()[0]), 0.0) / n_free))
        dh_max = float(fine["dh_max"].numpy()[0])
        # Keep a persistent convergence-check snapshot.  Comparing against
        # the original initial head makes dh shrink artificially over time and
        # can report a false convergence; the 2D solvers use this same
        # check-to-check definition.
        wp.launch(copy_field_3d_kernel, dim=fine["dim"],
                  inputs=[fine["x"], fine["x_prev"], fine["nx"], fine["ny"], fine["nz"]], device=device)
        dh_ok = (dh_rms_tol is None or dh_rms <= float(dh_rms_tol)) and (dh_max_tol is None or dh_max <= float(dh_max_tol))
        if rms <= tolerance and dh_ok:
            converged = True
            break
    rtr_end = residual_norm(fine, fine["x"], fine["residual"])
    # Head-equivalent (Jacobi-preconditioned) residual RMS [m] on the fine
    # level: reused from the last residual evaluation.
    fine["partials_a"].fill_(0.0)
    wp.launch(
        head_residual_partials_3d_f64_kernel, dim=fine["n_cells"],
        inputs=[fine["residual"], fine["diag"], fine["active"], fine["bc_mask"],
                fine["partials_a"], fine["nx"], fine["ny"], fine["nz"], BLOCK_SIZE_3D],
        device=device,
    )
    wp.launch(combine_partials_3d_f64_kernel, dim=1, inputs=[fine["partials_a"], fine["num"], fine["n_partials"]], device=device)
    h_rms_end = float(np.sqrt(max(float(fine["num"].numpy()[0]), 0.0) / n_free))
    info = {
        "solver_type": "kcycle_7point_3d", "implementation": "fast", "fast_operator": "fp64_face_arrays",
        "n_levels": len(levels), "level_shapes": [(level["nz"], level["ny"], level["nx"]) for level in levels],
        "n_cycles_used": cycles, "max_cycles": int(max_cycles), "converged": bool(converged),
        "r_rms0": rms0, "r_rms_end": float(np.sqrt(max(rtr_end, 0.0) / n_free)), "tol_abs": tolerance,
        "h_rms_end": h_rms_end,
        "dh_rms_lastcheck": dh_rms, "dh_max_lastcheck": dh_max, "smoother": str(smoother),
        "workspace_reused": bool(workspace_record.get("reused", False)),
        "cuda_graph_built_this_call": bool(graph_built), "cuda_graph_capture_failed": bool(graph_failed),
        "cuda_graph_reused": bool(graph_reused),
        "cuda_graph_replayed": bool(
            (graph_reused and cycles > 0) or (graph_built and cycles > 1)
        ),
        "transient": bool(transient),
        "transient_formulation": "confined" if bool(transient) else "steady",
        "dt": float(dt_used) if bool(transient) else float("nan"),
        "n_isolated_cells_excluded": n_isolated,
    }
    head = np.asarray(fine["x"].numpy(), dtype=NP_FLOAT)
    if n_isolated and isolated is not None:
        # Excluded singular cells carry no flux and no residual; report their
        # initial head rather than the zero the working mask leaves behind.
        head = head.copy()
        head[isolated] = restore_values[isolated]
    return (head, info) if return_info else head


def solve_fast_confined_kcycle_3d(
    *,
    tx_p: np.ndarray,
    tx_m: np.ndarray,
    ty_p: np.ndarray,
    ty_m: np.ndarray,
    tz_p: np.ndarray,
    tz_m: np.ndarray,
    rhs: np.ndarray,
    active: np.ndarray,
    bc_mask: np.ndarray,
    bc_values: np.ndarray,
    initial_head: np.ndarray | None = None,
    storage_diag: np.ndarray | None = None,
    transient: bool = False,
    storage_coeff: np.ndarray | float | None = None,
    dt: float | None = None,
    head_prev: np.ndarray | None = None,
    dx: float = 1.0,
    dy: float | None = None,
    dz: float = 1.0,
    max_cycles: int = 40,
    max_levels: int = 6,
    min_coarse_n: int = 2,
    nu_pre: int = 2,
    nu_post: int = 2,
    nu_coarse: int = 8,
    omega: float = 0.7,
    smoother: str = "chebyshev",
    cheby_lambda_min: float = 0.1,
    cheby_lambda_max: float = 2.0,
    rel_tol: float = 5.0e-7,
    abs_tol_min: float = 5.0e-7,
    dh_rms_tol: float | None = 1.0e-4,
    dh_max_tol: float | None = None,
    check_every_no: int = 5,
    device: str = "cuda:0",
    return_info: bool = True,
    **unsupported: Any,
):
    """Solve a steady or backward-Euler confined FP64 face-array system."""
    if WP_FLOAT is not wp.float64:
        raise ValueError("implementation='fast' requires DARCY_FLOAT=float64.")
    workspace_holder = unsupported.pop("_workspace_holder", None)
    for name in ("diag_preconditioner_backend",):
        unsupported.pop(name, None)
    # Line-relaxation controls have no meaning for the face-array backend;
    # callers must reject them at the case/CLI boundary instead of having the
    # backend silently ignore a requested smoother.
    line_controls = {name: unsupported.pop(name) for name in (
        "line_omega", "line_sweeps_pre", "line_sweeps_post",
        "line_sweeps_coarse", "vertical_line_max_nz") if name in unsupported}
    if bool(unsupported.pop("unconfined", False)):
        raise ValueError(
            "implementation='fast' confined entry called with unconfined=True; "
            "use solve_fast_unconfined_kcycle_3d via the solver session."
        )
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise TypeError(f"Unsupported fast 3D solver controls: {names}.")
    shape = np.asarray(tx_p).shape
    if len(shape) != 3:
        raise ValueError("3D face arrays must have shape (nz, ny, nx).")
    arrays = (tx_m, ty_p, ty_m, tz_p, tz_m, rhs, active, bc_mask, bc_values)
    if any(np.asarray(value).shape != shape for value in arrays):
        raise ValueError("All fast 3D fields must share the face-array shape.")
    if line_controls:
        raise ValueError("implementation='fast' does not support vertical line-relaxation controls.")
    all_fields = (tx_p, tx_m, ty_p, ty_m, tz_p, tz_m, rhs, bc_values)
    if any(not np.all(np.isfinite(np.asarray(value))) for value in all_fields):
        raise ValueError("fast 3D fields must contain finite values.")
    if any(np.any(np.asarray(value) < 0.0) for value in (tx_p, tx_m, ty_p, ty_m, tz_p, tz_m)):
        raise ValueError("face conductances must be non-negative.")
    if not np.all(np.isfinite(np.asarray(active))) or np.any((np.asarray(active) != 0) & (np.asarray(active) != 1)):
        raise ValueError("active must be a finite binary mask.")
    if not np.all(np.isfinite(np.asarray(bc_mask))) or np.any((np.asarray(bc_mask) != 0) & (np.asarray(bc_mask) != 1)):
        raise ValueError("bc_mask must be a finite binary mask.")
    if np.any((np.asarray(bc_mask) != 0) & (np.asarray(active) == 0)):
        raise ValueError("bc_mask may only select active cells.")
    rhs_effective, storage, h_prev_used, dt_used = _prepare_7point_transient_terms(
        rhs=np.asarray(rhs, dtype=np.float64),
        storage_diag=storage_diag,
        active=np.asarray(active, dtype=np.int32),
        bc_mask=np.asarray(bc_mask, dtype=np.int32),
        bc_values=np.asarray(bc_values, dtype=np.float64),
        transient=bool(transient),
        storage_coeff=storage_coeff,
        dt=dt,
        head_prev=head_prev,
        initial_head=initial_head,
        dx=float(dx),
        dy=dy,
        dz=float(dz),
    )
    # Detect physically singular rows (zero conductance AND zero storage)
    # before any smoothing: a nonzero source there is incompatible (raise);
    # otherwise the cell is excluded from the working set instead of being
    # silently pinned by an identity equation.
    isolated = _resolve_isolated_free_cells(
        faces=(tx_p, tx_m, ty_p, ty_m, tz_p, tz_m),
        active=np.asarray(active, dtype=np.int32),
        bc_mask=np.asarray(bc_mask, dtype=np.int32),
        rhs_eff=rhs_effective,
        storage_diag=storage,
    )
    n_isolated = int(np.count_nonzero(isolated))
    if n_isolated:
        active_work = np.asarray(active, dtype=np.int32).copy()
        active_work[isolated] = 0
    else:
        active_work = np.asarray(active, dtype=np.int32)
    mode = str(smoother).strip().lower()
    if mode not in {"jacobi", "chebyshev"}:
        raise ValueError("Fast 3D smoother must be 'jacobi' or 'chebyshev'.")
    pre_omegas = (
        _chebyshev_relaxation_sequence(order=int(nu_pre), lambda_min=float(cheby_lambda_min), lambda_max=float(cheby_lambda_max))
        if mode == "chebyshev" else tuple(float(omega) for _ in range(int(nu_pre)))
    )
    post_omegas = (
        _chebyshev_relaxation_sequence(order=int(nu_post), lambda_min=float(cheby_lambda_min), lambda_max=float(cheby_lambda_max))
        if mode == "chebyshev" else tuple(float(omega) for _ in range(int(nu_post)))
    )
    coarse_omegas = tuple(float(omega) for _ in range(max(1, int(nu_coarse))))
    storage_coeff_key = (
        None
        if storage_coeff is None
        else (
            ("scalar", float(np.asarray(storage_coeff).reshape(())))
            if np.asarray(storage_coeff).shape == ()
            else ("array", id(storage_coeff), np.asarray(storage_coeff).shape)
        )
    )
    storage_diag_key = (
        None
        if storage_diag is None
        else ("array", id(storage_diag), np.asarray(storage_diag).shape)
    )
    cache_key = (
        shape, int(max_levels), int(min_coarse_n), str(device),
        id(tx_p), id(tx_m), id(ty_p), id(ty_m), id(tz_p), id(tz_m),
        id(active_work), id(bc_mask), id(bc_values), bool(transient),
        float(dt_used) if bool(transient) else None,
        float(dx), None if dy is None else float(dy), float(dz),
        storage_coeff_key, storage_diag_key,
        mode, tuple(pre_omegas), tuple(post_omegas), tuple(coarse_omegas),
    )
    cached = None if workspace_holder is None else workspace_holder.get("workspace")
    if cached is not None and cached["key"] == cache_key:
        levels = cached["levels"]
        levels[0]["rhs"].assign(np.asarray(rhs_effective, dtype=np.float64))
        workspace_reused = True
        workspace_record = cached
    else:
        host_levels = _build_levels(
            faces=(tx_p, tx_m, ty_p, ty_m, tz_p, tz_m), active=active_work,
            bc_mask=bc_mask, bc_values=bc_values, rhs=rhs_effective,
            storage_diag=storage,
            max_levels=max_levels, min_coarse_n=min_coarse_n,
        )
        levels = _upload_levels(host_levels, device)
        workspace_reused = False
        workspace_record = {
            "key": cache_key,
            "levels": levels,
            "cycle_graph": None,
        }
        if workspace_holder is not None:
            workspace_holder["workspace"] = workspace_record
    workspace_record["reused"] = workspace_reused
    bc_f = np.asarray(bc_mask, dtype=np.int32)
    n_free = int(np.count_nonzero((active_work != 0) & (bc_f == 0)))
    x0 = np.zeros(shape, dtype=np.float64) if initial_head is None else np.asarray(initial_head, dtype=np.float64).copy()
    if x0.shape != shape:
        raise ValueError("initial_head shape must match the face arrays.")
    if not np.all(np.isfinite(x0)):
        raise ValueError("initial_head must contain finite values.")
    restore0 = x0.copy()
    x0[active_work == 0] = 0.0
    x0[bc_f != 0] = np.asarray(bc_values, dtype=np.float64)[bc_f != 0]

    head, info = _run_kcycles(
        levels=levels,
        workspace_record=workspace_record,
        x0=x0,
        n_free=n_free,
        device=device,
        smoother=mode,
        pre_omegas=pre_omegas,
        post_omegas=post_omegas,
        coarse_omegas=coarse_omegas,
        max_cycles=int(max_cycles),
        rel_tol=float(rel_tol),
        abs_tol_min=float(abs_tol_min),
        dh_rms_tol=dh_rms_tol,
        dh_max_tol=dh_max_tol,
        check_every_no=int(check_every_no),
        transient=bool(transient),
        dt_used=dt_used,
        isolated=isolated,
        n_isolated=n_isolated,
        restore_values=restore0,
        return_info=True,
    )
    from DARCY_WARP_PACKAGE.physics.budget_3d import compute_3d_flow_budget

    info["budget"] = compute_3d_flow_budget(
        head=head,
        source=np.asarray(rhs, dtype=np.float64),
        faces=(tx_p, tx_m, ty_p, ty_m, tz_p, tz_m),
        active=active_work,
        bc_mask=bc_f,
        storage_diag=storage if bool(transient) else None,
        head_prev=h_prev_used if bool(transient) else None,
    )
    return (head, info) if return_info else head


def solve_fast_unconfined_kcycle_3d(
    *,
    tx_p: np.ndarray,
    tx_m: np.ndarray,
    ty_p: np.ndarray,
    ty_m: np.ndarray,
    tz_p: np.ndarray,
    tz_m: np.ndarray,
    rhs: np.ndarray,
    active: np.ndarray,
    bc_mask: np.ndarray,
    bc_values: np.ndarray,
    kx_field: np.ndarray,
    ky_field: np.ndarray,
    kz_field: np.ndarray,
    zbot_field: np.ndarray,
    ztop_field: np.ndarray | None = None,
    initial_head: np.ndarray | None = None,
    storage_diag: np.ndarray | None = None,
    transient: bool = False,
    storage_coeff: np.ndarray | float | None = None,
    dt: float | None = None,
    head_prev: np.ndarray | None = None,
    sy: np.ndarray | float | None = None,
    ss: np.ndarray | float | None = None,
    dx: float = 1.0,
    dy: float | None = None,
    dz: float = 1.0,
    max_cycles: int = 40,
    max_levels: int = 6,
    min_coarse_n: int = 2,
    nu_pre: int = 2,
    nu_post: int = 2,
    nu_coarse: int = 8,
    omega: float = 0.7,
    smoother: str = "chebyshev",
    cheby_lambda_min: float = 0.1,
    cheby_lambda_max: float = 2.0,
    rel_tol: float = 5.0e-7,
    abs_tol_min: float = 5.0e-7,
    dh_rms_tol: float | None = 1.0e-4,
    dh_max_tol: float | None = None,
    check_every_no: int = 5,
    unconfined_min_sat: float = 0.1,
    unconfined_max_picard_iter: int = 30,
    unconfined_relax: float = 0.7,
    unconfined_head_tol: float = 1.0e-3,
    unconfined_startup_mode: str = "initial_head",
    unconfined_storage_mode: str = "secant_sy",
    omega_min: float = 0.1,
    omega_max: float = 0.9,
    residual_floor_tol: float | None = 1.0e-4,
    inner_forcing_eta: float = 0.10,
    inner_head_residual_tol_min: float | None = None,
    inner_head_residual_tol_max: float = 1.0e-2,
    inner_picard_scale_max_fraction: float = 0.10,
    chebyshev_enabled: bool = True,
    chebyshev_order: int = 3,
    chebyshev_lambda_min_fraction: float = 0.1,
    chebyshev_reset_on_residual_increase: bool = True,
    chebyshev_reset_factor: float = 1.2,
    chebyshev_minor_increase_patience: int = 2,
    chebyshev_rejection_factor: float = 1.2,
    unconfined_inner_max_cycles_early: int = 10,
    unconfined_inner_max_cycles_middle: int = 25,
    unconfined_inner_max_cycles_late: int = 60,
    unconfined_inner_late_dh: float = 1.0e-2,
    unconfined_inner_middle_dh: float = 1.0,
    max_head_change_per_outer_iteration: float = 10.0,
    dry_cell_flag_threshold: float = 0.1,
    final_head_residual_tol: float | None = None,
    device: str = "cuda:0",
    return_info: bool = True,
    **unsupported: Any,
):
    """Unconfined 3D solve on the fast face-array K-cycle path.

    The shared Picard driver (``_picard_unconfined_7point_3d``) rebuilds the
    linearised operator each outer iteration from the corrected unconfined
    conductance (``build_7point_face_conductance_unconfined_from_sat``) and the
    volume-conservative secant storage.  Each inner linear solve reuses a
    persistent fast workspace whose device buffers are refreshed in place, so
    the CUDA graph captured on the first inner solve is replayed on subsequent
    outer iterations.
    """
    if WP_FLOAT is not wp.float64:
        raise ValueError("implementation='fast' requires DARCY_FLOAT=float64.")
    workspace_holder = unsupported.pop("_workspace_holder", None)
    unsupported.pop("diag_preconditioner_backend", None)
    unsupported.pop("unconfined", None)  # implied True at this entry point
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise TypeError(f"Unsupported fast 3D unconfined solver controls: {names}.")
    shape = np.asarray(tx_p).shape
    if len(shape) != 3:
        raise ValueError("3D face arrays must have shape (nz, ny, nx).")

    mode = str(smoother).strip().lower()
    if mode not in {"jacobi", "chebyshev"}:
        raise ValueError("Fast 3D smoother must be 'jacobi' or 'chebyshev'.")
    pre_omegas = (
        _chebyshev_relaxation_sequence(order=int(nu_pre), lambda_min=float(cheby_lambda_min), lambda_max=float(cheby_lambda_max))
        if mode == "chebyshev" else tuple(float(omega) for _ in range(int(nu_pre)))
    )
    post_omegas = (
        _chebyshev_relaxation_sequence(order=int(nu_post), lambda_min=float(cheby_lambda_min), lambda_max=float(cheby_lambda_max))
        if mode == "chebyshev" else tuple(float(omega) for _ in range(int(nu_post)))
    )
    coarse_omegas = tuple(float(omega) for _ in range(max(1, int(nu_coarse))))

    act0 = np.asarray(active, dtype=np.int32)
    bcm0 = np.asarray(bc_mask, dtype=np.int32)
    bcv0 = np.asarray(bc_values, dtype=np.float64)
    b0 = np.asarray(rhs, dtype=NP_FLOAT)
    for name, arr in (("kx_field", kx_field), ("ky_field", ky_field), ("kz_field", kz_field), ("zbot_field", zbot_field)):
        if np.asarray(arr).shape != shape:
            raise ValueError(f"{name} shape {np.asarray(arr).shape} expected {shape}")

    # Persistent workspace: keyed on structure + smoother controls only; all
    # operator VALUES are refreshed in place per outer iteration, so the CUDA
    # graph captured on the first inner solve stays valid.  The record lives
    # in ``state`` regardless of the session holder so graphs also survive
    # across outer iterations when no session is involved.
    cache_key = (
        "unconfined", shape, int(max_levels), int(min_coarse_n), str(device),
        float(dx), None if dy is None else float(dy), float(dz),
        mode, tuple(pre_omegas), tuple(post_omegas), tuple(coarse_omegas),
    )
    record = None if workspace_holder is None else workspace_holder.get("workspace_unconfined")
    if record is not None and record["key"] != cache_key:
        record = None

    state: dict[str, Any] = {
        "levels": None if record is None else record["levels"],
        "record": record,
    }

    def _inner_solve_fast(txp_i, txm_i, typ_i, tym_i, tzp_i, tzm_i, h0, mcycles,
                          storage_diag_iter=None, rhs_eff_iter=None):
        faces_i = (txp_i, txm_i, typ_i, tym_i, tzp_i, tzm_i)
        if storage_diag_iter is not None:
            # Secant / phreatic path: Picard-built storage + RHS, no extra
            # confined term.
            rhs_i = np.asarray(rhs_eff_iter, dtype=np.float64)
            storage_i = np.asarray(storage_diag_iter, dtype=np.float64)
            dt_i = float("nan")
            transient_i = False
        else:
            # Steady or legacy confined-volume transient inner system.
            legacy_transient = bool(transient) and storage_coeff is not None
            rhs_i, storage_i, _, dt_i = _prepare_7point_transient_terms(
                rhs=np.asarray(b0, dtype=np.float64),
                storage_diag=storage_diag,
                active=act0,
                bc_mask=bcm0,
                bc_values=bcv0,
                transient=legacy_transient,
                storage_coeff=storage_coeff if legacy_transient else None,
                dt=dt if legacy_transient else None,
                head_prev=head_prev if legacy_transient else None,
                initial_head=h0,
                dx=float(dx),
                dy=dy,
                dz=float(dz),
            )
            transient_i = legacy_transient
        # Singular-row detection on this iteration's operator.
        isolated = _resolve_isolated_free_cells(
            faces=faces_i, active=act0, bc_mask=bcm0,
            rhs_eff=rhs_i, storage_diag=storage_i,
        )
        n_isolated = int(np.count_nonzero(isolated))
        if n_isolated:
            active_i = act0.copy()
            active_i[isolated] = 0
        else:
            active_i = act0
        rec = state["record"]
        if state["levels"] is None:
            host_levels = _build_levels(
                faces=faces_i, active=active_i, bc_mask=bcm0, bc_values=bcv0,
                rhs=rhs_i, storage_diag=storage_i,
                max_levels=max_levels, min_coarse_n=min_coarse_n,
            )
            state["levels"] = _upload_levels(host_levels, device)
            rec = {
                "key": cache_key,
                "levels": state["levels"],
                "cycle_graph": None,
                "reused": False,
            }
            state["record"] = rec
            if workspace_holder is not None:
                workspace_holder["workspace_unconfined"] = rec
        else:
            _refresh_levels_in_place(
                state["levels"],
                faces=faces_i, active=active_i, bc_mask=bcm0, bc_values=bcv0,
                rhs=rhs_i, storage_diag=storage_i,
                max_levels=max_levels, min_coarse_n=min_coarse_n,
            )
            rec["reused"] = True
        levels = state["levels"]
        n_free_i = int(np.count_nonzero((active_i != 0) & (bcm0 == 0)))
        restore0 = np.asarray(h0, dtype=np.float64).copy()
        x0 = np.asarray(h0, dtype=np.float64).copy()
        x0[active_i == 0] = 0.0
        x0[bcm0 != 0] = bcv0[bcm0 != 0]
        return _run_kcycles(
            levels=levels,
            workspace_record=rec,
            x0=x0,
            n_free=n_free_i,
            device=device,
            smoother=mode,
            pre_omegas=pre_omegas,
            post_omegas=post_omegas,
            coarse_omegas=coarse_omegas,
            max_cycles=int(mcycles),
            rel_tol=float(rel_tol),
            abs_tol_min=float(abs_tol_min),
            dh_rms_tol=dh_rms_tol,
            dh_max_tol=dh_max_tol,
            check_every_no=int(check_every_no),
            transient=bool(transient_i),
            dt_used=dt_i,
            isolated=isolated,
            n_isolated=n_isolated,
            restore_values=restore0,
            return_info=True,
        )

    return _picard_unconfined_7point_3d(
        _inner_solve_fast,
        shape=shape,
        active=act0,
        bc_mask=bcm0,
        bc_values=bcv0,
        rhs=b0,
        kx=np.asarray(kx_field, dtype=NP_FLOAT),
        ky=np.asarray(ky_field, dtype=NP_FLOAT),
        kz=np.asarray(kz_field, dtype=NP_FLOAT),
        zbot=np.asarray(zbot_field, dtype=NP_FLOAT),
        ztop=ztop_field,
        initial_head=initial_head,
        storage_diag=storage_diag,
        min_sat=float(unconfined_min_sat),
        max_outer=int(unconfined_max_picard_iter),
        pic_relax=float(unconfined_relax),
        pic_tol=float(unconfined_head_tol),
        omega_min=omega_min,
        omega_max=omega_max,
        dx=float(dx),
        dy=dy,
        dz=float(dz),
        device=str(device),
        unconfined_startup_mode=unconfined_startup_mode,
        transmissivity_relaxation_enabled=False,
        inner_forcing_eta=inner_forcing_eta,
        inner_head_residual_tol_min=inner_head_residual_tol_min,
        inner_head_residual_tol_max=inner_head_residual_tol_max,
        inner_picard_scale_max_fraction=inner_picard_scale_max_fraction,
        chebyshev_enabled=chebyshev_enabled,
        chebyshev_order=chebyshev_order,
        chebyshev_lambda_min_fraction=chebyshev_lambda_min_fraction,
        chebyshev_reset_on_residual_increase=chebyshev_reset_on_residual_increase,
        chebyshev_reset_factor=chebyshev_reset_factor,
        chebyshev_minor_increase_patience=chebyshev_minor_increase_patience,
        chebyshev_rejection_factor=chebyshev_rejection_factor,
        unconfined_inner_max_cycles_early=unconfined_inner_max_cycles_early,
        unconfined_inner_max_cycles_middle=unconfined_inner_max_cycles_middle,
        unconfined_inner_max_cycles_late=unconfined_inner_max_cycles_late,
        unconfined_inner_late_dh=unconfined_inner_late_dh,
        unconfined_inner_middle_dh=unconfined_inner_middle_dh,
        max_head_change_per_outer_iteration=max_head_change_per_outer_iteration,
        residual_floor_tol=residual_floor_tol,
        dh_rms_tol=dh_rms_tol,
        diag_preconditioner_backend="device",
        linear_solver_type_label="kcycle_7point_3d",
        solver_type_label="kcycle_7point_3d_unconfined_picard",
        transient=transient,
        dt=dt,
        sy=sy,
        ss=ss,
        head_prev=head_prev,
        unconfined_storage_mode=unconfined_storage_mode,
        dry_cell_flag_threshold=dry_cell_flag_threshold,
        final_head_residual_tol=final_head_residual_tol,
        legacy_storage_coeff=storage_coeff,
        return_info=return_info,
    )


__all__ = ["solve_fast_confined_kcycle_3d", "solve_fast_unconfined_kcycle_3d"]

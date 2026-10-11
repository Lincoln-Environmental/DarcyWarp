# SPDX-License-Identifier: AGPL-3.0-only
"""FP64 face-array K-cycle for steady and transient confined 3D flow.

This backend keeps six face conductances explicit, uses block-partial
reductions, keeps K-cycle alpha calculations on device, and graph-captures a
fixed cycle on CUDA when Warp capture is available. Backward-Euler storage is
assembled with the same helper as the classic backend and carried through the
semi-coarsened hierarchy. This module is selected through
``implementation="fast"``.
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
    residual_partials_3d_f64_kernel,
)
from DARCY_WARP_PACKAGE.solvers_3d import (
    _choose_horizontal_coarsening,
    _coarsen_max_edge_axes,
    _coarsen_mean_edge_axes,
    _prepare_7point_transient_terms,
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
        bc_mask_c = _coarsen_max_edge_axes(previous["bc_mask"], coarsen_y=coarsen_y, coarsen_x=coarsen_x)
        storage_c = _coarsen_mean_edge_axes(previous["storage"], coarsen_y=coarsen_y, coarsen_x=coarsen_x)
        storage_c[(active_c == 0) | (bc_mask_c != 0)] = 0.0
        face_c = tuple(
            _coarsen_mean_edge_axes(value, coarsen_y=coarsen_y, coarsen_x=coarsen_x)
            for value in previous["faces"]
        )
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
            "implementation='fast' currently supports confined 3D solves only."
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
    rhs_effective, storage, _, dt_used = _prepare_7point_transient_terms(
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
        id(active), id(bc_mask), id(bc_values), bool(transient),
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
            faces=(tx_p, tx_m, ty_p, ty_m, tz_p, tz_m), active=active,
            bc_mask=bc_mask, bc_values=bc_values, rhs=rhs_effective,
            storage_diag=storage,
            max_levels=max_levels, min_coarse_n=min_coarse_n,
        )
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
        workspace_reused = False
        workspace_record = {
            "key": cache_key,
            "levels": levels,
            "cycle_graph": None,
        }
        if workspace_holder is not None:
            workspace_holder["workspace"] = workspace_record
    fine = levels[0]
    active_f = np.asarray(active, dtype=np.int32)
    bc_f = np.asarray(bc_mask, dtype=np.int32)
    n_free = int(np.count_nonzero((active_f != 0) & (bc_f == 0)))
    x0 = np.zeros(shape, dtype=np.float64) if initial_head is None else np.asarray(initial_head, dtype=np.float64).copy()
    if x0.shape != shape:
        raise ValueError("initial_head shape must match the face arrays.")
    if not np.all(np.isfinite(x0)):
        raise ValueError("initial_head must contain finite values.")
    x0[active_f == 0] = 0.0
    x0[bc_f != 0] = np.asarray(bc_values, dtype=np.float64)[bc_f != 0]
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
        info = {
            "solver_type": "kcycle_7point_3d",
            "implementation": "fast",
            "converged": True,
            "n_cycles_used": 0,
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
    info = {
        "solver_type": "kcycle_7point_3d", "implementation": "fast", "fast_operator": "fp64_face_arrays",
        "n_levels": len(levels), "level_shapes": [(level["nz"], level["ny"], level["nx"]) for level in levels],
        "n_cycles_used": cycles, "max_cycles": int(max_cycles), "converged": bool(converged),
        "r_rms0": rms0, "r_rms_end": float(np.sqrt(max(rtr_end, 0.0) / n_free)), "tol_abs": tolerance,
        "dh_rms_lastcheck": dh_rms, "dh_max_lastcheck": dh_max, "smoother": mode,
        "workspace_reused": bool(workspace_reused),
        "cuda_graph_built_this_call": bool(graph_built), "cuda_graph_capture_failed": bool(graph_failed),
        "cuda_graph_reused": bool(graph_reused),
        "cuda_graph_replayed": bool(
            (graph_reused and cycles > 0) or (graph_built and cycles > 1)
        ),
        "transient": bool(transient),
        "transient_formulation": "confined" if bool(transient) else "steady",
        "dt": float(dt_used) if bool(transient) else float("nan"),
    }
    head = np.asarray(fine["x"].numpy(), dtype=NP_FLOAT)
    return (head, info) if return_info else head


__all__ = ["solve_fast_confined_kcycle_3d"]

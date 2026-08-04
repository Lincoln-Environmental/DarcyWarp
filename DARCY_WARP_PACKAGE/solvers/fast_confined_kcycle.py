# SPDX-License-Identifier: AGPL-3.0-only
"""Fast steady-confined K-cycle backend (production, FP64 only).

Opt-in via ``implementation="fast"`` on ``solve_multigrid_kcycle`` /
``solver.solve(solver="confined_kcycle", ...)``.  See
``face_kernels_f64.py`` for the kernel-level contract and
``MIXED_PRECISION_CAMPAIGN.md`` for the validation campaign behind the
kernel changes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from .face_kernels_f64 import (
    _BLOCK,
    applyA_dot_partials_f64_kernel,
    combine_partials_kernel,
    combine_partials_max_kernel,
    dot_partials_f64_kernel,
    face_build_f64_kernel,
    face_check_dh_residual_f64_kernel,
    face_jacobi_f64_kernel,
    face_residual_f64_kernel,
)


class FastFaceLevel:
    """Face-conductance arrays + reduction workspace for one hierarchy level."""

    __slots__ = ("nx", "ny", "Te", "Tw", "Tn", "Ts", "diag", "partials",
                 "partials_b", "partials_c", "out", "n_partials")

    def __init__(self, level, device: str):
        self.nx = int(level.nx)
        self.ny = int(level.ny)
        shape = (self.ny, self.nx)
        self.Te = wp.zeros(shape, dtype=wp.float64, device=device)
        self.Tw = wp.zeros(shape, dtype=wp.float64, device=device)
        self.Tn = wp.zeros(shape, dtype=wp.float64, device=device)
        self.Ts = wp.zeros(shape, dtype=wp.float64, device=device)
        self.diag = wp.zeros(shape, dtype=wp.float64, device=device)
        n_cells = self.nx * self.ny
        self.n_partials = (n_cells + _BLOCK - 1) // _BLOCK
        self.partials = wp.zeros(self.n_partials, dtype=wp.float64, device=device)
        self.partials_b = wp.zeros(self.n_partials, dtype=wp.float64, device=device)
        self.partials_c = wp.zeros(self.n_partials, dtype=wp.float64, device=device)
        self.out = wp.zeros(1, dtype=wp.float64, device=device)
        self.refresh(level, device)

    def refresh(self, level, device: str) -> None:
        """Recompute face conductances in place (arrays stay put, so CUDA
        graphs captured against them remain valid after T/GHB updates)."""
        wp.launch(
            kernel=face_build_f64_kernel,
            dim=(self.ny, self.nx),
            inputs=[
                level.T_wp, level.active_wp, level.gh_mask_wp, level.ghb_factor_wp,
                self.Te, self.Tw, self.Tn, self.Ts, self.diag, self.nx, self.ny,
            ],
            device=device,
        )


def ensure_fast_face_levels(model: Any):
    """Build (or reuse cached) face-conductance levels for the current hierarchy.

    Cache validity: the backend rebuilds the hierarchy whenever the operator
    is dirty, which replaces the level objects; the cache holds a strong
    reference to the level-0 object it was built from and is invalidated on
    identity mismatch.  In-place T/GHB updates (``update_T_in_place`` etc.)
    keep the level objects alive and instead set ``model._fast_faces_stale``;
    the face values are then recomputed into the same arrays so captured
    CUDA graphs remain valid.
    """
    levels = model.mg_levels
    cache = getattr(model, "_fast_face_cache", None)
    if (
        cache is not None
        and cache["levels_id"] == id(levels)
        and cache["level0"] is levels[0]
        and len(cache["faces"]) == len(levels)
    ):
        if getattr(model, "_fast_faces_stale", False):
            for fl, level in zip(cache["faces"], levels):
                fl.refresh(level, str(model.device_str))
            model._fast_faces_stale = False
        return cache["faces"]
    faces = [FastFaceLevel(level, str(model.device_str)) for level in levels]
    model._fast_faces_stale = False
    model._fast_face_cache = {
        "levels_id": id(levels),
        "level0": levels[0],
        "faces": faces,
    }
    return faces


def solve_confined_kcycle_fast_backend(
    model: Any,
    *,
    max_cycles: int = 20,
    nu_pre: int = 2,
    nu_post: int = 2,
    nu_coarse: int = 10,
    omega: float = 0.7,
    rel_tol: float = 5.0e-7,
    abs_tol_min: float = 5.0e-7,
    initial_head: np.ndarray | None = None,
    max_levels: int = 6,
    return_info: bool = True,
    check_every_no: int = 5,
    dh_rms_tol: float | None = 1.0e-4,
    dh_max_tol: float | None = None,
    dh_max_factor: float = 5.0,
    min_coarse_cells: int | None = 500,
    fallback_to_pcg: bool = True,
    divergence_cycle_start: int = 100,
    divergence_residual_factor: float = 3.0,
    fallback_pcg_max_iter: int | None = None,
    fallback_pcg_history_every: int | None = None,
    smoother: str = "chebyshev",
    cheby_lambda_min: float = 0.1,
    cheby_lambda_max: float = 2.0,
):
    """Steady confined K-cycle, fast face-array implementation (FP64).

    Convergence semantics mirror the classic backend: initial-residual-
    relative tolerance with abs floor, head-change safeguards, check cadence,
    PCG divergence fallback, and the same info fields.
    """
    from DARCY_WARP_PACKAGE import warped_darcy as kernel_module

    self = model
    np_ = kernel_module.np
    WP_FLOAT = kernel_module.WP_FLOAT
    _chebyshev_relaxation_sequence = kernel_module._chebyshev_relaxation_sequence
    add_correction_kernel = kernel_module.add_correction_kernel
    axpy_active_scalar_2dmask_kernel = kernel_module.axpy_active_scalar_2dmask_kernel
    axpy_active_scalar_kernel = kernel_module.axpy_active_scalar_kernel
    check_rtr_converged_kernel = kernel_module.check_rtr_converged_kernel
    compute_head_residual_no_storage_kernel = kernel_module.compute_head_residual_no_storage_kernel
    compute_safe_alpha_kernel = kernel_module.compute_safe_alpha_kernel
    copy_field_kernel = kernel_module.copy_field_kernel
    prolong_bilinear_any_kernel = kernel_module.prolong_bilinear_any_kernel
    restrict_blockavg_kernel = kernel_module.restrict_blockavg_kernel
    zero_scalar_kernel = kernel_module.zero_scalar_kernel
    device = self.device_str

    if WP_FLOAT is not wp.float64:
        raise ValueError(
            "implementation='fast' currently supports FP64 models only "
            "(DARCY_FLOAT=float64); use the classic implementation for FP32."
        )
    if float(self.head_scale) != 1.0:
        raise ValueError(
            "K-cycle runs in physical head units only. "
            "Set head_scale=1.0 for K-cycle, or use PCG / 2-level MG if you want scaling."
        )

    # Track storage state transitions the same way the classic backend does.
    storage_was_active = bool(self._storage_active)
    self._storage_active = False
    cleared_stale_storage = self._clear_transient_storage_state()
    if cleared_stale_storage or storage_was_active:
        self._update_fine_diag_preconditioner()
        self._operator_dirty = True
        self._kcycle_graph = None

    if not hasattr(self, "_kcycle_fast_graph"):
        self._kcycle_fast_graph = None
        self._kcycle_fast_graph_shape = None

    if self._operator_dirty or self.mg_levels is None:
        self.build_hierarchy(
            max_levels=int(max_levels),
            min_coarse_n=4,
            min_coarse_cells=min_coarse_cells,
        )
        self._kcycle_fast_graph = None

    levels = self.mg_levels
    if levels is None or len(levels) < 1:
        raise RuntimeError("No multigrid levels available. build_hierarchy() failed.")

    for lvl in levels:
        shape = (int(lvl.ny), int(lvl.nx))
        if getattr(lvl, "gh_mask_wp", None) is None or getattr(lvl, "ghb_factor_wp", None) is None:
            if bool(self.use_ghb):
                raise RuntimeError(
                    "Hierarchy level is missing GHB arrays (gh_mask_wp/ghb_factor_wp) "
                    "while use_ghb=True. Rebuild the hierarchy for this model."
                )
            if getattr(lvl, "gh_mask_wp", None) is None:
                lvl.gh_mask_wp = wp.zeros(shape, dtype=wp.int32, device=device)
            if getattr(lvl, "ghb_factor_wp", None) is None:
                lvl.ghb_factor_wp = wp.zeros(shape, dtype=WP_FLOAT, device=device)

    # Face arrays are established only after the GHB arrays above exist (the
    # face build kernel reads them).
    fast_levels = ensure_fast_face_levels(self)

    lvl0 = levels[0]
    ny0 = int(lvl0.ny)
    nx0 = int(lvl0.nx)
    dim0 = (ny0, nx0)

    required = (
        "b_wp", "x_wp", "r_wp", "Ax_wp", "e_wp", "rho_buf", "converged_flag",
        "rTr_buf", "x_prev_wp", "dh_max_buf",
    )
    for name in required:
        if getattr(lvl0, name, None) is None:
            raise RuntimeError(
                f"Level 0 missing {name}. Ensure build_hierarchy() allocates all level buffers."
            )
    if tuple(lvl0.b_wp.shape) != (ny0, nx0) or tuple(lvl0.x_wp.shape) != (ny0, nx0):
        raise RuntimeError("Level 0 buffers have wrong shape. Rebuild hierarchy for this geometry.")

    dh_rms_tol_f = None if dh_rms_tol is None else float(dh_rms_tol)
    if dh_max_tol is None:
        dh_max_tol = None if dh_rms_tol_f is None else float(dh_max_factor) * dh_rms_tol_f
    else:
        dh_max_tol = float(dh_max_tol)

    max_cycles_i = int(max_cycles)
    fallback_to_pcg_b = bool(fallback_to_pcg)
    divergence_cycle_start_i = max(1, int(divergence_cycle_start))
    divergence_residual_factor_f = float(divergence_residual_factor)
    if divergence_residual_factor_f <= 0.0:
        raise ValueError("divergence_residual_factor must be positive.")
    if fallback_pcg_max_iter is None:
        fallback_pcg_max_iter_i = max(5000, 50 * max_cycles_i)
    else:
        fallback_pcg_max_iter_i = int(fallback_pcg_max_iter)
        if fallback_pcg_max_iter_i < 1:
            raise ValueError("fallback_pcg_max_iter must be >= 1 when provided.")
    fallback_pcg_history_every_i = (
        None if fallback_pcg_history_every is None else int(fallback_pcg_history_every)
    )
    if fallback_pcg_history_every_i is not None and fallback_pcg_history_every_i <= 0:
        fallback_pcg_history_every_i = None

    smoother_mode = str(smoother).strip().lower()
    if smoother_mode not in {"chebyshev", "jacobi"}:
        raise ValueError("smoother must be 'chebyshev' or 'jacobi'.")
    if smoother_mode == "chebyshev":
        pre_omegas = _chebyshev_relaxation_sequence(
            order=int(nu_pre), lambda_min=float(cheby_lambda_min), lambda_max=float(cheby_lambda_max)
        )
        post_omegas = _chebyshev_relaxation_sequence(
            order=int(nu_post), lambda_min=float(cheby_lambda_min), lambda_max=float(cheby_lambda_max)
        )
    else:
        omega_f = float(omega)
        pre_omegas = tuple(omega_f for _ in range(int(nu_pre)))
        post_omegas = tuple(omega_f for _ in range(int(nu_post)))
    if len(pre_omegas) == 0:
        pre_omegas = (float(omega),)
    if len(post_omegas) == 0:
        post_omegas = (float(omega),)
    coarse_omegas = tuple(float(omega) for _ in range(int(nu_coarse)))

    # Solver-level CPU staging buffer for the initial head guess.
    if (
        not hasattr(self, "_kcycle_stage_x")
        or self._kcycle_stage_x is None
        or tuple(self._kcycle_stage_x.shape) != (ny0, nx0)
    ):
        self._kcycle_stage_x = wp.zeros((ny0, nx0), dtype=WP_FLOAT, device="cpu")

    # Finest RHS assembled via the model's selected backend (same as classic).
    self._build_rhs_fine(lvl0.b_wp)
    lvl0.storage_diag_host = None

    # Initial guess (host), then copy into persistent lvl0.x_wp
    x0 = np.zeros((ny0, nx0), dtype=np.float64)
    if initial_head is not None:
        init_arr = np.asarray(initial_head, dtype=np.float64)
        if init_arr.shape != (ny0, nx0):
            raise ValueError(f"initial_head must have shape ({ny0}, {nx0}), got {init_arr.shape}")
        x0[:, :] = init_arr
    bc_idx = np.asarray(self.bc_mask_host, dtype=np.int32) != 0
    x0[bc_idx] = np.asarray(self.bc_values_host, dtype=np.float64)[bc_idx]
    x0[np.asarray(self.active_host, dtype=np.int32) == 0] = 0.0
    stage_x_np = self._kcycle_stage_x.numpy()
    stage_x_np[...] = x0
    wp.copy(lvl0.x_wp, self._kcycle_stage_x)

    # Snapshot initial x for dvclose-like metrics
    wp.launch(kernel=copy_field_kernel, dim=dim0,
              inputs=[lvl0.x_wp, lvl0.x_prev_wp, nx0, ny0], device=device)

    # Zero coarse level buffers (still standalone; no reallocs)
    for k in range(1, len(levels)):
        levels[k].x_wp.fill_(WP_FLOAT(0.0))
        levels[k].b_wp.fill_(WP_FLOAT(0.0))
        levels[k].r_wp.fill_(WP_FLOAT(0.0))
        levels[k].Ax_wp.fill_(WP_FLOAT(0.0))
        levels[k].e_wp.fill_(WP_FLOAT(0.0))
        levels[k].z_wp.fill_(WP_FLOAT(0.0))
        levels[k].p_wp.fill_(WP_FLOAT(0.0))
        levels[k].Ap_wp.fill_(WP_FLOAT(0.0))
        levels[k].rTr_buf.fill_(0.0)
        levels[k].rho_buf.fill_(0.0)
        levels[k].rho_new_buf.fill_(0.0)
        levels[k].pAp_buf.fill_(0.0)
        levels[k].alpha_buf.fill_(0.0)
        levels[k].beta_buf.fill_(0.0)
        levels[k].converged_flag.fill_(0)
        if getattr(levels[k], "dh_max_buf", None) is not None:
            levels[k].dh_max_buf.fill_(0.0)
        if getattr(levels[k], "x_prev_wp", None) is not None:
            levels[k].x_prev_wp.fill_(WP_FLOAT(0.0))

    active_host_i32 = np.asarray(self.active_host, dtype=np.int32)
    bc_host_i32 = np.asarray(self.bc_mask_host, dtype=np.int32)
    n_free0 = int(np.count_nonzero((active_host_i32 != 0) & (bc_host_i32 == 0)))
    if n_free0 <= 0:
        head_out = lvl0.x_wp.numpy()
        info = {"solver_type": "kcycle", "implementation": "fast",
                "n_cycles_used": 0, "converged": True}
        return (head_out, info) if return_info else head_out

    f0 = fast_levels[0]
    n0 = nx0 * ny0

    def face_residual_into(fl, x_arr, b_arr, r_arr, level):
        wp.launch(
            kernel=face_residual_f64_kernel,
            dim=(fl.ny, fl.nx),
            inputs=[x_arr, b_arr, fl.Te, fl.Tw, fl.Tn, fl.Ts, fl.diag,
                    level.active_wp, level.bc_mask_wp, r_arr, fl.nx, fl.ny],
            device=device,
        )

    # Initial residual for tol computation (one scalar readback per solve)
    face_residual_into(f0, lvl0.x_wp, lvl0.b_wp, lvl0.r_wp, lvl0)
    wp.launch(kernel=dot_partials_f64_kernel, dim=n0,
              inputs=[lvl0.r_wp, lvl0.r_wp, lvl0.active_wp, lvl0.bc_mask_wp,
                      f0.partials, nx0, ny0, _BLOCK],
              device=device)
    wp.launch(kernel=combine_partials_kernel, dim=1,
              inputs=[f0.partials, lvl0.rTr_buf, f0.n_partials], device=device)
    rTr0 = float(lvl0.rTr_buf.numpy()[0])
    r_rms0 = float(np.sqrt(max(rTr0, 0.0) / float(n_free0)))
    tol_abs = float(max(abs_tol_min, rel_tol * r_rms0))
    thr_rTr = wp.float64((tol_abs * tol_abs) * float(n_free0))

    def smooth(fl, level, omegas) -> None:
        nxL, nyL = fl.nx, fl.ny
        dimL = (nyL, nxL)
        x_in = level.x_wp
        x_out = level.Ax_wp
        for omega_step in omegas:
            wp.launch(
                kernel=face_jacobi_f64_kernel,
                dim=dimL,
                inputs=[level.b_wp, x_in, fl.Te, fl.Tw, fl.Tn, fl.Ts, fl.diag,
                        level.active_wp, level.bc_mask_wp, level.bc_values_wp,
                        float(omega_step), nxL, nyL, x_out],
                device=device,
            )
            x_in, x_out = x_out, x_in
        if x_in is not level.x_wp:
            wp.launch(kernel=copy_field_kernel, dim=dimL,
                      inputs=[x_in, level.x_wp, nxL, nyL], device=device)

    def kcycle(level_id: int) -> None:
        fl = fast_levels[level_id]
        level = levels[level_id]
        nxL, nyL = fl.nx, fl.ny
        dimL = (nyL, nxL)

        smooth(fl, level, pre_omegas)
        face_residual_into(fl, level.x_wp, level.b_wp, level.r_wp, level)

        if level_id == len(levels) - 1:
            # Coarsest: fixed Jacobi block (no PCG, no reductions).
            smooth(fl, level, coarse_omegas)
            return

        fc = fast_levels[level_id + 1]
        coarse = levels[level_id + 1]
        nxC, nyC = fc.nx, fc.ny
        dimC = (nyC, nxC)

        wp.launch(kernel=restrict_blockavg_kernel, dim=dimC,
                  inputs=[level.r_wp, level.active_wp, level.bc_mask_wp, coarse.b_wp,
                          nxL, nyL, nxC, nyC],
                  device=device)
        coarse.x_wp.fill_(WP_FLOAT(0.0))
        kcycle(level_id + 1)

        coarse_is_coarsest = (level_id + 1) == len(levels) - 1
        if coarse_is_coarsest:
            wp.launch(kernel=copy_field_kernel, dim=dimC,
                      inputs=[coarse.x_wp, coarse.e_wp, nxC, nyC], device=device)
            z1_wp = coarse.e_wp
        else:
            wp.launch(kernel=copy_field_kernel, dim=dimC,
                      inputs=[coarse.x_wp, coarse.z_wp, nxC, nyC], device=device)
            z1_wp = coarse.z_wp

        # r1 = b - A z1, then second descent on r1
        face_residual_into(fc, z1_wp, coarse.b_wp, coarse.r_wp, coarse)
        wp.launch(kernel=copy_field_kernel, dim=dimC,
                  inputs=[coarse.r_wp, coarse.b_wp, nxC, nyC], device=device)

        coarse.x_wp.fill_(WP_FLOAT(0.0))
        kcycle(level_id + 1)

        # alpha = (r1 . z2) / (z2 . A z2); z1 += alpha * z2
        n_c = nxC * nyC
        wp.launch(kernel=dot_partials_f64_kernel, dim=n_c,
                  inputs=[coarse.b_wp, coarse.x_wp, coarse.active_wp,
                          coarse.bc_mask_wp, fc.partials, nxC, nyC, _BLOCK],
                  device=device)
        wp.launch(kernel=combine_partials_kernel, dim=1,
                  inputs=[fc.partials, coarse.rho_buf, fc.n_partials], device=device)
        wp.launch(kernel=applyA_dot_partials_f64_kernel, dim=n_c,
                  inputs=[coarse.x_wp, fc.Te, fc.Tw, fc.Tn, fc.Ts, fc.diag,
                          coarse.active_wp, coarse.bc_mask_wp, fc.partials,
                          nxC, nyC, _BLOCK],
                  device=device)
        wp.launch(kernel=combine_partials_kernel, dim=1,
                  inputs=[fc.partials, coarse.pAp_buf, fc.n_partials], device=device)
        wp.launch(kernel=compute_safe_alpha_kernel, dim=1,
                  inputs=[coarse.rho_buf, coarse.pAp_buf, coarse.alpha_buf],
                  device=device)

        if len(coarse.active_wp.shape) == 1:
            _axpy_k = axpy_active_scalar_kernel
        else:
            _axpy_k = axpy_active_scalar_2dmask_kernel
        wp.launch(kernel=_axpy_k, dim=dimC,
                  inputs=[z1_wp, coarse.x_wp, coarse.active_wp, coarse.bc_mask_wp,
                          coarse.alpha_buf, nxC, nyC],
                  device=device)

        wp.launch(kernel=prolong_bilinear_any_kernel, dim=dimL,
                  inputs=[z1_wp, level.e_wp, nxL, nyL, nxC, nyC], device=device)
        wp.launch(kernel=add_correction_kernel, dim=dimL,
                  inputs=[level.x_wp, level.e_wp, level.active_wp, level.bc_mask_wp,
                          level.bc_values_wp, nxL, nyL],
                  device=device)

        smooth(fl, level, post_omegas)

    # Outer cycles
    n_cycles_used = 0
    converged = False
    check_every = check_every_no

    graph_key = (
        "kcycle_fast",
        int(len(levels)),
        tuple((int(l.ny), int(l.nx)) for l in levels),
        int(nu_pre), int(nu_post), int(nu_coarse),
        str(smoother_mode),
        tuple(float(v) for v in pre_omegas),
        tuple(float(v) for v in post_omegas),
        float(omega),
    )
    graph_built_this_call = False
    graph_capture_failed = False
    use_cuda_graph = str(device).startswith("cuda")

    dh_rms_lastcheck = float("nan")
    dh_max_lastcheck = float("nan")
    history: list[dict[str, float | int | bool | None]] = [
        {
            "cycle": 0,
            "r_rms": float(r_rms0),
            "tol_abs": float(tol_abs),
            "dh_rms": None,
            "dh_max": None,
            "res_ok": None,
            "dh_ok": None,
        }
    ]

    if rTr0 <= float(thr_rTr):
        converged = True
        n_cycles_used = 0

    for cyc in range(max_cycles_i if not converged else 0):
        n_cycles_used = cyc + 1

        if (not use_cuda_graph) or graph_capture_failed:
            kcycle(0)
        elif self._kcycle_fast_graph is None or self._kcycle_fast_graph_shape != graph_key:
            try:
                with wp.ScopedCapture() as cap:
                    kcycle(0)
            except Exception:
                # Capture failed (e.g. profiling tools active): fall back to
                # eager execution for the rest of this solve call.
                self._kcycle_fast_graph = None
                self._kcycle_fast_graph_shape = None
                graph_capture_failed = True
                kcycle(0)
            else:
                self._kcycle_fast_graph = cap.graph
                self._kcycle_fast_graph_shape = graph_key
                graph_built_this_call = True
                # ScopedCapture records without executing; launch immediately
                # so every counted cycle is an executed cycle.
                wp.capture_launch(self._kcycle_fast_graph)
        else:
            wp.capture_launch(self._kcycle_fast_graph)

        if (cyc % int(check_every)) != (int(check_every) - 1):
            continue

        # dh stats + flux residual check in one face-kernel pass
        wp.launch(
            kernel=face_check_dh_residual_f64_kernel,
            dim=n0,
            inputs=[lvl0.x_wp, lvl0.x_prev_wp, lvl0.b_wp,
                    f0.Te, f0.Tw, f0.Tn, f0.Ts, f0.diag,
                    lvl0.active_wp, lvl0.bc_mask_wp,
                    f0.partials, f0.partials_b, f0.partials_c,
                    nx0, ny0, _BLOCK],
            device=device,
        )
        wp.launch(kernel=combine_partials_kernel, dim=1,
                  inputs=[f0.partials, lvl0.rho_buf, f0.n_partials], device=device)
        wp.launch(kernel=combine_partials_max_kernel, dim=1,
                  inputs=[f0.partials_b, lvl0.dh_max_buf, f0.n_partials], device=device)
        wp.launch(kernel=combine_partials_kernel, dim=1,
                  inputs=[f0.partials_c, lvl0.rTr_buf, f0.n_partials], device=device)
        wp.launch(kernel=check_rtr_converged_kernel, dim=1,
                  inputs=[lvl0.rTr_buf, thr_rTr, lvl0.converged_flag], device=device)

        dh2 = float(lvl0.rho_buf.numpy()[0])
        dh_rms_lastcheck = float(np.sqrt(max(dh2, 0.0) / float(n_free0)))
        dh_max_lastcheck = float(lvl0.dh_max_buf.numpy()[0])

        dh_ok = True
        if dh_max_tol is not None and dh_rms_tol_f is not None:
            dh_ok = dh_max_lastcheck <= float(dh_max_tol) and dh_rms_lastcheck <= float(dh_rms_tol_f)

        res_ok = int(lvl0.converged_flag.numpy()[0]) != 0
        rTr_check = float(lvl0.rTr_buf.numpy()[0])
        r_rms_check = float(np.sqrt(max(rTr_check, 0.0) / float(n_free0)))
        history.append(
            {
                "cycle": int(n_cycles_used),
                "r_rms": float(r_rms_check),
                "tol_abs": float(tol_abs),
                "dh_rms": float(dh_rms_lastcheck),
                "dh_max": float(dh_max_lastcheck),
                "res_ok": bool(res_ok),
                "dh_ok": bool(dh_ok),
            }
        )

        if res_ok and dh_ok:
            converged = True
            break

        if (
            fallback_to_pcg_b
            and n_cycles_used >= divergence_cycle_start_i
            and r_rms_check > (divergence_residual_factor_f * r_rms0)
        ):
            fallback_head0 = np.asarray(lvl0.x_wp.numpy(), dtype=np.float64)
            head_pcg, info_pcg = self._solve_pcg_device_loop(
                max_iter=int(fallback_pcg_max_iter_i),
                rel_tol=float(rel_tol),
                abs_tol_min=float(abs_tol_min),
                initial_head=fallback_head0,
                history_every=fallback_pcg_history_every_i,
            )
            info_pcg = dict(info_pcg)
            info_pcg["implementation"] = "fast"
            info_pcg["fallback_from"] = "kcycle"
            info_pcg["fallback_reason"] = "diverging_residual"
            info_pcg["fallback_trigger_cycle"] = int(n_cycles_used)
            info_pcg["fallback_trigger_r_rms"] = float(r_rms_check)
            info_pcg["fallback_trigger_threshold"] = float(divergence_residual_factor_f * r_rms0)
            info_pcg["kcycle_history_before_fallback"] = list(history)
            return (head_pcg, info_pcg) if return_info else head_pcg

    # Final head pullback
    head_out = lvl0.x_wp.numpy()

    # Final flux residual RMS for reporting (face residual + block reduction)
    face_residual_into(f0, lvl0.x_wp, lvl0.b_wp, lvl0.r_wp, lvl0)
    wp.launch(kernel=dot_partials_f64_kernel, dim=n0,
              inputs=[lvl0.r_wp, lvl0.r_wp, lvl0.active_wp, lvl0.bc_mask_wp,
                      f0.partials, nx0, ny0, _BLOCK],
              device=device)
    wp.launch(kernel=combine_partials_kernel, dim=1,
              inputs=[f0.partials, lvl0.rTr_buf, f0.n_partials], device=device)
    rTr_end = float(lvl0.rTr_buf.numpy()[0])
    r_rms_end = float(np.sqrt(max(rTr_end, 0.0) / float(n_free0)))

    # Head-equivalent residual RMS for reporting (classic kernel, once)
    wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[lvl0.rTr_buf], device=device)
    wp.launch(
        kernel=compute_head_residual_no_storage_kernel,
        dim=dim0,
        inputs=[lvl0.x_wp, lvl0.b_wp, lvl0.T_wp, lvl0.active_wp, lvl0.bc_mask_wp,
                lvl0.gh_mask_wp, lvl0.ghb_factor_wp, lvl0.r_wp, lvl0.rTr_buf, nx0, ny0],
        device=device,
    )
    hrTr_end = float(lvl0.rTr_buf.numpy()[0])
    h_rms_end = float(np.sqrt(max(hrTr_end, 0.0) / float(n_free0)))

    dh_rms_end = float(dh_rms_lastcheck)
    dh_max_end = float(dh_max_lastcheck)

    info = {
        "solver_type": "kcycle",
        "implementation": "fast",
        "n_levels": int(len(levels)),
        "max_cycles": int(max_cycles),
        "n_cycles_used": int(n_cycles_used),
        "nu_pre": int(nu_pre),
        "nu_post": int(nu_post),
        "nu_coarse": int(nu_coarse),
        "coarsest_solve": "jacobi_block",
        "smoother": str(smoother_mode),
        "omega": float(omega),
        "cheby_lambda_min": float(cheby_lambda_min) if smoother_mode == "chebyshev" else float("nan"),
        "cheby_lambda_max": float(cheby_lambda_max) if smoother_mode == "chebyshev" else float("nan"),
        "cheby_pre_omegas": [float(v) for v in pre_omegas],
        "cheby_post_omegas": [float(v) for v in post_omegas],
        "rel_tol": float(rel_tol),
        "abs_tol_min": float(abs_tol_min),
        "tol_abs": float(tol_abs),
        "r_rms0": float(r_rms0),
        "r_rms_end": float(r_rms_end),
        "h_rms_end": float(h_rms_end),
        "dh_rms_lastcheck": float(dh_rms_lastcheck),
        "dh_max_lastcheck": float(dh_max_lastcheck),
        "dh_rms_end": float(dh_rms_end),
        "dh_max_end": float(dh_max_end),
        "converged": bool(converged),
        "aq_thickness": float(self.aq_thickness),
        "use_ghb": bool(self.use_ghb),
        "diag_preconditioner_backend": self._diag_backend_env_or_default(),
        # A graph built this call is launched immediately after capture, so
        # cuda_graph_built_this_call and cuda_graph_reused are mutually
        # exclusive and every counted cycle is an executed cycle.
        "cuda_graph_reused": bool((not graph_built_this_call) and (self._kcycle_fast_graph is not None)),
        "cuda_graph_built_this_call": bool(graph_built_this_call),
        "cuda_graph_capture_failed": bool(graph_capture_failed),
        "check_every": int(check_every),
        "min_coarse_cells": None if min_coarse_cells is None else int(min_coarse_cells),
    }
    if (not history) or int(history[-1]["cycle"]) != int(n_cycles_used):
        history.append(
            {
                "cycle": int(n_cycles_used),
                "r_rms": float(r_rms_end),
                "tol_abs": float(tol_abs),
                "dh_rms": float(dh_rms_end) if np.isfinite(dh_rms_end) else None,
                "dh_max": float(dh_max_end) if np.isfinite(dh_max_end) else None,
                "res_ok": bool(r_rms_end <= float(tol_abs)),
                "dh_ok": (
                    None
                    if (dh_rms_tol_f is None or dh_max_tol is None)
                    else bool(
                        np.isfinite(dh_rms_end)
                        and np.isfinite(dh_max_end)
                        and dh_rms_end <= float(dh_rms_tol_f)
                        and dh_max_end <= float(dh_max_tol)
                    )
                ),
            }
        )
    info["history"] = history
    return (head_out, info) if return_info else head_out


__all__ = [
    "FastFaceLevel",
    "ensure_fast_face_levels",
    "solve_confined_kcycle_fast_backend",
]

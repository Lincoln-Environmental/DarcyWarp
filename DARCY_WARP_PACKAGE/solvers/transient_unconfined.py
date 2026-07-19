# SPDX-License-Identifier: AGPL-3.0-only
"""Production multi-period unconfined Picard/K-cycle driver boundary."""

from __future__ import annotations

from typing import Any

from .context import SolverContext
from .registry import select_backend
from .capabilities import CAPABILITIES
from .transient_experimental import solve_transient_unconfined_experimental

def solve_transient_unconfined_backend(
        *,
        model: Any,
        initial_head: np.ndarray,
        recharge_rates: np.ndarray,
        k_field: np.ndarray,
        zbot_field: np.ndarray,
        ztop_field: np.ndarray,
        sy: float,
        ss: float,
        dt: float,
        active: np.ndarray | None = None,
        bc_mask: np.ndarray | None = None,
        bc_values: np.ndarray | None = None,
        storage_mode: str = "mf6_convertible_secant_sy",
        storage_reference: str = "current_picard",
        solve_controls: dict | None = None,
        min_saturated_thickness: float = 0.1,
        save_diagnostics: bool = False,
        return_info: bool = True,
):
    from DARCY_WARP_PACKAGE import warped_darcy as kernel_module
    globals().update(kernel_module.__dict__)
    self = model
    """
    Step a 2D unconfined transient solve through multiple stress periods.

    This is solver infrastructure only: callers remain responsible for MF6
    artifact loading, comparisons, reporting, mass balance, and persistence.

    :param initial_head: Initial/previous head for period 1.
    :param recharge_rates: One recharge value per stress period.
    :param k_field: Hydraulic conductivity field.
    :param zbot_field: Cell bottom field.
    :param ztop_field: Cell top field.
    :param sy: Specific yield.
    :param ss: Specific storage.
    :param dt: Transient time step.
    :param active: Optional active mask; defaults to all active.
    :param bc_mask: Optional Dirichlet mask; defaults to no Dirichlet cells.
    :param bc_values: Optional Dirichlet values; defaults to initial heads.
    :param storage_mode: 2D unconfined storage mode passed to the Picard solver.
    :param storage_reference: ``current_picard`` or ``previous_period``.
    :param solve_controls: Extra controls forwarded to :meth:`solve`.
    :param min_saturated_thickness: Minimum saturated thickness.
    :param save_diagnostics: Save full-grid storage/reference arrays.
    :param return_info: Return ``(heads_per_period, info)`` when true.
    :return: Heads per period, plus diagnostics when ``return_info`` is true.
    """
    h0 = np.asarray(initial_head, dtype=NP_FLOAT)
    if h0.shape != (self.ny, self.nx):
        raise ValueError(f"initial_head shape {h0.shape} expected {(self.ny, self.nx)}")
    k = np.asarray(k_field, dtype=NP_FLOAT)
    bottom = np.asarray(zbot_field, dtype=NP_FLOAT)
    top = np.asarray(ztop_field, dtype=NP_FLOAT)
    for name, arr in (("k_field", k), ("zbot_field", bottom), ("ztop_field", top)):
        if arr.shape != h0.shape:
            raise ValueError(f"{name} shape {arr.shape} expected {h0.shape}")

    if active is None:
        active_i = np.ones(h0.shape, dtype=np.int32)
    else:
        active_i = np.asarray(active, dtype=np.int32)
    if bc_mask is None:
        bc_i = np.zeros(h0.shape, dtype=np.int32)
    else:
        bc_i = np.asarray(bc_mask, dtype=np.int32)
    if bc_values is None:
        bc_v = np.asarray(h0, dtype=NP_FLOAT)
    else:
        bc_v = np.asarray(bc_values, dtype=NP_FLOAT)
    for name, arr in (("active", active_i), ("bc_mask", bc_i), ("bc_values", bc_v)):
        if arr.shape != h0.shape:
            raise ValueError(f"{name} shape {arr.shape} expected {h0.shape}")

    rates = np.asarray(recharge_rates, dtype=NP_FLOAT).reshape(-1)
    if rates.size < 1:
        raise ValueError("recharge_rates must contain at least one period.")
    dt_f = float(dt)
    if not np.isfinite(dt_f) or dt_f <= 0.0:
        raise ValueError("dt must be finite and > 0.")

    controls = {} if solve_controls is None else dict(solve_controls)
    save_diagnostics_b = bool(controls.pop("save_transient_diagnostics", save_diagnostics))
    fast_path_controls = dict(controls)
    # Drop keys consumed by this wrapper or the device fast path so the
    # ``**controls`` spread forwarded to ``solve()`` does not inject unexpected
    # keywords.
    for _control_key in (
        "strict_head_residual_tol",
        "practical_head_residual_tol",
        "unconfined_inner_max_cycles_early",
        "unconfined_inner_max_cycles_middle",
        "unconfined_inner_max_cycles_late",
        "unconfined_inner_middle_dh",
        "unconfined_inner_late_dh",
        "adaptive_unconfined_inner_enabled",
        "adaptive_inner_initial_block_cycles",
        "adaptive_inner_min_block_cycles",
        "adaptive_inner_max_block_cycles",
        "adaptive_inner_min_total_cycles",
        "adaptive_inner_eta_initial",
        "adaptive_inner_eta_min",
        "adaptive_inner_eta_max",
        "adaptive_inner_eta_gamma",
        "adaptive_inner_eta_power",
        "adaptive_inner_good_contraction_ratio",
        "adaptive_inner_weak_contraction_ratio",
        "adaptive_inner_stall_contraction_ratio",
        "adaptive_inner_divergence_contraction_ratio",
        "adaptive_inner_stall_patience",
        "adaptive_inner_minimum_usable_reduction_ratio",
        "adaptive_inner_residual_floor",
        "adaptive_inner_relative_flow_residual_target",
        "adaptive_inner_save_block_history",
        "allow_unaccepted_transient_period",
        "use_device_transient_fast_path",
        "profile_transient_fast_path",
        "use_incremental_picard",
        "adaptive_dt_enabled",
        "adaptive_dt_min_fraction",
        "adaptive_dt_shrink_factor",
        "adaptive_dt_grow_factor",
        "adaptive_dt_strict_max_outer",
        "adaptive_dt_max_growth_steps",
        "adaptive_dt_early_shrink_enabled",
        "adaptive_dt_early_shrink_min_outer",
        "adaptive_dt_early_shrink_patience",
        "adaptive_dt_extension_enabled",
        "adaptive_dt_extension_factor",
        "adaptive_dt_extension_max_outer",
        "adaptive_dt_extension_contraction_ratio",
    ):
        controls.pop(_control_key, None)
    min_sat = float(controls.get("min_saturated_thickness", min_saturated_thickness))
    thickness = np.clip(h0 - bottom, min_sat, np.maximum(top - bottom, min_sat))
    initial_T = np.asarray(k * thickness, dtype=NP_FLOAT)
    initial_T[active_i == 0] = 0.0
    recharge_field = np.zeros(h0.shape, dtype=NP_FLOAT)
    self.build_from_fields(
        T_field=initial_T,
        R_field=recharge_field,
        active=active_i,
        bc_mask=bc_i,
        bc_values=bc_v,
    )

    n_periods = int(rates.size)
    heads_per_period = np.zeros((n_periods, self.ny, self.nx), dtype=np.float64)
    if save_diagnostics_b:
        heads_old_per_period = np.zeros_like(heads_per_period)
        storage_reference_heads = np.zeros_like(heads_per_period)
        storage_coeffs = np.zeros_like(heads_per_period)
        sy_coeffs = np.zeros_like(heads_per_period)
        ss_coeffs = np.zeros_like(heads_per_period)
        storage_terms = np.zeros_like(heads_per_period)
        sy_terms = np.zeros_like(heads_per_period)
        ss_terms = np.zeros_like(heads_per_period)
        sy_crossing_terms = np.zeros_like(heads_per_period)
    else:
        heads_old_per_period = None
        storage_reference_heads = None
        storage_coeffs = None
        sy_coeffs = None
        ss_coeffs = None
        storage_terms = None
        sy_terms = None
        ss_terms = None
        sy_crossing_terms = None
    period_infos: list[dict] = []
    period_times = np.zeros(n_periods, dtype=np.float64)
    counters = {
        "host_to_device_full_grid_copies": 0,
        "device_to_host_full_grid_copies": 0,
        "full_grid_allocations_inside_period_loop": 0,
        "full_grid_allocations_inside_outer_loop": 0,
        "hierarchy_rebuilds": 0,
        "hierarchy_rebuilds_inside_picard": 0,
        "hierarchy_device_coarse_value_refreshes": 0,
        "T_device_updates": 0,
        "storage_device_updates": 0,
        "R_device_updates": 0,
        "rhs_device_updates": 0,
        "scalar_reductions": 0,
        "gpu_scalar_synchronizations": 0,
        "head_downloads": 0,
        "full_head_downloads_inside_picard": 0,
        "host_T_builds_inside_picard": 0,
        "host_storage_builds_inside_picard": 0,
        "host_rhs_builds_inside_picard": 0,
        "host_to_device_T_uploads_inside_picard": 0,
        "host_to_device_storage_uploads_inside_picard": 0,
        "diagnostic_full_grid_arrays_saved": int(save_diagnostics_b),
        "device_side_picard_fast_path_active": 0,
    }
    if save_diagnostics_b:
        counters["full_grid_allocations_inside_period_loop"] += 9

    head_prev = np.asarray(h0, dtype=np.float64).copy()
    total_t0 = time.perf_counter()
    last_info: dict = {}

    use_device_fast_path = bool(fast_path_controls.get("use_device_transient_fast_path", False))
    use_incremental_picard = bool(fast_path_controls.get("use_incremental_picard", False))
    # Per-block h_iter = h^k + delta sync clip: large enough to be a no-op so
    # the residual check sees the true current head iterate (delta is a head
    # correction, bounded by the domain scale; the final relaxed update does
    # the real clipping via max_head_change_per_outer_iteration).
    incremental_picard_sync_max_change = 1.0e9
    fast_path = (
        use_device_fast_path
        and storage_mode == "mf6_convertible_secant_sy"
        and storage_reference == "current_picard"
    )
    if fast_path:
        controls = fast_path_controls
        if self.use_ghb:
            raise NotImplementedError("device transient fast path does not yet support GHB RHS assembly")
        counters["device_side_picard_fast_path_active"] = 1
        device = self.device_str
        n_free = int(np.count_nonzero((active_i != 0) & (bc_i == 0)))

        h_prev_wp = wp.array(head_prev, dtype=WP_FLOAT, device=device)
        h_iter_wp = wp.array(head_prev, dtype=WP_FLOAT, device=device)
        h_substep_start_wp = wp.array(head_prev, dtype=WP_FLOAT, device=device)
        h_snapshot_wp = wp.array(head_prev, dtype=WP_FLOAT, device=device)
        h_inner_snapshot_wp = wp.array(head_prev, dtype=WP_FLOAT, device=device)
        bottom_wp = wp.array(bottom, dtype=WP_FLOAT, device=device)
        top_wp = wp.array(top, dtype=WP_FLOAT, device=device)
        k_field_wp = wp.array(k, dtype=WP_FLOAT, device=device)

        storage_diag_wp = wp.zeros((self.ny, self.nx), dtype=WP_FLOAT, device=device)
        storage_diag_prev_wp = wp.zeros((self.ny, self.nx), dtype=WP_FLOAT, device=device)
        storage_coeff_wp = wp.zeros((self.ny, self.nx), dtype=WP_FLOAT, device=device)
        sy_coeff_wp = wp.zeros((self.ny, self.nx), dtype=WP_FLOAT, device=device)
        ss_coeff_wp = wp.zeros((self.ny, self.nx), dtype=WP_FLOAT, device=device)
        rhs_eff_wp = wp.zeros((self.ny, self.nx), dtype=WP_FLOAT, device=device)

        # Incremental-Picard (correction) scratch buffers. ``delta_wp`` holds
        # the per-outer-iteration correction solved from ``A*delta = r^k``;
        # ``residual_wp`` holds the nonlinear residual field ``b - A*h^k``;
        # ``zero_bc_values_wp`` pins the correction to 0 on Dirichlet cells;
        # ``delta_snapshot_wp`` supports adaptive-block rollback. Allocated
        # unconditionally (cheap) and only touched when ``use_incremental_picard``.
        delta_wp = wp.zeros((self.ny, self.nx), dtype=WP_FLOAT, device=device)
        residual_wp = wp.zeros((self.ny, self.nx), dtype=WP_FLOAT, device=device)
        zero_bc_values_wp = wp.zeros((self.ny, self.nx), dtype=WP_FLOAT, device=device)
        delta_snapshot_wp = wp.zeros((self.ny, self.nx), dtype=WP_FLOAT, device=device)

        storage_change_sum_sq_buf = wp.zeros(1, dtype=wp.float64, device=device)
        storage_change_max_buf = wp.zeros(1, dtype=wp.float64, device=device)
        dh_max_buf = wp.zeros(1, dtype=wp.float64, device=device)
        dh_rms_buf = wp.zeros(1, dtype=wp.float64, device=device)
        flow_rTr_buf = wp.zeros(1, dtype=wp.float64, device=device)
        head_rTr_buf = wp.zeros(1, dtype=wp.float64, device=device)
        rhs_rTr_buf = wp.zeros(1, dtype=wp.float64, device=device)
        converged_flag_buf = wp.zeros(1, dtype=wp.int32, device=device)
        head_nonfinite_flag_buf = wp.zeros(1, dtype=wp.int32, device=device)

        self.storage_diag_wp = storage_diag_wp
        self._storage_active = True

        if not hasattr(self, "storage_diag_host") or self.storage_diag_host is None:
            self.storage_diag_host = np.zeros_like(self.T_field_host)

        if self.mg_levels is None:
            self.build_hierarchy(
                max_levels=int(controls.get("max_levels", 5)),
                min_coarse_n=4,
                min_coarse_cells=controls.get("min_coarse_cells", 500),
            )
        if self.mg_levels:
            self.mg_levels[0].T_wp = self.T_wp
            self.mg_levels[0].storage_diag_wp = storage_diag_wp
        fast_path_coarse_operator_mode = "device_refreshed_dynamic_coarse_operator"

        max_outer = int(controls.get("unconfined_max_picard_iter", controls.get("max_outer_iterations", 100)))
        max_cycles_hard_i = int(controls.get("max_cycles", 200))
        hclose = float(controls.get("unconfined_head_tol", controls.get("hclose", 1.0e-4)))
        strict_head_residual_tol_f = float(controls.get("strict_head_residual_tol", hclose))
        min_practical_outer_iterations_i = int(controls.get("min_practical_outer_iterations", 20))
        practical_head_residual_tol_f = float(
            controls.get("practical_head_residual_tol", controls.get("practical_residual_tol", 1.0e-4))
        )
        practical_residual_tol_alias_used = "practical_head_residual_tol" not in controls and "practical_residual_tol" in controls
        practical_dh_rms_tol_f = float(controls.get("practical_dh_rms_tol", 3.0e-3))
        practical_storage_diag_change_rms_tol_f = float(
            controls.get("practical_storage_diag_change_rms_tol", 30.0)
        )
        practical_picard_acceptance_enabled_b = bool(
            controls.get("practical_picard_acceptance_enabled", False)
        )
        omega_current_f = float(controls.get("unconfined_relax", controls.get("omega", 0.8)))
        omega_min_f = float(controls.get("omega_min", 0.05))
        omega_max_f = float(controls.get("omega_max", 0.75))
        if not (0.0 < omega_min_f <= omega_max_f):
            raise ValueError("omega_min and omega_max must satisfy 0 < omega_min <= omega_max.")
        omega_current_f = min(max(omega_current_f, omega_min_f), omega_max_f)
        max_update_f = float(controls.get("max_head_change_per_outer_iteration", 5.0))
        if max_update_f <= 0.0 or not np.isfinite(max_update_f):
            raise ValueError("max_head_change_per_outer_iteration must be positive and finite.")
        inner_max_cycles_early_i = int(controls.get("unconfined_inner_max_cycles_early", 10))
        inner_max_cycles_middle_i = int(controls.get("unconfined_inner_max_cycles_middle", 25))
        inner_max_cycles_late_i = int(controls.get("unconfined_inner_max_cycles_late", 60))
        inner_middle_dh_f = float(controls.get("unconfined_inner_middle_dh", 1.0))
        inner_late_dh_f = float(controls.get("unconfined_inner_late_dh", 1.0e-2))
        inner_head_residual_tol_min_f = float(
            controls.get("inner_head_residual_tol_min", controls.get("inner_head_residual_tol", hclose))
        )
        inner_head_residual_tol_max_f = float(controls.get("inner_head_residual_tol_max", 1.0e-2))
        inner_picard_scale_max_fraction_f = float(controls.get("inner_picard_scale_max_fraction", 0.10))
        allow_unaccepted_transient_period_b = bool(controls.get("allow_unaccepted_transient_period", False))
        startup_mode = str(controls.get("unconfined_startup_mode", "initial_head")).strip().lower()
        if startup_mode not in {"initial_head", "confined_pre_solve"}:
            raise ValueError("device transient fast path supports startup modes 'initial_head' and 'confined_pre_solve'.")
        profile_fast_path_b = bool(controls.get("profile_transient_fast_path", False))
        adaptive_inner_config = _build_adaptive_inner_solve_config_from_controls(
            controls=controls,
            max_cycles=max_cycles_hard_i,
        )
        min_sat_f = float(min_sat)
        sy_f = float(sy)
        ss_f = float(ss)
        dx_f = float(self.dx)
        dt_f_val = float(dt_f)
        adaptive_dt_enabled_b = bool(controls.get("adaptive_dt_enabled", False))
        adaptive_dt_min_fraction_f = float(controls.get("adaptive_dt_min_fraction", 0.0625))
        adaptive_dt_shrink_factor_f = float(controls.get("adaptive_dt_shrink_factor", 0.5))
        adaptive_dt_grow_factor_f = float(controls.get("adaptive_dt_grow_factor", 2.0))
        adaptive_dt_strict_max_outer_i = int(controls.get("adaptive_dt_strict_max_outer", 20))
        adaptive_dt_max_growth_steps_i = int(controls.get("adaptive_dt_max_growth_steps", 2))
        adaptive_dt_early_shrink_enabled_b = bool(controls.get("adaptive_dt_early_shrink_enabled", True))
        adaptive_dt_early_shrink_min_outer_i = int(controls.get("adaptive_dt_early_shrink_min_outer", 6))
        adaptive_dt_early_shrink_patience_i = int(controls.get("adaptive_dt_early_shrink_patience", 3))
        adaptive_dt_extension_enabled_b = bool(controls.get("adaptive_dt_extension_enabled", True))
        adaptive_dt_extension_factor_f = float(controls.get("adaptive_dt_extension_factor", 5.0))
        adaptive_dt_extension_max_outer_i = int(controls.get("adaptive_dt_extension_max_outer", 4))
        adaptive_dt_extension_contraction_ratio_f = float(
            controls.get("adaptive_dt_extension_contraction_ratio", 0.8)
        )
        if adaptive_dt_enabled_b:
            if not (0.0 < adaptive_dt_min_fraction_f <= 1.0):
                raise ValueError("adaptive_dt_min_fraction must be in (0, 1].")
            if not (0.0 < adaptive_dt_shrink_factor_f < 1.0):
                raise ValueError("adaptive_dt_shrink_factor must be in (0, 1).")
            if adaptive_dt_grow_factor_f < 1.0:
                raise ValueError("adaptive_dt_grow_factor must be >= 1.")
            if adaptive_dt_strict_max_outer_i < 1:
                raise ValueError("adaptive_dt_strict_max_outer must be >= 1.")
            if adaptive_dt_strict_max_outer_i > max_outer:
                raise ValueError(
                    "adaptive_dt_strict_max_outer must be <= unconfined_max_picard_iter/max_outer."
                )
            if adaptive_dt_max_growth_steps_i < 0:
                raise ValueError("adaptive_dt_max_growth_steps must be >= 0.")
            if adaptive_dt_early_shrink_min_outer_i < 1:
                raise ValueError("adaptive_dt_early_shrink_min_outer must be >= 1.")
            if adaptive_dt_early_shrink_patience_i < 1:
                raise ValueError("adaptive_dt_early_shrink_patience must be >= 1.")
            if adaptive_dt_extension_factor_f < 1.0:
                raise ValueError("adaptive_dt_extension_factor must be >= 1.")
            if adaptive_dt_extension_max_outer_i < 1:
                raise ValueError("adaptive_dt_extension_max_outer must be >= 1.")
            if not (0.0 < adaptive_dt_extension_contraction_ratio_f < 1.0):
                raise ValueError("adaptive_dt_extension_contraction_ratio must be in (0, 1).")

        if inner_head_residual_tol_min_f < 0.0 or not np.isfinite(inner_head_residual_tol_min_f):
            raise ValueError("inner_head_residual_tol_min must be non-negative and finite.")
        if inner_head_residual_tol_max_f < inner_head_residual_tol_min_f:
            raise ValueError("inner_head_residual_tol_max must be >= inner_head_residual_tol_min.")
        if inner_picard_scale_max_fraction_f < 0.0 or inner_picard_scale_max_fraction_f > 1.0:
            raise ValueError("inner_picard_scale_max_fraction must be in [0, 1].")

        dim2d = (self.ny, self.nx)

        def _fast_path_phase_start() -> float:
            if profile_fast_path_b and str(device).startswith("cuda"):
                wp.synchronize_device(device)
            return time.perf_counter()

        def _fast_path_phase_elapsed(t_start: float) -> float:
            if profile_fast_path_b and str(device).startswith("cuda"):
                wp.synchronize_device(device)
            return float(time.perf_counter() - t_start)

        def _fast_path_head_residual_check() -> tuple[float, float, float, bool]:
            wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[flow_rTr_buf], device=device)
            wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[head_rTr_buf], device=device)
            wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[rhs_rTr_buf], device=device)
            wp.launch(kernel=zero_int_scalar_kernel, dim=1, inputs=[head_nonfinite_flag_buf], device=device)
            wp.launch(
                kernel=compute_dual_residual_kernel,
                dim=dim2d,
                inputs=[
                    h_iter_wp,
                    rhs_eff_wp,
                    self.T_wp,
                    self.active_wp,
                    self.bc_mask_wp,
                    self.mg_levels[0].gh_mask_wp,
                    self.mg_levels[0].ghb_factor_wp,
                    storage_diag_wp,
                    flow_rTr_buf,
                    head_rTr_buf,
                    self.nx,
                    self.ny,
                ],
                device=device,
            )
            wp.launch(
                kernel=detect_nonfinite_field_kernel,
                dim=dim2d,
                inputs=[h_iter_wp, head_nonfinite_flag_buf, self.nx, self.ny],
                device=device,
            )
            wp.launch(
                kernel=compute_active_rhs_l2_kernel,
                dim=dim2d,
                inputs=[rhs_eff_wp, self.active_wp, self.bc_mask_wp, rhs_rTr_buf, self.nx, self.ny],
                device=device,
            )
            counters["scalar_reductions"] += 1
            head_rtr = float(head_rTr_buf.numpy()[0])
            flow_rtr = float(flow_rTr_buf.numpy()[0])
            head_nonfinite = bool(int(head_nonfinite_flag_buf.numpy()[0]) != 0)
            head_rms = float(np.sqrt(max(head_rtr, 0.0) / float(max(n_free, 1))))
            flow_rms = float(np.sqrt(max(flow_rtr, 0.0) / float(max(n_free, 1))))
            rhs_rms = float(np.sqrt(max(float(rhs_rTr_buf.numpy()[0]), 0.0) / float(max(n_free, 1))))
            relative_flow_rms = flow_rms / max(rhs_rms, float(adaptive_inner_config.residual_floor))
            return head_rms, flow_rms, relative_flow_rms, head_nonfinite

        def evaluate_refreshed_nonlinear_candidate(
            *,
            outer_iteration: int,
            info_lin: dict[str, Any],
            dh_max: float,
            dh_rms: float,
            substep_dt: float,
            require_strict: bool = False,
        ) -> dict[str, Any]:
            """Refresh the nonlinear operator and evaluate authoritative acceptance."""
            wp.launch(kernel=copy_field_kernel, dim=dim2d, inputs=[storage_diag_wp, storage_diag_prev_wp, self.nx, self.ny], device=device)
            wp.launch(
                kernel=update_unconfined_transmissivity_from_head_kernel,
                dim=dim2d,
                inputs=[h_iter_wp, k_field_wp, bottom_wp, top_wp, self.active_wp, min_sat_f, self.nx, self.ny, self.T_wp],
                device=device,
            )
            wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[storage_change_sum_sq_buf], device=device)
            wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[storage_change_max_buf], device=device)
            wp.launch(
                kernel=update_secant_sy_storage_kernel,
                dim=dim2d,
                inputs=[
                    h_iter_wp, h_prev_wp, bottom_wp, top_wp, self.active_wp, self.bc_mask_wp,
                    sy_f, ss_f, dx_f, substep_dt, min_sat_f, 1.0e-12, self.nx, self.ny,
                    storage_coeff_wp, sy_coeff_wp, ss_coeff_wp, storage_diag_wp, storage_diag_prev_wp,
                    storage_change_sum_sq_buf, storage_change_max_buf,
                ],
                device=device,
            )
            wp.launch(
                kernel=build_transient_rhs_from_storage_kernel,
                dim=dim2d,
                inputs=[self.R_wp, storage_diag_wp, h_prev_wp, self.active_wp, self.bc_mask_wp, self.bc_values_wp, dx_f, self.nx, self.ny, rhs_eff_wp],
                device=device,
            )
            head_rms, flow_rms, relative_flow_rms, _ = _fast_path_head_residual_check()
            storage_change_rms = float(
                np.sqrt(max(float(storage_change_sum_sq_buf.numpy()[0]), 0.0) / float(max(n_free, 1)))
            )
            storage_change_max = float(storage_change_max_buf.numpy()[0])
            adaptive_used = bool(
                adaptive_inner_config.enabled and info_lin.get("adaptive_inner_controller_used", False)
            )
            inner_solved = _adaptive_practical_acceptance_allowed(
                practical_acceptance_enabled=True,
                adaptive_controller_used=adaptive_used,
                inner_target_achieved=bool(info_lin.get("inner_target_achieved", False)),
            )
            strict = bool(
                inner_solved and dh_max <= hclose and head_rms <= strict_head_residual_tol_f
            )
            practical = bool(
                (not require_strict)
                and _adaptive_practical_acceptance_allowed(
                    practical_acceptance_enabled=practical_picard_acceptance_enabled_b,
                    adaptive_controller_used=adaptive_used,
                    inner_target_achieved=bool(info_lin.get("inner_target_achieved", False)),
                )
                and int(outer_iteration) >= min_practical_outer_iterations_i
                and np.isfinite(head_rms) and head_rms <= practical_head_residual_tol_f
                and np.isfinite(dh_rms) and dh_rms <= practical_dh_rms_tol_f
                and np.isfinite(storage_change_rms)
                and storage_change_rms <= practical_storage_diag_change_rms_tol_f
            )
            return {
                "head_residual_rms": float(head_rms),
                "flow_residual_rms": float(flow_rms),
                "relative_flow_residual_rms": float(relative_flow_rms),
                "storage_diag_change_max": storage_change_max,
                "storage_diag_change_rms": storage_change_rms,
                "strict_acceptance_passed": strict,
                "practical_acceptance_passed": practical,
                "production_acceptance_passed": bool(strict or practical),
            }

        for period_index in range(n_periods):
            self.update_uniform_recharge_in_place(float(rates[period_index]))
            counters["R_device_updates"] += 1
            wp.launch(
                kernel=copy_field_kernel,
                dim=dim2d,
                inputs=[h_prev_wp, h_substep_start_wp, self.nx, self.ny],
                device=device,
            )
            if save_diagnostics_b:
                period_head_old = np.asarray(head_prev, dtype=np.float64).copy()

            period_t0 = time.perf_counter()
            T_update_seconds = 0.0
            storage_kernel_seconds = 0.0
            fine_m_inv_refresh_seconds = 0.0
            dynamic_coarse_refresh_seconds = 0.0
            rhs_assembly_seconds = 0.0
            storage_assembly_seconds = 0.0
            inner_solver_seconds = 0.0
            outer_convergence_check_seconds = 0.0
            final_nonlinear_residual_check_seconds = 0.0
            head_download_seconds = 0.0
            startup_inner_cycles = 0
            startup_converged = None

            if startup_mode == "confined_pre_solve":
                startup_t0 = _fast_path_phase_start()
                phase_t0 = _fast_path_phase_start()
                wp.launch(
                    kernel=update_unconfined_transmissivity_from_head_kernel,
                    dim=dim2d,
                    inputs=[h_iter_wp, k_field_wp, bottom_wp, top_wp, self.active_wp, min_sat_f, self.nx, self.ny, self.T_wp],
                    device=device,
                )
                T_update_seconds += _fast_path_phase_elapsed(phase_t0)
                counters["T_device_updates"] += 1
                phase_t0 = _fast_path_phase_start()
                wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[storage_change_sum_sq_buf], device=device)
                wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[storage_change_max_buf], device=device)
                wp.launch(
                    kernel=update_secant_sy_storage_kernel,
                    dim=dim2d,
                    inputs=[
                        h_iter_wp, h_prev_wp, bottom_wp, top_wp, self.active_wp, self.bc_mask_wp,
                        sy_f, ss_f, dx_f, dt_f_val, min_sat_f, 1.0e-12, self.nx, self.ny,
                        storage_coeff_wp, sy_coeff_wp, ss_coeff_wp, storage_diag_wp, storage_diag_prev_wp,
                        storage_change_sum_sq_buf, storage_change_max_buf,
                    ],
                    device=device,
                )
                storage_kernel_seconds += _fast_path_phase_elapsed(phase_t0)
                counters["storage_device_updates"] += 1
                if hasattr(self, "_update_diag_preconditioner_device"):
                    phase_t0 = _fast_path_phase_start()
                    self._update_diag_preconditioner_device(
                        T_wp=self.T_wp,
                        active_wp=self.active_wp,
                        bc_mask_wp=self.bc_mask_wp,
                        gh_mask_wp=self.mg_levels[0].gh_mask_wp,
                        ghb_factor_wp=self.mg_levels[0].ghb_factor_wp,
                        M_inv_wp=self.mg_levels[0].M_inv_wp,
                        nx=self.nx,
                        ny=self.ny,
                        use_ghb=bool(self.use_ghb),
                        storage_diag_wp=storage_diag_wp,
                    )
                    fine_m_inv_refresh_seconds += _fast_path_phase_elapsed(phase_t0)
                phase_t0 = _fast_path_phase_start()
                self._refresh_transient_device_hierarchy_values(levels=self.mg_levels)
                dynamic_coarse_refresh_seconds += _fast_path_phase_elapsed(phase_t0)
                counters["hierarchy_device_coarse_value_refreshes"] += 1
                counters["rhs_device_updates"] += 1
                phase_t0 = _fast_path_phase_start()
                wp.launch(
                    kernel=build_transient_rhs_from_storage_kernel,
                    dim=dim2d,
                    inputs=[
                        self.R_wp,
                        storage_diag_wp,
                        h_prev_wp,
                        self.active_wp,
                        self.bc_mask_wp,
                        self.bc_values_wp,
                        dx_f,
                        self.nx,
                        self.ny,
                        rhs_eff_wp,
                    ],
                    device=device,
                )
                rhs_assembly_seconds += _fast_path_phase_elapsed(phase_t0)
                startup_controls = dict(controls)
                startup_controls["max_cycles"] = int(controls.get("max_cycles", 200))
                startup_controls["coarse_operator_mode"] = fast_path_coarse_operator_mode
                phase_t0 = _fast_path_phase_start()
                startup_info = self._solve_multigrid_kcycle_device_buffers(
                    x_wp=h_iter_wp,
                    rhs_wp=rhs_eff_wp,
                    T_wp=self.T_wp,
                    storage_diag_wp=storage_diag_wp,
                    active_wp=self.active_wp,
                    bc_mask_wp=self.bc_mask_wp,
                    bc_values_wp=self.bc_values_wp,
                    levels=self.mg_levels,
                    solve_controls=startup_controls,
                    return_scalar_info=True,
                )
                inner_solver_seconds += _fast_path_phase_elapsed(phase_t0)
                startup_inner_cycles = int(startup_info.get("n_cycles_used", 0) or 0)
                startup_converged = bool(startup_info.get("converged", False))
                phase_t0 = _fast_path_phase_start()
                wp.launch(
                    kernel=clamp_unconfined_head_kernel,
                    dim=dim2d,
                    inputs=[
                        h_iter_wp,
                        bottom_wp,
                        top_wp,
                        self.active_wp,
                        self.bc_mask_wp,
                        self.bc_values_wp,
                        min_sat_f,
                        self.nx,
                        self.ny,
                    ],
                    device=device,
                )
                storage_diag_prev_wp.fill_(WP_FLOAT(0.0))
                inner_solver_seconds += _fast_path_phase_elapsed(phase_t0)
            last_dh_max = float("nan")
            last_dh_rms = float("nan")
            last_flow_residual_rms = float("nan")
            last_head_residual_rms = float("nan")
            last_storage_diag_change_max = float("nan")
            last_storage_diag_change_rms = float("nan")
            strict_picard_convergence_passed = False
            practical_picard_acceptance_passed = False
            production_acceptance_passed = False
            previous_dh_measure = None
            previous_outer_head_residual_rms = None
            previous_initial_head_residual_rms = None
            previous_outer_dh_rms = None
            total_inner_kcycles = 0
            maximum_inner_kcycles_in_one_outer_iteration = 0
            inner_kcycle_caps: list[int] = []
            inner_kcycle_used: list[int] = []
            inner_block_counts: list[int] = []
            inner_residual_check_count = 0
            adaptive_target_achievement_count = 0
            legacy_dh_fallback_count = 0
            stalled_inner_solve_count = 0
            divergent_inner_solve_count = 0
            rolled_back_block_count = 0
            outer_iteration_summaries: list[dict[str, Any]] = []
            period_gpu_scalar_syncs = 0
            info_lin = {
                "converged": False,
                "coarse_operator_mode": fast_path_coarse_operator_mode,
                "fine_operator_residual_checked": True,
            }

            period_dt_f = dt_f_val
            remaining_dt_f = period_dt_f
            current_dt_f = period_dt_f
            actual_dt_f = period_dt_f
            dt_min_f = period_dt_f * adaptive_dt_min_fraction_f
            adaptive_dt_growth_steps_i = 0
            adaptive_dt_substep_dts: list[float] = []
            adaptive_dt_retry_count = 0
            adaptive_dt_practical_fallback_count = 0
            adaptive_dt_total_outer_iterations_i = 0
            adaptive_dt_practical_at_min_b = not adaptive_dt_enabled_b
            adaptive_dt_dh_history: list[float] = []
            adaptive_dt_extension_used_b = False
            adaptive_dt_early_shrink_streak_i = 0
            adaptive_dt_early_shrink_count = 0
            adaptive_dt_extension_count = 0
            substep_outer_limit_i = (
                adaptive_dt_strict_max_outer_i if adaptive_dt_enabled_b else max_outer
            )
            outer_iter = 0
            while outer_iter < substep_outer_limit_i:
                adaptive_dt_total_outer_iterations_i += 1
                storage_t0 = _fast_path_phase_start()
                phase_t0 = _fast_path_phase_start()
                wp.launch(
                    kernel=update_unconfined_transmissivity_from_head_kernel,
                    dim=dim2d,
                    inputs=[h_iter_wp, k_field_wp, bottom_wp, top_wp, self.active_wp, min_sat_f, self.nx, self.ny, self.T_wp],
                    device=device
                )
                T_update_seconds += _fast_path_phase_elapsed(phase_t0)
                counters["T_device_updates"] += 1

                phase_t0 = _fast_path_phase_start()
                wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[storage_change_sum_sq_buf], device=device)
                wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[storage_change_max_buf], device=device)

                wp.launch(
                    kernel=update_secant_sy_storage_kernel,
                    dim=dim2d,
                    inputs=[
                        h_iter_wp, h_prev_wp, bottom_wp, top_wp, self.active_wp, self.bc_mask_wp,
                        sy_f, ss_f, dx_f, actual_dt_f, min_sat_f, 1.0e-12, self.nx, self.ny,
                        storage_coeff_wp, sy_coeff_wp, ss_coeff_wp, storage_diag_wp, storage_diag_prev_wp,
                        storage_change_sum_sq_buf, storage_change_max_buf
                    ],
                    device=device
                )
                wp.launch(kernel=copy_field_kernel, dim=dim2d, inputs=[storage_diag_wp, storage_diag_prev_wp, self.nx, self.ny], device=device)
                storage_kernel_seconds += _fast_path_phase_elapsed(phase_t0)
                counters["storage_device_updates"] += 1
                storage_assembly_seconds += _fast_path_phase_elapsed(storage_t0)

                if hasattr(self, "_update_diag_preconditioner_device"):
                    phase_t0 = _fast_path_phase_start()
                    self._update_diag_preconditioner_device(
                        T_wp=self.T_wp,
                        active_wp=self.active_wp,
                        bc_mask_wp=self.bc_mask_wp,
                        gh_mask_wp=self.mg_levels[0].gh_mask_wp,
                        ghb_factor_wp=self.mg_levels[0].ghb_factor_wp,
                        M_inv_wp=self.mg_levels[0].M_inv_wp,
                        nx=self.nx,
                        ny=self.ny,
                        use_ghb=bool(self.use_ghb),
                        storage_diag_wp=storage_diag_wp
                    )
                    fine_m_inv_refresh_seconds += _fast_path_phase_elapsed(phase_t0)
                phase_t0 = _fast_path_phase_start()
                self._refresh_transient_device_hierarchy_values(levels=self.mg_levels)
                dynamic_coarse_refresh_seconds += _fast_path_phase_elapsed(phase_t0)
                counters["hierarchy_device_coarse_value_refreshes"] += 1

                rhs_t0 = _fast_path_phase_start()
                counters["rhs_device_updates"] += 1
                wp.launch(
                    kernel=build_transient_rhs_from_storage_kernel,
                    dim=dim2d,
                    inputs=[
                        self.R_wp,
                        storage_diag_wp,
                        h_prev_wp,
                        self.active_wp,
                        self.bc_mask_wp,
                        self.bc_values_wp,
                        dx_f,
                        self.nx,
                        self.ny,
                        rhs_eff_wp,
                    ],
                    device=device
                )
                rhs_assembly_seconds += _fast_path_phase_elapsed(rhs_t0)

                wp.launch(kernel=copy_field_kernel, dim=dim2d, inputs=[h_iter_wp, h_snapshot_wp, self.nx, self.ny], device=device)
                # Incremental Picard: materialise the nonlinear residual field
                # r^k = b - A*h^k (h_snapshot == h^k here) and reset the correction
                # to zero, so the inner solve targets A*delta = r^k with delta=0 on
                # Dirichlet cells. rhs_rTr_buf is reused as an unread scratch for the
                # kernel's rTr reduction; it is re-zeroed before any later read.
                if use_incremental_picard:
                    wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[rhs_rTr_buf], device=device)
                    wp.launch(
                        kernel=compute_residual_kernel,
                        dim=dim2d,
                        inputs=[
                            h_snapshot_wp,
                            rhs_eff_wp,
                            self.T_wp,
                            self.active_wp,
                            self.bc_mask_wp,
                            self.mg_levels[0].gh_mask_wp,
                            self.mg_levels[0].ghb_factor_wp,
                            storage_diag_wp,
                            residual_wp,
                            rhs_rTr_buf,
                            self.nx,
                            self.ny,
                        ],
                        device=device,
                    )
                    wp.launch(kernel=zero_field_kernel, dim=dim2d, inputs=[delta_wp, self.nx, self.ny], device=device)
                adaptive_fallback_reason = ""
                adaptive_controller_used = bool(adaptive_inner_config.enabled)
                legacy_dh_fallback_used = False
                forcing_eta_used = float("nan")
                inner_initial_head_residual_rms = float("nan")
                inner_initial_flow_residual_rms = float("nan")
                inner_initial_relative_flow_residual_rms = float("nan")
                inner_target_head_residual_rms = float("nan")
                inner_target_relative_flow_residual_rms = float("nan")
                inner_final_head_residual_rms = float("nan")
                adaptive_state: AdaptiveInnerSolveState | None = None
                adaptive_pre_fallback_cycles = 0
                adaptive_pre_fallback_blocks = 0

                if adaptive_inner_config.enabled:
                    residual_check_t0 = _fast_path_phase_start()
                    (
                        initial_head_residual_rms,
                        initial_flow_residual_rms,
                        initial_relative_flow_residual_rms,
                        initial_head_nonfinite,
                    ) = _fast_path_head_residual_check()
                    inner_solver_seconds += _fast_path_phase_elapsed(residual_check_t0)
                    inner_residual_check_count += 1
                    inner_initial_head_residual_rms = float(initial_head_residual_rms)
                    inner_initial_flow_residual_rms = float(initial_flow_residual_rms)
                    inner_initial_relative_flow_residual_rms = float(
                        initial_relative_flow_residual_rms
                    )

                    if initial_head_nonfinite or not np.isfinite(initial_head_residual_rms):
                        adaptive_fallback_reason = "nonfinite_initial_head_residual"
                        adaptive_controller_used = False
                        legacy_dh_fallback_used = True
                    else:
                        forcing_eta_used = _compute_inner_forcing_eta(
                            current_outer_residual_rms=initial_head_residual_rms,
                            previous_outer_residual_rms=previous_initial_head_residual_rms,
                            config=adaptive_inner_config,
                        )
                        inner_target_head_residual_rms = _compute_inner_target_residual(
                            initial_residual_rms=initial_head_residual_rms,
                            forcing_eta=forcing_eta_used,
                            residual_floor=float(adaptive_inner_config.residual_floor),
                            inner_head_residual_tol_min=inner_head_residual_tol_min_f,
                            inner_head_residual_tol_max=inner_head_residual_tol_max_f,
                            inner_picard_scale_max_fraction=inner_picard_scale_max_fraction_f,
                            previous_outer_dh_rms=previous_outer_dh_rms,
                            hclose=hclose,
                        )
                        inner_target_relative_flow_residual_rms = max(
                            float(adaptive_inner_config.residual_floor),
                            float(forcing_eta_used) * float(initial_flow_residual_rms),
                        )

                        def _run_adaptive_block(block_cycles: int) -> dict[str, Any]:
                            nonlocal inner_solver_seconds
                            if use_incremental_picard:
                                # Snapshot the running correction so a divergent
                                # block can be rolled back; continue delta in place.
                                wp.launch(
                                    kernel=copy_field_kernel,
                                    dim=dim2d,
                                    inputs=[delta_wp, delta_snapshot_wp, self.nx, self.ny],
                                    device=device,
                                )
                                block_controls = dict(controls)
                                block_controls["max_cycles"] = int(block_cycles)
                                block_controls["check_every_no"] = int(block_cycles)
                                block_controls["coarse_operator_mode"] = fast_path_coarse_operator_mode
                                block_t0 = _fast_path_phase_start()
                                block_info = self._solve_multigrid_kcycle_device_buffers(
                                    x_wp=delta_wp,
                                    rhs_wp=residual_wp,
                                    T_wp=self.T_wp,
                                    storage_diag_wp=storage_diag_wp,
                                    active_wp=self.active_wp,
                                    bc_mask_wp=self.bc_mask_wp,
                                    bc_values_wp=zero_bc_values_wp,
                                    levels=self.mg_levels,
                                    solve_controls=block_controls,
                                    return_scalar_info=False,
                                )
                                inner_solver_seconds += _fast_path_phase_elapsed(block_t0)
                                # Sync h_iter = h^k + delta so the (unchanged) residual
                                # check measures ||b - A*(h^k + delta)|| = ||r^k - A*delta||.
                                wp.launch(
                                    kernel=apply_relaxed_correction_kernel,
                                    dim=dim2d,
                                    inputs=[
                                        h_snapshot_wp,
                                        delta_wp,
                                        self.active_wp,
                                        self.bc_mask_wp,
                                        self.bc_values_wp,
                                        WP_FLOAT(1.0),
                                        WP_FLOAT(incremental_picard_sync_max_change),
                                        self.nx,
                                        self.ny,
                                        h_iter_wp,
                                    ],
                                    device=device,
                                )
                                residual_t0 = _fast_path_phase_start()
                                head_rms_after, flow_rms_after, _, head_nonfinite_after = _fast_path_head_residual_check()
                                inner_solver_seconds += _fast_path_phase_elapsed(residual_t0)
                                return {
                                    "actual_cycles": int(
                                        block_info["n_cycles_used"]
                                        if block_info.get("n_cycles_used") is not None else block_cycles
                                    ),
                                    "residual_after_rms": float(head_rms_after),
                                    "relative_flow_residual_rms": float(flow_rms_after),
                                    "rollback_required": bool(
                                        head_nonfinite_after or (not np.isfinite(head_rms_after))
                                    ),
                                    "head_nonfinite": bool(head_nonfinite_after),
                                    "numerical_breakdown": False,
                                }
                            wp.launch(
                                kernel=copy_field_kernel,
                                dim=dim2d,
                                inputs=[h_iter_wp, h_inner_snapshot_wp, self.nx, self.ny],
                                device=device,
                            )
                            block_controls = dict(controls)
                            block_controls["max_cycles"] = int(block_cycles)
                            block_controls["check_every_no"] = int(block_cycles)
                            block_controls["coarse_operator_mode"] = fast_path_coarse_operator_mode
                            block_t0 = _fast_path_phase_start()
                            block_info = self._solve_multigrid_kcycle_device_buffers(
                                x_wp=h_iter_wp,
                                rhs_wp=rhs_eff_wp,
                                T_wp=self.T_wp,
                                storage_diag_wp=storage_diag_wp,
                                active_wp=self.active_wp,
                                bc_mask_wp=self.bc_mask_wp,
                                bc_values_wp=self.bc_values_wp,
                                levels=self.mg_levels,
                                solve_controls=block_controls,
                                return_scalar_info=False,
                            )
                            inner_solver_seconds += _fast_path_phase_elapsed(block_t0)
                            residual_t0 = _fast_path_phase_start()
                            head_rms_after, flow_rms_after, _, head_nonfinite_after = _fast_path_head_residual_check()
                            inner_solver_seconds += _fast_path_phase_elapsed(residual_t0)
                            return {
                                "actual_cycles": int(
                                    block_info["n_cycles_used"]
                                    if block_info.get("n_cycles_used") is not None else block_cycles
                                ),
                                "residual_after_rms": float(head_rms_after),
                                "relative_flow_residual_rms": float(flow_rms_after),
                                "rollback_required": bool(
                                    head_nonfinite_after or (not np.isfinite(head_rms_after))
                                ),
                                "head_nonfinite": bool(head_nonfinite_after),
                                "numerical_breakdown": False,
                            }

                        def _rollback_adaptive_block() -> None:
                            if use_incremental_picard:
                                wp.launch(
                                    kernel=copy_field_kernel,
                                    dim=dim2d,
                                    inputs=[delta_snapshot_wp, delta_wp, self.nx, self.ny],
                                    device=device,
                                )
                                return
                            wp.launch(
                                kernel=copy_field_kernel,
                                dim=dim2d,
                                inputs=[h_inner_snapshot_wp, h_iter_wp, self.nx, self.ny],
                                device=device,
                            )

                        adaptive_state = _run_adaptive_inner_kcycle_blocks(
                            initial_residual_rms=initial_head_residual_rms,
                            target_residual_rms=inner_target_head_residual_rms,
                            forcing_eta=forcing_eta_used,
                            previous_outer_residual_rms=previous_outer_head_residual_rms,
                            previous_outer_dh_rms=previous_outer_dh_rms,
                            max_cycles=max_cycles_hard_i,
                            config=adaptive_inner_config,
                            run_block=_run_adaptive_block,
                            rollback_block=_rollback_adaptive_block,
                            initial_relative_flow_residual_rms=initial_flow_residual_rms,
                            target_relative_flow_residual_rms=inner_target_relative_flow_residual_rms,
                        )
                        previous_initial_head_residual_rms = float(initial_head_residual_rms)
                        inner_residual_check_count += int(adaptive_state.residual_check_count)
                        inner_cycles_used_i = int(adaptive_state.total_cycles)
                        adaptive_pre_fallback_cycles = int(adaptive_state.total_cycles)
                        adaptive_pre_fallback_blocks = int(adaptive_state.block_index)
                        inner_block_counts.append(int(adaptive_state.block_index))
                        if adaptive_state.target_achieved:
                            adaptive_target_achievement_count += 1
                        if adaptive_state.stalled:
                            stalled_inner_solve_count += 1
                        if adaptive_state.diverged:
                            divergent_inner_solve_count += 1
                        if adaptive_state.rollback_count:
                            rolled_back_block_count += int(adaptive_state.rollback_count)
                        inner_final_head_residual_rms = float(adaptive_state.final_residual_rms)
                        info_lin = {
                            "converged": bool(adaptive_state.converged),
                            "n_cycles_used": int(adaptive_state.total_cycles),
                            "h_rms_end": (
                                float(adaptive_state.final_residual_rms)
                                if np.isfinite(float(adaptive_state.final_residual_rms))
                                else None
                            ),
                            "adaptive_inner_residual_check_count": int(adaptive_state.residual_check_count),
                            "coarse_operator_mode": fast_path_coarse_operator_mode,
                            "fine_operator_residual_checked": True,
                            "adaptive_inner_controller_enabled": True,
                            "adaptive_inner_controller_used": True,
                            "adaptive_inner_fallback_to_legacy_dh": False,
                            "adaptive_inner_fallback_reason": "",
                            "inner_target_achieved": bool(adaptive_state.target_achieved),
                            "inner_usable_for_picard": bool(adaptive_state.usable_for_picard),
                            "inner_stalled": bool(adaptive_state.stalled),
                            "inner_diverged": bool(adaptive_state.diverged),
                            "inner_rollback_count": int(adaptive_state.rollback_count),
                            "inner_termination_reason": str(adaptive_state.termination_reason),
                            "initial_head_residual_rms": float(initial_head_residual_rms),
                            "initial_relative_flow_residual_rms": float(initial_relative_flow_residual_rms),
                            "initial_flow_residual_rms": float(initial_flow_residual_rms),
                            "target_head_residual_rms": float(inner_target_head_residual_rms),
                            "target_relative_flow_residual_rms": float(inner_target_relative_flow_residual_rms),
                            "final_flow_residual_rms": float(adaptive_state.final_relative_flow_residual_rms),
                            "head_reduction_ratio": (
                                float(adaptive_state.final_residual_rms)
                                / max(float(adaptive_state.initial_residual_rms), float(adaptive_inner_config.residual_floor))
                            ),
                            "flow_reduction_ratio": (
                                float(adaptive_state.final_relative_flow_residual_rms)
                                / max(float(adaptive_state.initial_relative_flow_residual_rms), float(adaptive_inner_config.residual_floor))
                            ),
                            "head_q": list(adaptive_state.head_per_cycle_convergence_factors),
                            "flow_q": list(adaptive_state.flow_per_cycle_convergence_factors),
                            "controller_q": list(adaptive_state.controller_per_cycle_convergence_factors),
                            "head_target_gap": (
                                float(adaptive_state.final_residual_rms)
                                / max(float(inner_target_head_residual_rms), float(adaptive_inner_config.residual_floor))
                            ),
                            "flow_target_gap": (
                                float(adaptive_state.final_relative_flow_residual_rms)
                                / max(float(inner_target_relative_flow_residual_rms), float(adaptive_inner_config.residual_floor))
                            ),
                            "controller_target_gap": max(
                                float(adaptive_state.final_residual_rms)
                                / max(float(inner_target_head_residual_rms), float(adaptive_inner_config.residual_floor)),
                                float(adaptive_state.final_relative_flow_residual_rms)
                                / max(float(inner_target_relative_flow_residual_rms), float(adaptive_inner_config.residual_floor)),
                            ),
                            "final_head_residual_rms": (
                                float(adaptive_state.final_residual_rms)
                                if np.isfinite(float(adaptive_state.final_residual_rms))
                                else None
                            ),
                            "forcing_eta": float(forcing_eta_used),
                            "controller_mode": "adaptive_residual_blocks",
                            "inner_block_count": int(adaptive_state.block_index),
                            "inner_cycles_per_block": list(adaptive_state.cycles_per_block),
                        }
                        if adaptive_inner_config.save_block_history:
                            info_lin.update(
                                {
                                    "inner_cycles_per_block": list(adaptive_state.cycles_per_block),
                                    "inner_residuals_per_block": list(adaptive_state.residuals_per_block),
                                    "inner_contraction_ratios": list(adaptive_state.contraction_ratios),
                                    "inner_per_cycle_convergence_factors": list(
                                        adaptive_state.per_cycle_convergence_factors
                                    ),
                                    "inner_predicted_cycles_per_block": list(
                                        adaptive_state.predicted_cycles_per_block
                                    ),
                                }
                            )

                        if not adaptive_state.target_achieved:
                            adaptive_fallback_reason = (
                                adaptive_state.fallback_reason or adaptive_state.termination_reason
                            )
                            adaptive_controller_used = False
                            legacy_dh_fallback_used = True

                if not adaptive_controller_used:
                    legacy_dh_fallback_count += 1 if legacy_dh_fallback_used else 0
                    inner_max_cycles_i = _select_legacy_unconfined_inner_max_cycles_from_dh(
                        previous_dh_measure=previous_dh_measure,
                        early_cycles=inner_max_cycles_early_i,
                        middle_cycles=inner_max_cycles_middle_i,
                        late_cycles=inner_max_cycles_late_i,
                        middle_dh=inner_middle_dh_f,
                        late_dh=inner_late_dh_f,
                    )
                    legacy_cycles_requested_i = int(inner_max_cycles_i)
                    inner_max_cycles_i = _remaining_legacy_fallback_cycles(
                        max_cycles=max_cycles_hard_i,
                        adaptive_cycles_used=adaptive_pre_fallback_cycles,
                        selected_legacy_cycles=inner_max_cycles_i,
                    )
                    inner_controls = dict(controls)
                    inner_controls["max_cycles"] = int(inner_max_cycles_i)
                    inner_controls["coarse_operator_mode"] = fast_path_coarse_operator_mode
                    inner_kcycle_caps.append(int(inner_max_cycles_i))

                    if inner_max_cycles_i > 0:
                        inner_t0 = _fast_path_phase_start()
                        if use_incremental_picard:
                            _legacy_x_wp = delta_wp
                            _legacy_rhs_wp = residual_wp
                            _legacy_bc_values_wp = zero_bc_values_wp
                        else:
                            _legacy_x_wp = h_iter_wp
                            _legacy_rhs_wp = rhs_eff_wp
                            _legacy_bc_values_wp = self.bc_values_wp
                        info_lin = self._solve_multigrid_kcycle_device_buffers(
                            x_wp=_legacy_x_wp,
                            rhs_wp=_legacy_rhs_wp,
                            T_wp=self.T_wp,
                            storage_diag_wp=storage_diag_wp,
                            active_wp=self.active_wp,
                            bc_mask_wp=self.bc_mask_wp,
                            bc_values_wp=_legacy_bc_values_wp,
                            levels=self.mg_levels,
                            solve_controls=inner_controls,
                            return_scalar_info=False
                        )
                        inner_solver_seconds += _fast_path_phase_elapsed(inner_t0)
                        inner_cycles_used_i = int(
                            info_lin["n_cycles_used"]
                            if info_lin.get("n_cycles_used") is not None else inner_max_cycles_i
                        )
                    else:
                        info_lin = {
                            "n_cycles_used": 0,
                            "converged": False,
                            "inner_termination_reason": "hard_cycle_ceiling_before_legacy_fallback",
                        }
                        inner_cycles_used_i = 0
                    legacy_cycles_used_i = int(inner_cycles_used_i)
                    if adaptive_pre_fallback_blocks > 0 and inner_block_counts:
                        inner_block_counts.pop()
                    inner_cycles_used_i += int(adaptive_pre_fallback_cycles)
                    inner_block_counts.append(int(1 + adaptive_pre_fallback_blocks))
                    inner_final_head_residual_rms = float("nan")
                    info_lin.update(
                        {
                            "adaptive_inner_controller_enabled": bool(adaptive_inner_config.enabled),
                            "adaptive_inner_controller_used": False,
                            "adaptive_inner_fallback_to_legacy_dh": bool(legacy_dh_fallback_used),
                            "adaptive_inner_fallback_reason": str(adaptive_fallback_reason),
                            "controller_mode": "legacy_dh_schedule",
                            "inner_block_count": int(1 + adaptive_pre_fallback_blocks),
                            "inner_cycles_per_block": (
                                list(adaptive_state.cycles_per_block)
                                if adaptive_state is not None else []
                            ) + [legacy_cycles_used_i],
                            "adaptive_cycles_before_fallback": int(adaptive_pre_fallback_cycles),
                            "legacy_fallback_cycles": int(legacy_cycles_used_i),
                            "legacy_fallback_cycles_requested": int(legacy_cycles_requested_i),
                            "inner_termination_reason": (
                                "legacy_fixed_cycle_cap"
                                if inner_max_cycles_i > 0
                                else "hard_cycle_ceiling_before_legacy_fallback"
                            ),
                            "initial_head_residual_rms": (
                                float(inner_initial_head_residual_rms)
                                if np.isfinite(inner_initial_head_residual_rms)
                                else None
                            ),
                            "target_head_residual_rms": (
                                float(inner_target_head_residual_rms)
                                if np.isfinite(inner_target_head_residual_rms)
                                else None
                            ),
                            "final_head_residual_rms": (
                                float(inner_final_head_residual_rms)
                                if np.isfinite(inner_final_head_residual_rms)
                                else None
                            ),
                            "forcing_eta": float(forcing_eta_used) if np.isfinite(forcing_eta_used) else None,
                        }
                    )
                else:
                    inner_kcycle_caps.append(int(max_cycles_hard_i))
                    inner_cycles_used_i = int(info_lin.get("n_cycles_used", 0))

                inner_kcycle_used.append(inner_cycles_used_i)
                total_inner_kcycles += inner_cycles_used_i
                maximum_inner_kcycles_in_one_outer_iteration = max(
                    maximum_inner_kcycles_in_one_outer_iteration,
                    inner_cycles_used_i,
                )
                if use_incremental_picard:
                    wp.launch(
                        kernel=apply_relaxed_correction_kernel,
                        dim=dim2d,
                        inputs=[
                            h_snapshot_wp,
                            delta_wp,
                            self.active_wp,
                            self.bc_mask_wp,
                            self.bc_values_wp,
                            WP_FLOAT(omega_current_f),
                            WP_FLOAT(max_update_f),
                            self.nx,
                            self.ny,
                            h_iter_wp,
                        ],
                        device=device,
                    )
                else:
                    wp.launch(
                        kernel=apply_relaxed_clipped_picard_update_kernel,
                        dim=dim2d,
                        inputs=[
                            h_iter_wp,
                            h_snapshot_wp,
                            self.active_wp,
                            self.bc_mask_wp,
                            self.bc_values_wp,
                            WP_FLOAT(omega_current_f),
                            WP_FLOAT(max_update_f),
                            self.nx,
                            self.ny,
                            h_iter_wp,
                        ],
                        device=device,
                    )

                outer_check_t0 = _fast_path_phase_start()
                wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[dh_rms_buf], device=device)
                wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[dh_max_buf], device=device)
                wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[flow_rTr_buf], device=device)
                wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[head_rTr_buf], device=device)
                wp.launch(
                    kernel=kcycle_check_dh_and_dual_residual_kernel,
                    dim=dim2d,
                    inputs=[
                        h_iter_wp,
                        h_snapshot_wp,
                        rhs_eff_wp,
                        self.T_wp,
                        self.active_wp,
                        self.bc_mask_wp,
                        self.mg_levels[0].gh_mask_wp,
                        self.mg_levels[0].ghb_factor_wp,
                        storage_diag_wp,
                        dh_rms_buf,
                        dh_max_buf,
                        flow_rTr_buf,
                        head_rTr_buf,
                        int(1 if self.use_ghb else 0),
                        self.nx,
                        self.ny,
                    ],
                    device=device
                )
                counters["scalar_reductions"] += 1
                counters["gpu_scalar_synchronizations"] += 6
                period_gpu_scalar_syncs += 6
                last_dh_max = float(dh_max_buf.numpy()[0])
                last_dh_rms = float(np.sqrt(max(float(dh_rms_buf.numpy()[0]), 0.0) / float(max(n_free, 1))))
                if adaptive_dt_enabled_b:
                    adaptive_dt_dh_history.append(float(last_dh_max))
                last_flow_residual_rms = float(
                    np.sqrt(max(float(flow_rTr_buf.numpy()[0]), 0.0) / float(max(n_free, 1)))
                )
                last_head_residual_rms = float(
                    np.sqrt(max(float(head_rTr_buf.numpy()[0]), 0.0) / float(max(n_free, 1)))
                )
                last_storage_diag_change_max = float(storage_change_max_buf.numpy()[0])
                last_storage_diag_change_rms = float(
                    np.sqrt(max(float(storage_change_sum_sq_buf.numpy()[0]), 0.0) / float(max(n_free, 1)))
                )
                outer_convergence_check_seconds += _fast_path_phase_elapsed(outer_check_t0)
                previous_outer_head_residual_rms_before = previous_outer_head_residual_rms
                previous_outer_dh_rms_before = previous_outer_dh_rms
                previous_dh_measure = float(last_dh_max)
                previous_outer_head_residual_rms = float(last_head_residual_rms)
                previous_outer_dh_rms = float(last_dh_rms)

                adaptive_final_linearisation = bool(
                    adaptive_inner_config.enabled
                    and info_lin.get("adaptive_inner_controller_used", False)
                )
                final_linearisation_solved = _adaptive_practical_acceptance_allowed(
                    practical_acceptance_enabled=True,
                    adaptive_controller_used=adaptive_final_linearisation,
                    inner_target_achieved=bool(info_lin.get("inner_target_achieved", False)),
                )
                strict_picard_convergence_passed = bool(
                    final_linearisation_solved
                    and last_dh_max <= hclose
                    and last_head_residual_rms <= strict_head_residual_tol_f
                )
                practical_picard_acceptance_passed = bool(
                    _adaptive_practical_acceptance_allowed(
                        practical_acceptance_enabled=practical_picard_acceptance_enabled_b,
                        adaptive_controller_used=adaptive_final_linearisation,
                        inner_target_achieved=bool(info_lin.get("inner_target_achieved", False)),
                    )
                    and int(outer_iter + 1) >= min_practical_outer_iterations_i
                    and np.isfinite(last_head_residual_rms)
                    and last_head_residual_rms <= practical_head_residual_tol_f
                    and np.isfinite(last_dh_rms)
                    and last_dh_rms <= practical_dh_rms_tol_f
                    and np.isfinite(last_storage_diag_change_rms)
                    and last_storage_diag_change_rms <= practical_storage_diag_change_rms_tol_f
                )
                production_acceptance_passed = bool(
                    strict_picard_convergence_passed or practical_picard_acceptance_passed
                )
                outer_summary = {
                    "outer_iteration": int(outer_iter + 1),
                    "controller_mode": str(info_lin.get("controller_mode", "legacy_dh_schedule")),
                    "initial_head_residual_rms": (
                        float(inner_initial_head_residual_rms)
                        if np.isfinite(inner_initial_head_residual_rms)
                        else None
                    ),
                    "target_head_residual_rms": (
                        float(inner_target_head_residual_rms)
                        if np.isfinite(inner_target_head_residual_rms)
                        else None
                    ),
                    "initial_relative_flow_residual_rms": (
                        float(inner_initial_relative_flow_residual_rms)
                        if np.isfinite(inner_initial_relative_flow_residual_rms) else None
                    ),
                    "initial_flow_residual_rms": (
                        float(inner_initial_flow_residual_rms)
                        if np.isfinite(inner_initial_flow_residual_rms) else None
                    ),
                    "target_flow_residual_rms": (
                        float(inner_target_relative_flow_residual_rms)
                        if np.isfinite(inner_target_relative_flow_residual_rms) else None
                    ),
                    "final_flow_residual_rms": info_lin.get("final_flow_residual_rms"),
                    "target_relative_flow_residual_rms": (
                        float(inner_target_relative_flow_residual_rms)
                        if np.isfinite(inner_target_relative_flow_residual_rms) else None
                    ),
                    "final_head_residual_rms": float(last_head_residual_rms),
                    "final_max_abs_head_change": float(last_dh_max),
                    "final_rms_head_change": float(last_dh_rms),
                    "final_relative_flow_residual_rms": info_lin.get(
                        "final_relative_flow_residual_rms"
                    ),
                    "head_reduction_ratio": info_lin.get("head_reduction_ratio"),
                    "flow_reduction_ratio": info_lin.get("flow_reduction_ratio"),
                    "head_q": list(info_lin.get("head_q", [])),
                    "flow_q": list(info_lin.get("flow_q", [])),
                    "controller_q": list(info_lin.get("controller_q", [])),
                    "head_target_gap": info_lin.get("head_target_gap"),
                    "flow_target_gap": info_lin.get("flow_target_gap"),
                    "controller_target_gap": info_lin.get("controller_target_gap"),
                    "adaptive_cycles_before_fallback": int(
                        info_lin.get("adaptive_cycles_before_fallback", adaptive_pre_fallback_cycles)
                    ),
                    "legacy_fallback_cycles": int(info_lin.get("legacy_fallback_cycles", 0)),
                    "total_cycles": int(inner_cycles_used_i),
                    "refreshed_acceptance_passed": None,
                    "refreshed_acceptance_checked": False,
                    "provisional_picard_acceptance_passed": bool(production_acceptance_passed),
                    "outer_iteration_of_acceptance": None,
                    "termination_reason": "continuing_picard",
                    "forcing_eta": float(forcing_eta_used) if np.isfinite(forcing_eta_used) else None,
                    "previous_outer_head_residual_rms": (
                        float(previous_outer_head_residual_rms_before)
                        if previous_outer_head_residual_rms_before is not None
                        and np.isfinite(previous_outer_head_residual_rms_before)
                        else None
                    ),
                    "previous_outer_dh_rms": (
                        float(previous_outer_dh_rms_before)
                        if previous_outer_dh_rms_before is not None and np.isfinite(previous_outer_dh_rms_before)
                        else None
                    ),
                    "total_inner_kcycles": int(inner_cycles_used_i),
                    "inner_block_count": int(info_lin.get("inner_block_count", 1)),
                    "inner_target_achieved": bool(info_lin.get("inner_target_achieved", False)),
                    "inner_usable_for_picard": bool(info_lin.get("inner_usable_for_picard", True)),
                    "inner_stalled": bool(info_lin.get("inner_stalled", False)),
                    "inner_diverged": bool(info_lin.get("inner_diverged", False)),
                    "inner_rollback_count": int(info_lin.get("inner_rollback_count", 0) or 0),
                    "inner_termination_reason": str(
                        info_lin.get("inner_termination_reason", "legacy_fixed_cycle_cap")
                    ),
                    "legacy_dh_fallback_used": bool(info_lin.get("adaptive_inner_fallback_to_legacy_dh", False)),
                    "inner_cycles_per_block": list(info_lin.get("inner_cycles_per_block", [])),
                    "adaptive_dt_substep_index": int(len(adaptive_dt_substep_dts)),
                    "adaptive_dt_substep_dt": float(actual_dt_f),
                    "adaptive_dt_practical_at_min": bool(adaptive_dt_practical_at_min_b),
                }
                if adaptive_inner_config.save_block_history:
                    outer_summary["inner_cycles_per_block"] = list(info_lin.get("inner_cycles_per_block", []))
                    outer_summary["inner_residuals_per_block"] = list(info_lin.get("inner_residuals_per_block", []))
                    outer_summary["inner_contraction_ratios"] = list(info_lin.get("inner_contraction_ratios", []))
                    outer_summary["inner_per_cycle_convergence_factors"] = list(
                        info_lin.get("inner_per_cycle_convergence_factors", [])
                    )
                    outer_summary["inner_predicted_cycles_per_block"] = list(
                        info_lin.get("inner_predicted_cycles_per_block", [])
                    )
                outer_iteration_summaries.append(outer_summary)
                if production_acceptance_passed and (
                    not adaptive_dt_enabled_b
                    or strict_picard_convergence_passed
                    or adaptive_dt_practical_at_min_b
                ):
                    refreshed_result = evaluate_refreshed_nonlinear_candidate(
                        outer_iteration=int(outer_iter + 1),
                        info_lin=info_lin,
                        dh_max=last_dh_max,
                        dh_rms=last_dh_rms,
                        substep_dt=actual_dt_f,
                        require_strict=bool(adaptive_dt_enabled_b and not adaptive_dt_practical_at_min_b),
                    )
                    outer_summary["refreshed_acceptance_checked"] = True
                    outer_summary["refreshed_acceptance_passed"] = bool(
                        refreshed_result["production_acceptance_passed"]
                    )
                    if refreshed_result["production_acceptance_passed"]:
                        last_head_residual_rms = float(refreshed_result["head_residual_rms"])
                        last_flow_residual_rms = float(refreshed_result["flow_residual_rms"])
                        last_storage_diag_change_max = float(refreshed_result["storage_diag_change_max"])
                        last_storage_diag_change_rms = float(refreshed_result["storage_diag_change_rms"])
                        strict_picard_convergence_passed = bool(refreshed_result["strict_acceptance_passed"])
                        practical_picard_acceptance_passed = bool(refreshed_result["practical_acceptance_passed"])
                        production_acceptance_passed = True
                        outer_summary["termination_reason"] = (
                            "refreshed_strict_acceptance"
                            if strict_picard_convergence_passed else "refreshed_practical_acceptance"
                        )
                        outer_summary["outer_iteration_of_acceptance"] = int(outer_iter + 1)
                        if not adaptive_dt_enabled_b:
                            break
                        adaptive_dt_substep_dts.append(float(actual_dt_f))
                        remaining_dt_f = max(0.0, remaining_dt_f - actual_dt_f)
                        wp.launch(
                            kernel=copy_field_kernel,
                            dim=dim2d,
                            inputs=[h_iter_wp, h_substep_start_wp, self.nx, self.ny],
                            device=device,
                        )
                        if remaining_dt_f <= max(1.0e-12, period_dt_f * 1.0e-12):
                            break
                        wp.launch(
                            kernel=copy_field_kernel,
                            dim=dim2d,
                            inputs=[h_substep_start_wp, h_prev_wp, self.nx, self.ny],
                            device=device,
                        )
                        if refreshed_result["strict_acceptance_passed"] and not adaptive_dt_practical_at_min_b and not adaptive_dt_extension_used_b:
                            # Grow only after a clean strict acceptance (no practical
                            # fallback or budget extension touched this sub-step). A
                            # sub-step that needed assistance keeps dt instead of
                            # re-attempting strict at a larger dt and shrinking
                            # straight back (retry storm / grow-shrink oscillation).
                            if adaptive_dt_growth_steps_i < adaptive_dt_max_growth_steps_i:
                                current_dt_f = min(
                                    period_dt_f,
                                    current_dt_f * adaptive_dt_grow_factor_f,
                                )
                                adaptive_dt_growth_steps_i += 1
                        actual_dt_f = min(current_dt_f, remaining_dt_f)
                        if remaining_dt_f - actual_dt_f < dt_min_f:
                            # Absorb a sub-dt_min sliver into the final sub-step.
                            actual_dt_f = remaining_dt_f
                        substep_outer_limit_i = adaptive_dt_strict_max_outer_i
                        adaptive_dt_practical_at_min_b = False
                        adaptive_dt_dh_history = []
                        adaptive_dt_extension_used_b = False
                        adaptive_dt_early_shrink_streak_i = 0
                        previous_dh_measure = None
                        previous_outer_head_residual_rms = None
                        previous_initial_head_residual_rms = None
                        previous_outer_dh_rms = None
                        storage_diag_prev_wp.fill_(WP_FLOAT(0.0))
                        outer_iter = 0
                        continue
                    production_acceptance_passed = False

                adaptive_dt_budget_exhausted_b = bool(
                    adaptive_dt_enabled_b and outer_iter + 1 >= substep_outer_limit_i
                )
                adaptive_dt_early_shrink_b = False
                if (
                    adaptive_dt_enabled_b
                    and not adaptive_dt_budget_exhausted_b
                    and not adaptive_dt_practical_at_min_b
                    and adaptive_dt_early_shrink_enabled_b
                    and actual_dt_f > dt_min_f + max(1.0e-12, period_dt_f * 1.0e-12)
                ):
                    # Early shrink: the dh contraction projection says strict
                    # cannot reach hclose within the remaining budget — but the
                    # comparison is against budget + available extension, since an
                    # extension at exhaustion finishes a near-miss far cheaper
                    # than a shrink + full retry. Only genuinely hopeless
                    # sub-steps shrink early, and only after the pessimistic
                    # projection persists for early_shrink_patience consecutive
                    # checks (early-iteration contraction is often pessimistic;
                    # it accelerates as the Picard iterate settles).
                    adaptive_dt_effective_budget_i = int(substep_outer_limit_i)
                    if adaptive_dt_extension_enabled_b and not adaptive_dt_extension_used_b:
                        adaptive_dt_effective_budget_i += adaptive_dt_extension_max_outer_i
                    if _adaptive_dt_should_early_shrink(
                        adaptive_dt_dh_history,
                        tol=hclose,
                        outer_iterations_done=int(outer_iter + 1),
                        budget=int(adaptive_dt_effective_budget_i),
                        min_outer=adaptive_dt_early_shrink_min_outer_i,
                    ):
                        adaptive_dt_early_shrink_streak_i += 1
                        adaptive_dt_early_shrink_b = bool(
                            adaptive_dt_early_shrink_streak_i >= adaptive_dt_early_shrink_patience_i
                        )
                    else:
                        adaptive_dt_early_shrink_streak_i = 0
                        adaptive_dt_early_shrink_b = False
                if adaptive_dt_budget_exhausted_b or adaptive_dt_early_shrink_b:
                    if (
                        adaptive_dt_budget_exhausted_b
                        and not adaptive_dt_early_shrink_b
                        and not adaptive_dt_practical_at_min_b
                        and adaptive_dt_extension_enabled_b
                        and not adaptive_dt_extension_used_b
                        and _adaptive_dt_should_extend_budget(
                            adaptive_dt_dh_history,
                            tol=hclose,
                            extension_factor=adaptive_dt_extension_factor_f,
                            extension_contraction_ratio=adaptive_dt_extension_contraction_ratio_f,
                        )
                    ):
                        # Budget extension: strict is close and still contracting,
                        # so a few extra iterations are cheaper than a shrink and
                        # a full retry of the sub-step. At most one per sub-step.
                        substep_outer_limit_i += adaptive_dt_extension_max_outer_i
                        adaptive_dt_extension_used_b = True
                        adaptive_dt_extension_count += 1
                        outer_iter += 1
                        continue
                    if actual_dt_f > dt_min_f + max(1.0e-12, period_dt_f * 1.0e-12):
                        current_dt_f = max(dt_min_f, actual_dt_f * adaptive_dt_shrink_factor_f)
                        actual_dt_f = min(current_dt_f, remaining_dt_f)
                        if remaining_dt_f - actual_dt_f < dt_min_f:
                            # Absorb a sub-dt_min sliver into the final sub-step.
                            actual_dt_f = remaining_dt_f
                        adaptive_dt_growth_steps_i = 0
                        adaptive_dt_retry_count += 1
                        if adaptive_dt_early_shrink_b:
                            adaptive_dt_early_shrink_count += 1
                        wp.launch(
                            kernel=copy_field_kernel,
                            dim=dim2d,
                            inputs=[h_substep_start_wp, h_prev_wp, self.nx, self.ny],
                            device=device,
                        )
                        wp.launch(
                            kernel=copy_field_kernel,
                            dim=dim2d,
                            inputs=[h_substep_start_wp, h_iter_wp, self.nx, self.ny],
                            device=device,
                        )
                        storage_diag_prev_wp.fill_(WP_FLOAT(0.0))
                        adaptive_dt_dh_history = []
                        adaptive_dt_extension_used_b = False
                        adaptive_dt_early_shrink_streak_i = 0
                        previous_dh_measure = None
                        previous_outer_head_residual_rms = None
                        previous_initial_head_residual_rms = None
                        previous_outer_dh_rms = None
                        outer_iter = 0
                        continue
                    if not adaptive_dt_practical_at_min_b:
                        adaptive_dt_practical_at_min_b = True
                        adaptive_dt_practical_fallback_count += 1
                        substep_outer_limit_i = max_outer
                        wp.launch(
                            kernel=copy_field_kernel,
                            dim=dim2d,
                            inputs=[h_substep_start_wp, h_prev_wp, self.nx, self.ny],
                            device=device,
                        )
                        wp.launch(
                            kernel=copy_field_kernel,
                            dim=dim2d,
                            inputs=[h_substep_start_wp, h_iter_wp, self.nx, self.ny],
                            device=device,
                        )
                        storage_diag_prev_wp.fill_(WP_FLOAT(0.0))
                        adaptive_dt_dh_history = []
                        adaptive_dt_extension_used_b = False
                        adaptive_dt_early_shrink_streak_i = 0
                        previous_dh_measure = None
                        previous_outer_head_residual_rms = None
                        previous_initial_head_residual_rms = None
                        previous_outer_dh_rms = None
                        outer_iter = 0
                        continue

                outer_iter += 1

            phase_t0 = _fast_path_phase_start()
            wp.launch(kernel=copy_field_kernel, dim=dim2d, inputs=[storage_diag_wp, storage_diag_prev_wp, self.nx, self.ny], device=device)
            wp.launch(
                kernel=update_unconfined_transmissivity_from_head_kernel,
                dim=dim2d,
                inputs=[h_iter_wp, k_field_wp, bottom_wp, top_wp, self.active_wp, min_sat_f, self.nx, self.ny, self.T_wp],
                device=device
            )
            T_update_seconds += _fast_path_phase_elapsed(phase_t0)
            counters["T_device_updates"] += 1
            phase_t0 = _fast_path_phase_start()
            wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[storage_change_sum_sq_buf], device=device)
            wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[storage_change_max_buf], device=device)
            wp.launch(
                kernel=update_secant_sy_storage_kernel,
                dim=dim2d,
                inputs=[
                    h_iter_wp, h_prev_wp, bottom_wp, top_wp, self.active_wp, self.bc_mask_wp,
                    sy_f, ss_f, dx_f, actual_dt_f, min_sat_f, 1.0e-12, self.nx, self.ny,
                    storage_coeff_wp, sy_coeff_wp, ss_coeff_wp, storage_diag_wp, storage_diag_prev_wp,
                    storage_change_sum_sq_buf, storage_change_max_buf
                ],
                device=device
            )
            storage_kernel_seconds += _fast_path_phase_elapsed(phase_t0)
            counters["storage_device_updates"] += 1
            counters["rhs_device_updates"] += 1
            phase_t0 = _fast_path_phase_start()
            wp.launch(
                kernel=build_transient_rhs_from_storage_kernel,
                dim=dim2d,
                inputs=[
                    self.R_wp,
                    storage_diag_wp,
                    h_prev_wp,
                    self.active_wp,
                    self.bc_mask_wp,
                    self.bc_values_wp,
                    dx_f,
                    self.nx,
                    self.ny,
                    rhs_eff_wp,
                ],
                device=device
            )
            rhs_assembly_seconds += _fast_path_phase_elapsed(phase_t0)
            phase_t0 = _fast_path_phase_start()
            wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[flow_rTr_buf], device=device)
            wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[head_rTr_buf], device=device)
            wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[rhs_rTr_buf], device=device)
            wp.launch(
                kernel=compute_dual_residual_kernel,
                dim=dim2d,
                inputs=[
                    h_iter_wp,
                    rhs_eff_wp,
                    self.T_wp,
                    self.active_wp,
                    self.bc_mask_wp,
                    self.mg_levels[0].gh_mask_wp,
                    self.mg_levels[0].ghb_factor_wp,
                    storage_diag_wp,
                    flow_rTr_buf,
                    head_rTr_buf,
                    self.nx,
                    self.ny,
                ],
                device=device,
            )
            wp.launch(
                kernel=compute_active_rhs_l2_kernel,
                dim=dim2d,
                inputs=[rhs_eff_wp, self.active_wp, self.bc_mask_wp, rhs_rTr_buf, self.nx, self.ny],
                device=device,
            )
            final_nonlinear_residual_check_seconds += _fast_path_phase_elapsed(phase_t0)
            counters["scalar_reductions"] += 1
            counters["gpu_scalar_synchronizations"] += 4
            period_gpu_scalar_syncs += 4
            last_flow_residual_rms = float(
                np.sqrt(max(float(flow_rTr_buf.numpy()[0]), 0.0) / float(max(n_free, 1)))
            )
            last_head_residual_rms = float(
                np.sqrt(max(float(head_rTr_buf.numpy()[0]), 0.0) / float(max(n_free, 1)))
            )
            final_rhs_rms = float(
                np.sqrt(max(float(rhs_rTr_buf.numpy()[0]), 0.0) / float(max(n_free, 1)))
            )
            final_relative_flow_residual_rms = last_flow_residual_rms / max(
                final_rhs_rms, float(adaptive_inner_config.residual_floor)
            )
            last_storage_diag_change_max = float(storage_change_max_buf.numpy()[0])
            last_storage_diag_change_rms = float(
                np.sqrt(max(float(storage_change_sum_sq_buf.numpy()[0]), 0.0) / float(max(n_free, 1)))
            )
            strict_picard_convergence_passed = bool(
                final_linearisation_solved
                and last_dh_max <= hclose
                and last_head_residual_rms <= strict_head_residual_tol_f
            )
            practical_picard_acceptance_passed = bool(
                _adaptive_practical_acceptance_allowed(
                    practical_acceptance_enabled=practical_picard_acceptance_enabled_b,
                    adaptive_controller_used=adaptive_final_linearisation,
                    inner_target_achieved=bool(info_lin.get("inner_target_achieved", False)),
                    final_relative_flow_residual_rms=final_relative_flow_residual_rms,
                    relative_flow_target=float(adaptive_inner_config.relative_flow_residual_target),
                )
                and int(min(max_outer, outer_iter + 1)) >= min_practical_outer_iterations_i
                and np.isfinite(last_head_residual_rms)
                and last_head_residual_rms <= practical_head_residual_tol_f
                and np.isfinite(last_dh_rms)
                and last_dh_rms <= practical_dh_rms_tol_f
                and np.isfinite(last_storage_diag_change_rms)
                and last_storage_diag_change_rms <= practical_storage_diag_change_rms_tol_f
            )
            production_acceptance_passed = bool(
                strict_picard_convergence_passed or practical_picard_acceptance_passed
            )
            if outer_iteration_summaries:
                outer_iteration_summaries[-1]["refreshed_acceptance_passed"] = bool(
                    production_acceptance_passed
                )

            refreshed_result = evaluate_refreshed_nonlinear_candidate(
                outer_iteration=int(min(max_outer, outer_iter + 1)),
                info_lin=info_lin,
                dh_max=last_dh_max,
                dh_rms=last_dh_rms,
                substep_dt=actual_dt_f,
                require_strict=bool(adaptive_dt_enabled_b and not adaptive_dt_practical_at_min_b),
            )
            strict_picard_convergence_passed = bool(refreshed_result["strict_acceptance_passed"])
            practical_picard_acceptance_passed = bool(refreshed_result["practical_acceptance_passed"])
            production_acceptance_passed = bool(refreshed_result["production_acceptance_passed"])
            last_head_residual_rms = float(refreshed_result["head_residual_rms"])
            last_flow_residual_rms = float(refreshed_result["flow_residual_rms"])
            last_storage_diag_change_max = float(refreshed_result["storage_diag_change_max"])
            last_storage_diag_change_rms = float(refreshed_result["storage_diag_change_rms"])
            if outer_iteration_summaries:
                outer_iteration_summaries[-1]["refreshed_acceptance_checked"] = True
                outer_iteration_summaries[-1]["refreshed_acceptance_passed"] = bool(production_acceptance_passed)
                if not production_acceptance_passed:
                    outer_iteration_summaries[-1]["termination_reason"] = "max_outer_iterations"

            if (not production_acceptance_passed) and (not allow_unaccepted_transient_period_b):
                if adaptive_dt_enabled_b and actual_dt_f <= dt_min_f + max(
                    1.0e-12,
                    period_dt_f * 1.0e-12,
                ):
                    raise RuntimeError(f"adaptive dt failed at dt_min={dt_min_f}")
                raise RuntimeError(
                    _format_unaccepted_transient_period_error(
                        period_index=period_index,
                        outer_iterations=int(min(max_outer, outer_iter + 1)),
                        final_max_abs_head_change=last_dh_max,
                        final_rms_head_change=last_dh_rms,
                        final_head_residual_rms=last_head_residual_rms,
                        final_flow_residual_rms=last_flow_residual_rms,
                        storage_diag_change_max=last_storage_diag_change_max,
                        storage_diag_change_rms=last_storage_diag_change_rms,
                        storage_mode=str(storage_mode),
                        storage_reference=str(storage_reference),
                        coarse_operator_mode=str(fast_path_coarse_operator_mode),
                        coarse_krylov_method="recursive_kcycle_safe_alpha",
                        total_inner_cycles=int(total_inner_kcycles),
                        inner_controller_mode=str(info_lin.get("controller_mode", "legacy_dh_schedule")),
                        last_inner_termination_reason=str(
                            info_lin.get("inner_termination_reason", "legacy_fixed_cycle_cap")
                        ),
                        last_inner_initial_residual=info_lin.get("initial_head_residual_rms"),
                        last_inner_target_residual=info_lin.get("target_head_residual_rms"),
                        last_inner_final_residual=info_lin.get("final_head_residual_rms"),
                        last_inner_block_count=info_lin.get("inner_block_count"),
                        stalled_inner_solve_count=int(stalled_inner_solve_count),
                        divergent_inner_solve_count=int(divergent_inner_solve_count),
                    )
                )

            wp.launch(kernel=copy_field_kernel, dim=dim2d, inputs=[h_iter_wp, h_prev_wp, self.nx, self.ny], device=device)

            period_times[period_index] = time.perf_counter() - period_t0
            counters["device_to_host_full_grid_copies"] += 1
            counters["head_downloads"] += 1
            head_download_t0 = _fast_path_phase_start()
            head_arr = np.asarray(h_iter_wp.numpy(), dtype=np.float64)
            head_download_seconds += _fast_path_phase_elapsed(head_download_t0)
            heads_per_period[period_index] = head_arr
            if save_diagnostics_b:
                counters["device_to_host_full_grid_copies"] += 3
                storage_ref_arr = head_arr.copy()
                storage_coeff_arr = np.asarray(storage_coeff_wp.numpy(), dtype=np.float64)
                sy_coeff_arr = np.asarray(sy_coeff_wp.numpy(), dtype=np.float64)
                ss_coeff_arr = np.asarray(ss_coeff_wp.numpy(), dtype=np.float64)
                delta_head = head_arr - period_head_old
                exact_storage_term, exact_sy_term, exact_ss_term = exact_unconfined_storage_terms(
                    head_new=storage_ref_arr,
                    head_old=period_head_old,
                    bottom=bottom,
                    top=top,
                    specific_yield=float(sy),
                    specific_storage=float(ss),
                    dt=dt_f,
                )
                heads_old_per_period[period_index] = period_head_old
                storage_reference_heads[period_index] = storage_ref_arr
                storage_coeffs[period_index] = storage_coeff_arr
                sy_coeffs[period_index] = sy_coeff_arr
                ss_coeffs[period_index] = ss_coeff_arr
                storage_terms[period_index] = exact_storage_term
                sy_terms[period_index] = exact_sy_term
                ss_terms[period_index] = exact_ss_term
                sy_crossing_terms[period_index] = exact_sy_term
            info_period = dict(info_lin) if isinstance(info_lin, dict) else {}
            info_period.update(
                {
                    "solver_type": "kcycle_unconfined_picard_device_fast_path",
                    "converged": bool(production_acceptance_passed),
                    "outer_iterations": int(min(max_outer, outer_iter + 1)),
                    "strict_picard_convergence_passed": bool(strict_picard_convergence_passed),
                    "practical_picard_acceptance_passed": bool(practical_picard_acceptance_passed),
                    "production_acceptance_passed": bool(production_acceptance_passed),
                    "final_max_abs_head_change": float(last_dh_max),
                    "final_rms_head_change": float(last_dh_rms),
                    "final_flow_residual_rms": float(last_flow_residual_rms),
                    "final_rhs_rms": float(final_rhs_rms),
                    "final_relative_flow_residual_rms": float(final_relative_flow_residual_rms),
                    "refreshed_acceptance_passed": bool(production_acceptance_passed),
                    "final_head_residual_rms": float(last_head_residual_rms),
                    "final_residual": float(last_head_residual_rms),
                    "adaptive_inner_controller_enabled": bool(adaptive_inner_config.enabled),
                    "adaptive_inner_controller_used": bool(info_lin.get("adaptive_inner_controller_used", False)),
                    "adaptive_inner_fallback_to_legacy_dh": bool(
                        info_lin.get("adaptive_inner_fallback_to_legacy_dh", False)
                    ),
                    "adaptive_inner_fallback_reason": str(
                        info_lin.get("adaptive_inner_fallback_reason", "")
                    ),
                    "storage_diag_change_max": float(last_storage_diag_change_max),
                    "storage_diag_change_rms": float(last_storage_diag_change_rms),
                    "storage_mode": str(storage_mode),
                    "storage_specific_storage_formulation": "secant_potential",
                    "unconfined_storage_mode_2d": str(storage_mode),
                    "storage_reference": str(storage_reference),
                    "incremental_picard_enabled": bool(use_incremental_picard),
                    "adaptive_dt_enabled": bool(adaptive_dt_enabled_b),
                    "adaptive_dt_min_fraction": float(adaptive_dt_min_fraction_f),
                    "adaptive_dt_retry_count": int(adaptive_dt_retry_count),
                    "adaptive_dt_practical_fallback_count": int(adaptive_dt_practical_fallback_count),
                    "adaptive_dt_early_shrink_count": int(adaptive_dt_early_shrink_count),
                    "adaptive_dt_extension_count": int(adaptive_dt_extension_count),
                    "adaptive_dt_total_outer_iterations": int(adaptive_dt_total_outer_iterations_i),
                    "adaptive_dt_substep_count": int(len(adaptive_dt_substep_dts)),
                    "adaptive_dt_substep_dts": [float(value) for value in adaptive_dt_substep_dts],
                    "device_side_picard_fast_path_active": True,
                    "unconfined_startup_mode": str(startup_mode),
                    "startup_inner_kcycles": int(startup_inner_cycles),
                    "startup_converged": startup_converged,
                    "practical_picard_acceptance_enabled": bool(practical_picard_acceptance_enabled_b),
                    "picard_relax": float(omega_current_f),
                    "max_head_change_per_outer_iteration": float(max_update_f),
                    "strict_head_residual_tol": float(strict_head_residual_tol_f),
                    "min_practical_outer_iterations": int(min_practical_outer_iterations_i),
                    "practical_head_residual_tol": float(practical_head_residual_tol_f),
                    "practical_residual_tol": float(practical_head_residual_tol_f),
                    "practical_residual_tol_deprecated_alias_used": bool(practical_residual_tol_alias_used),
                    "practical_dh_rms_tol": float(practical_dh_rms_tol_f),
                    "practical_storage_diag_change_rms_tol": float(practical_storage_diag_change_rms_tol_f),
                    "total_inner_kcycles": int(total_inner_kcycles),
                    "maximum_inner_kcycles_in_one_outer_iteration": int(maximum_inner_kcycles_in_one_outer_iteration),
                    "mean_inner_kcycles_per_outer_iteration": float(
                        total_inner_kcycles / float(max(int(min(max_outer, outer_iter + 1)), 1))
                    ),
                    "total_inner_blocks": int(sum(inner_block_counts)),
                    "mean_cycles_per_block": float(
                        total_inner_kcycles / float(max(sum(inner_block_counts), 1))
                    ),
                    "stalled_inner_solve_count": int(stalled_inner_solve_count),
                    "divergent_inner_solve_count": int(divergent_inner_solve_count),
                    "rolled_back_block_count": int(rolled_back_block_count),
                    "legacy_dh_fallback_count": int(legacy_dh_fallback_count),
                    "adaptive_target_achievement_count": int(adaptive_target_achievement_count),
                    "adaptive_inner_residual_check_count": int(inner_residual_check_count),
                    "coarse_operator_mode": str(fast_path_coarse_operator_mode),
                    "fine_operator_residual_checked": True,
                    "coarse_krylov_method": "recursive_kcycle_safe_alpha",
                    "gpu_scalar_synchronization_count": int(period_gpu_scalar_syncs),
                    "outer_iteration_summaries": outer_iteration_summaries,
                    "T_update_seconds": float(T_update_seconds),
                    "storage_kernel_seconds": float(storage_kernel_seconds),
                    "fine_m_inv_refresh_seconds": float(fine_m_inv_refresh_seconds),
                    "dynamic_coarse_refresh_seconds": float(dynamic_coarse_refresh_seconds),
                    "rhs_assembly_seconds": float(rhs_assembly_seconds),
                    "storage_assembly_seconds": float(storage_assembly_seconds),
                    "inner_solver_seconds": float(inner_solver_seconds),
                    "outer_convergence_check_seconds": float(outer_convergence_check_seconds),
                    "final_nonlinear_residual_check_seconds": float(final_nonlinear_residual_check_seconds),
                    "head_download_seconds": float(head_download_seconds),
                    "period_total_seconds": float(period_times[period_index]),
                }
            )
            if save_diagnostics_b:
                info_period["inner_kcycle_caps"] = [int(v) for v in inner_kcycle_caps]
                info_period["inner_kcycle_used"] = [int(v) for v in inner_kcycle_used]
                info_period["inner_block_counts"] = [int(v) for v in inner_block_counts]
            period_infos.append(info_period)
            last_info = info_period
            head_prev = head_arr

    else:
        for period_index in range(n_periods):
            self.update_uniform_recharge_in_place(float(rates[period_index]))
            counters["R_device_updates"] += 1
            if save_diagnostics_b:
                period_head_old = np.asarray(head_prev, dtype=np.float64).copy()
            else:
                period_head_old = head_prev
            period_t0 = time.perf_counter()
            head, info = self.solve(
                formulation="unconfined",
                initial_head=head_prev,
                K_field=k,
                zbot_field=bottom,
                ztop_field=top,
                transient=True,
                storage_coeff=None,
                dt=dt_f,
                head_prev=head_prev,
                return_info=True,
                storage_reference=storage_reference,
                unconfined_storage_mode_2d=storage_mode,
                save_transient_diagnostics=save_diagnostics_b,
                sy=float(sy),
                ss=float(ss),
                **controls,
            )
            period_times[period_index] = time.perf_counter() - period_t0
            counters["device_to_host_full_grid_copies"] += 1
            counters["head_downloads"] += 1
            head_arr = np.asarray(head, dtype=np.float64)
            info_out = dict(info) if isinstance(info, dict) else {}
            info_out.setdefault("incremental_picard_enabled", False)
            info_out.setdefault("adaptive_dt_enabled", False)
            storage_ref = info_out.pop("storage_reference_head_last_linearization_array", None)
            storage_coeff = info_out.pop("storage_coeff_last_linearization_array", None)
            sy_coeff = info_out.pop("sy_storage_coeff_last_linearization_array", None)
            ss_coeff = info_out.pop("ss_storage_coeff_last_linearization_array", None)
            if bool(info_out.get("cuda_graph_built_this_call", False)):
                counters["hierarchy_rebuilds"] += 1

            heads_per_period[period_index] = head_arr
            if save_diagnostics_b:
                if storage_ref is None:
                    storage_ref = head_arr if storage_reference == "current_picard" else period_head_old
                if storage_coeff is None:
                    storage_coeff = np.zeros_like(head_arr)
                if sy_coeff is None:
                    sy_coeff = np.zeros_like(head_arr)
                if ss_coeff is None:
                    ss_coeff = np.asarray(storage_coeff, dtype=np.float64) - np.asarray(sy_coeff, dtype=np.float64)

                storage_ref_arr = np.asarray(storage_ref, dtype=np.float64)
                storage_coeff_arr = np.asarray(storage_coeff, dtype=np.float64)
                sy_coeff_arr = np.asarray(sy_coeff, dtype=np.float64)
                ss_coeff_arr = np.asarray(ss_coeff, dtype=np.float64)
                delta_head = head_arr - period_head_old

                exact_storage_term, exact_sy_term, exact_ss_term = exact_unconfined_storage_terms(
                    head_new=storage_ref_arr,
                    head_old=period_head_old,
                    bottom=bottom,
                    top=top,
                    specific_yield=float(sy),
                    specific_storage=float(ss),
                    dt=dt_f,
                )
                heads_old_per_period[period_index] = period_head_old
                storage_reference_heads[period_index] = storage_ref_arr
                storage_coeffs[period_index] = storage_coeff_arr
                sy_coeffs[period_index] = sy_coeff_arr
                ss_coeffs[period_index] = ss_coeff_arr
                storage_terms[period_index] = exact_storage_term
                sy_terms[period_index] = exact_sy_term
                ss_terms[period_index] = exact_ss_term
                sy_crossing_terms[period_index] = exact_sy_term
            period_infos.append(info_out)
            last_info = info_out
            head_prev = head_arr

    info_all = {
        "heads_per_period": heads_per_period,
        "heads_final": heads_per_period[-1],
        "period_infos": period_infos,
        "last_info": last_info,
        "period_times": period_times,
        "total_time": float(time.perf_counter() - total_t0),
        "n_periods": n_periods,
        "storage_reference": storage_reference,
        "dt": dt_f,
        "solve_controls": controls,
        "save_diagnostics": bool(save_diagnostics_b),
        "transient_replay_counters": counters,
    }
    if save_diagnostics_b:
        info_all.update(
            {
                "heads_old_per_period": heads_old_per_period,
                "storage_reference_heads_per_period": storage_reference_heads,
                "storage_coeffs_per_period": storage_coeffs,
                "sy_storage_coeffs_per_period": sy_coeffs,
                "ss_storage_coeffs_per_period": ss_coeffs,
                "storage_terms_per_period": storage_terms,
                "sy_storage_terms_per_period": sy_terms,
                "ss_storage_terms_per_period": ss_terms,
                "sy_crossing_volume_terms_per_period": sy_crossing_terms,
            }
        )
    self._transient_replay_counters = dict(counters)
    return (heads_per_period, info_all) if return_info else heads_per_period

def solve_transient_unconfined(
    context: SolverContext,
    *,
    solver: str | None = "unconfined_picard_kcycle",
    **kwargs: Any,
):
    """Run the model-owned transient driver through its compatibility bridge.

    The bridge is deliberately narrow: all period, adaptive-dt, and diagnostic
    behaviour remains byte-for-byte in the existing implementation until the
    single-step Picard and K-cycle bodies have been extracted.
    """
    if context.formulation != "unconfined":
        raise ValueError("transient unconfined backend requires formulation='unconfined'.")
    backend = select_backend(
        solver=solver,
        formulation=context.formulation,
        transient=True,
        default="unconfined_picard_kcycle",
    )
    if not CAPABILITIES[backend.name].supports_production_period_driver:
        capable = ", ".join(
            name for name, item in CAPABILITIES.items() if item.supports_production_period_driver
        )
        raise ValueError(
            f"solver={backend.name!r} cannot drive the multi-period transient "
            f"production driver; choose one of: {capable}."
        )
    if backend.name == "unconfined_picard_kcycle":
        # Production default: the Picard period driver, byte-for-byte unchanged.
        result = solve_transient_unconfined_backend(model=context.model, **kwargs)
    else:
        # Explicitly selected experimental nonlinear backends run through the
        # capability-gated experimental driver (per-timestep solves, retry,
        # fallback, budgets, full histories).
        result = solve_transient_unconfined_experimental(
            model=context.model,
            backend_name=backend.name,
            **kwargs,
        )
    if kwargs.get("return_info", True):
        heads, info = result
        info_out = dict(info)
        info_out["solver_backend"] = backend.name
        last_info = info_out.get("last_info")
        if isinstance(last_info, dict):
            info_out["last_info"] = dict(last_info, solver_backend=backend.name)
        period_infos = info_out.get("period_infos")
        if isinstance(period_infos, list):
            info_out["period_infos"] = [
                dict(period_info, solver_backend=backend.name)
                if isinstance(period_info, dict)
                else period_info
                for period_info in period_infos
            ]
        return heads, info_out
    return result

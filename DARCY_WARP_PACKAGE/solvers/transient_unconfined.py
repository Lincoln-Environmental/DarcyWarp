# SPDX-License-Identifier: AGPL-3.0-or-later
"""Production multi-period unconfined Picard/K-cycle driver boundary."""

from __future__ import annotations

from typing import Any
import time

import numpy as np
from DARCY_WARP_PACKAGE.physics.storage_2d import exact_unconfined_storage_terms

from .device_picard_periods import run_device_picard_periods, select_device_picard_options
from .context import SolverContext
from .registry import select_backend
from .capabilities import CAPABILITIES
from .transient_experimental import solve_transient_unconfined_experimental

def _prepare_transient_fields(
    *, model, initial_head, k_field, zbot_field, ztop_field, active, bc_mask,
    bc_values, recharge_rates, dt, dtype,
):
    """Normalise the grid inputs without writing any model or device state.

    :param model: Model providing nx and ny.
    :param initial_head: Initial head grid (ny, nx), in length units.
    :param k_field: Hydraulic conductivity grid, in length per time.
    :param zbot_field: Cell-bottom elevation grid, in length units.
    :param ztop_field: Cell-top elevation grid, in length units.
    :param active: Active-cell mask, or None for an entirely active grid.
    :param bc_mask: Prescribed-head mask, or None for no prescribed heads.
    :param bc_values: Prescribed head grid, or None to use initial_head.
    :param recharge_rates: Uniform recharge rate per period, length per time.
    :param dt: Positive stress period duration in model time units.
    :param dtype: NumPy floating-point type used by the selected 2D solver.
    :returns: Head, conductivity, bottom, top, active mask, boundary mask,
        boundary heads, one-dimensional recharge rates and period duration.
    :raises ValueError: If shapes are inconsistent, rates empty or dt invalid.
    """
    h0 = np.asarray(initial_head, dtype=dtype)
    if h0.shape != (model.ny, model.nx):
        raise ValueError(f"initial_head shape {h0.shape} expected {(model.ny, model.nx)}")
    k = np.asarray(k_field, dtype=dtype)
    bottom = np.asarray(zbot_field, dtype=dtype)
    top = np.asarray(ztop_field, dtype=dtype)
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
        bc_v = np.asarray(h0, dtype=dtype)
    else:
        bc_v = np.asarray(bc_values, dtype=dtype)
    for name, arr in (("active", active_i), ("bc_mask", bc_i), ("bc_values", bc_v)):
        if arr.shape != h0.shape:
            raise ValueError(f"{name} shape {arr.shape} expected {h0.shape}")

    rates = np.asarray(recharge_rates, dtype=dtype).reshape(-1)
    if rates.size < 1:
        raise ValueError("recharge_rates must contain at least one period.")
    dt_f = float(dt)
    if not np.isfinite(dt_f) or dt_f <= 0.0:
        raise ValueError("dt must be finite and > 0.")
    return h0, k, bottom, top, active_i, bc_i, bc_v, rates, dt_f


def _prepare_transient_controls(
    *, solve_controls, save_diagnostics: bool, return_info: bool,
    min_saturated_thickness: float,
):
    """Separate single-solve controls from period and device-only controls.

    :param solve_controls: Optional input mapping; it is copied, not mutated.
    :param save_diagnostics: Default for saving full-grid storage histories.
    :param return_info: Whether diagnostic results are returned to the caller.
    :param min_saturated_thickness: Default thickness floor, in length units.
    :returns: Host solve controls, device controls, diagnostics flag,
        rejected-trial reporting flag and validated thickness floor.
    :raises ValueError: If the selected minimum thickness is not positive and finite.
    """
    controls = {} if solve_controls is None else dict(solve_controls)
    save_diagnostics_b = bool(controls.pop("save_transient_diagnostics", save_diagnostics))
    allow_rejected_trial = bool(controls.get("allow_unaccepted_transient_period", False)) and return_info
    fast_path_controls = dict(controls)
    # Drop keys consumed by this wrapper or the device fast path so the
    # ``**controls`` spread forwarded to ``solve()`` does not inject unexpected
    # keywords.
    for _control_key in (
        "unconfined_ghb_relax",
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
        "transient_face_operator_enabled",
        "transient_face_graphs_enabled",
        "transient_mixed_precision_enabled",
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
    if not np.isfinite(min_sat) or min_sat <= 0.0:
        raise ValueError("min_saturated_thickness must be positive and finite.")
    controls["min_saturated_thickness"] = min_sat
    return controls, fast_path_controls, save_diagnostics_b, allow_rejected_trial, min_sat


def _assemble_transient_result(
    *, model, heads_per_period, head_prev, accepted_periods, n_periods, period_infos,
    last_info, period_times, total_t0, dt_f, storage_reference, controls, counters,
    active_i, bc_i, return_info, histories,
):
    """Trim histories to returned periods and publish the last accepted state.

    :param model: Model receiving the final replay counters.
    :param heads_per_period: Preallocated head history, including any reported trial.
    :param head_prev: Last accepted head, not the last rejected trial.
    :param accepted_periods: Number of fully accepted periods.
    :param n_periods: Requested number of periods.
    :param period_infos: Diagnostic record for each returned period.
    :param last_info: Final returned period's diagnostic record.
    :param period_times: Per-period elapsed times in seconds.
    :param total_t0: perf_counter timestamp at the start of execution.
    :param dt_f: Period duration in model time units.
    :param storage_reference: Storage linearisation reference policy.
    :param controls: Effective solve control mapping.
    :param counters: Work and data-transfer counters.
    :param active_i: Active-cell mask, shape (ny, nx).
    :param bc_i: Prescribed-head mask, shape (ny, nx).
    :param return_info: Return a diagnostic dictionary alongside the heads.
    :param histories: Optional named storage and head history arrays, mutated in
        place to zero storage terms and coefficients at excluded cells.
    :returns: Head history, or (head history, diagnostic dictionary).
    """
    returned_periods = len(period_infos)
    heads_per_period = heads_per_period[:returned_periods]
    info_all = {
        "heads_per_period": heads_per_period,
        "heads_final": np.asarray(head_prev).copy(),
        "accepted_periods": accepted_periods,
        "accepted_time": accepted_periods * dt_f,
        "requested_periods": n_periods,
        "period_infos": period_infos,
        "last_info": last_info,
        "period_times": period_times[:returned_periods],
        "total_time": float(time.perf_counter() - total_t0),
        "n_periods": returned_periods,
        "storage_reference": storage_reference,
        "dt": dt_f,
        "solve_controls": controls,
        "save_diagnostics": bool(histories is not None),
        "transient_replay_counters": counters,
    }
    if histories is not None:
        # Prescribed and inactive cells have no storage equation. Head history
        # still records their values; only storage rates and coefficients vanish.
        excluded = (active_i == 0) | (bc_i != 0)
        for name, field in histories.items():
            if name not in ("heads_old_per_period", "storage_reference_heads_per_period"):
                field[:returned_periods, excluded] = 0.0
            info_all[name] = field[:returned_periods]
    model._transient_replay_counters = dict(counters)
    return (heads_per_period, info_all) if return_info else heads_per_period


def _run_host_picard_periods(
    *, model, rates, head_prev, k, bottom, top, sy, ss, dt_f, storage_reference,
    storage_mode, controls, allow_rejected_trial, heads_per_period, period_times,
    period_infos, counters, histories,
):
    """Advance the host-controlled Picard path through requested periods.

    :param model: Built 2D model (updated with each period's recharge and solve).
    :param rates: Uniform recharge rates, length per time, in period order.
    :param head_prev: Last accepted head (initial head on entry).
    :param k: Hydraulic conductivity grid, length per time.
    :param bottom: Cell bottom elevations.
    :param top: Cell top elevations.
    :param sy: Dimensionless specific yield.
    :param ss: Specific storage, inverse length.
    :param dt_f: Period duration in model time units.
    :param storage_reference: Policy selecting the reference head for storage.
    :param storage_mode: Storage constitutive relation passed to solve.
    :param controls: Host Picard solver controls, forwarded unchanged.
    :param allow_rejected_trial: Return a terminal rejected trial rather than raise.
    :param heads_per_period: Writable preallocated head history.
    :param period_times: Writable elapsed time array in seconds.
    :param period_infos: List to which each returned period record is appended.
    :param counters: Work counters, updated in place.
    :param histories: Optional writable full-grid diagnostic histories.
    :returns: Last accepted head, last period info and number of accepted periods.
    :raises RuntimeError: If a period is rejected and trial reporting is disabled.
    """
    save_diagnostics_b = histories is not None
    accepted_periods = 0
    last_info = {}
    for period_index in range(int(rates.size)):
        model.update_uniform_recharge_in_place(float(rates[period_index]))
        counters["R_device_updates"] += 1
        if save_diagnostics_b:
            period_head_old = np.asarray(head_prev, dtype=np.float64).copy()
        else:
            period_head_old = head_prev
        period_t0 = time.perf_counter()
        head, info = model.solve(
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
        period_accepted = bool(info_out.get("converged", False))
        if not period_accepted and not allow_rejected_trial:
            raise RuntimeError(f"Unconfined transient period {period_index + 1} did not converge.")
        info_out["state"] = "accepted" if period_accepted else "rejected_trial"
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
            if storage_reference == "previous_period":
                delta_head = head_arr - period_head_old
                exact_sy_term = sy_coeff_arr * delta_head / dt_f
                exact_ss_term = ss_coeff_arr * delta_head / dt_f
                exact_storage_term = exact_sy_term + exact_ss_term
            else:
                exact_storage_term, exact_sy_term, exact_ss_term = exact_unconfined_storage_terms(
                    head_new=head_arr, head_old=period_head_old, bottom=bottom, top=top,
                    specific_yield=float(sy), specific_storage=float(ss), dt=dt_f,
                )
            histories["heads_old_per_period"][period_index] = period_head_old
            histories["storage_reference_heads_per_period"][period_index] = storage_ref_arr
            histories["storage_coeffs_per_period"][period_index] = storage_coeff_arr
            histories["sy_storage_coeffs_per_period"][period_index] = sy_coeff_arr
            histories["ss_storage_coeffs_per_period"][period_index] = ss_coeff_arr
            histories["storage_terms_per_period"][period_index] = exact_storage_term
            histories["sy_storage_terms_per_period"][period_index] = exact_sy_term
            histories["ss_storage_terms_per_period"][period_index] = exact_ss_term
            histories["sy_crossing_volume_terms_per_period"][period_index] = exact_sy_term
        period_infos.append(info_out)
        last_info = info_out
        if not period_accepted:
            break
        accepted_periods += 1
        head_prev = head_arr
    return head_prev, last_info, accepted_periods


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
        gh_mask: np.ndarray | None = None,
        gh_head: np.ndarray | None = None,
        gh_width: np.ndarray | None = None,
        gh_alpha: float | np.ndarray = 1.0,
        aq_thickness: float | np.ndarray | None = None,
        storage_mode: str = "mf6_convertible_secant_sy",
        storage_reference: str = "current_picard",
        solve_controls: dict | None = None,
        min_saturated_thickness: float = 0.1,
        save_diagnostics: bool = False,
        return_info: bool = True,
):
    """Advance a 2D unconfined groundwater model through uniform-recharge periods.

    All grid fields use shape (ny, nx) and consistent model length/time units.
    Each returned period is identified as accepted or rejected; only accepted
    heads advance the storage reference for subsequent periods.

    :param model: Model whose grid, boundary state and device buffers are used.
    :param initial_head: Initial hydraulic head grid, length units.
    :param recharge_rates: One recharge rate per period, length per time.
    :param k_field: Saturated hydraulic conductivity grid, length per time.
    :param zbot_field: Cell bottom elevations, length units.
    :param ztop_field: Cell top elevations, length units.
    :param sy: Specific yield, dimensionless.
    :param ss: Specific storage, inverse length.
    :param dt: Duration of each stress period in model time units.
    :param active: Optional active mask; defaults to all active.
    :param bc_mask: Optional Dirichlet mask; defaults to no Dirichlet cells.
    :param bc_values: Prescribed heads; defaults to initial heads.
    :param gh_mask: GHB cell mask, required when model.use_ghb is true.
    :param gh_head: External GHB stage grid in length units.
    :param gh_width: GHB connection width grid in length units.
    :param gh_alpha: GHB conductance scaling coefficient, scalar or grid.
        See compute_ghb_factor_from_raw_fields for the dimensional convention.
    :param aq_thickness: Aquifer thickness used to derive GHB conductance.
    :param storage_mode: Constitutive storage relation for the Picard solve.
    :param storage_reference: Current Picard iterate or previous accepted period.
    :param solve_controls: Optional solver and timestep controls, not mutated.
    :param min_saturated_thickness: Positive thickness floor in length units.
    :param save_diagnostics: Include full-grid storage and reference histories.
    :param return_info: Return (heads, info) instead of just the head history.
    :returns: Head history, optionally accompanied by diagnostic information.
    :raises ValueError: If input geometry or solver controls are invalid.
    :raises RuntimeError: If a period cannot be accepted and trial reporting is off.
    """
    import warp as wp
    from DARCY_WARP_PACKAGE.warped_darcy import NP_FLOAT, WP_FLOAT
    self = model
    h0, k, bottom, top, active_i, bc_i, bc_v, rates, dt_f = _prepare_transient_fields(
        model=self, initial_head=initial_head, k_field=k_field, zbot_field=zbot_field,
        ztop_field=ztop_field, active=active, bc_mask=bc_mask, bc_values=bc_values,
        recharge_rates=recharge_rates, dt=dt, dtype=NP_FLOAT,
    )

    controls, fast_path_controls, save_diagnostics_b, allow_rejected_trial, min_sat = (
        _prepare_transient_controls(
            solve_controls=solve_controls, save_diagnostics=save_diagnostics,
            return_info=return_info, min_saturated_thickness=min_saturated_thickness,
        )
    )
    accepted_periods = 0
    thickness = np.clip(h0 - bottom, min_sat, np.maximum(top - bottom, min_sat))
    initial_T = np.asarray(k * thickness, dtype=NP_FLOAT)
    initial_T[active_i == 0] = 0.0
    recharge_field = np.zeros(h0.shape, dtype=NP_FLOAT)
    if bool(self.use_ghb):
        # Phase D (UNCONFINED_FAST_PLAN.md): GHB state is forwarded to the
        # model build; ghb_factor is derived from gh_width/gh_alpha/
        # aq_thickness inside build_from_fields.  The device fast path
        # supports GHB only with the face operator (see the gate below).
        for _name, _arr in (("gh_mask", gh_mask), ("gh_head", gh_head), ("gh_width", gh_width)):
            if _arr is None:
                raise ValueError(f"{_name} is required when the model has use_ghb=True.")
            if np.asarray(_arr).shape != h0.shape:
                raise ValueError(f"{_name} shape {np.asarray(_arr).shape} expected {h0.shape}")
    self.build_from_fields(
        T_field=initial_T,
        R_field=recharge_field,
        active=active_i,
        bc_mask=bc_i,
        bc_values=bc_v,
        gh_mask=gh_mask,
        gh_head=gh_head,
        gh_width=gh_width,
        gh_alpha=gh_alpha,
        aq_thickness=aq_thickness,
    )

    n_periods = int(rates.size)
    heads_per_period = np.zeros((n_periods, self.ny, self.nx), dtype=np.float64)
    histories = None
    if save_diagnostics_b:
        histories = {
            "heads_old_per_period": np.zeros_like(heads_per_period),
            "storage_reference_heads_per_period": np.zeros_like(heads_per_period),
            "storage_coeffs_per_period": np.zeros_like(heads_per_period),
            "sy_storage_coeffs_per_period": np.zeros_like(heads_per_period),
            "ss_storage_coeffs_per_period": np.zeros_like(heads_per_period),
            "storage_terms_per_period": np.zeros_like(heads_per_period),
            "sy_storage_terms_per_period": np.zeros_like(heads_per_period),
            "ss_storage_terms_per_period": np.zeros_like(heads_per_period),
            "sy_crossing_volume_terms_per_period": np.zeros_like(heads_per_period),
        }
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

    options = select_device_picard_options(
        fast_path_controls=fast_path_controls, fp64_fields=WP_FLOAT == wp.float64,
        use_ghb=bool(self.use_ghb), storage_mode=storage_mode,
        storage_reference=storage_reference,
    )
    if options.enabled:
        controls = fast_path_controls
        head_prev, last_info, accepted_periods = run_device_picard_periods(
            model=self, options=options, rates=rates, head_prev=head_prev,
            k=k, bottom=bottom, top=top, active_i=active_i, bc_i=bc_i, sy=sy, ss=ss,
            dt_f=dt_f, storage_reference=storage_reference, storage_mode=storage_mode,
            min_sat=min_sat, controls=controls, allow_rejected_trial=allow_rejected_trial,
            heads_per_period=heads_per_period, period_times=period_times,
            period_infos=period_infos, counters=counters, histories=histories,
        )
    else:
        head_prev, last_info, accepted_periods = _run_host_picard_periods(
            model=self, rates=rates, head_prev=head_prev, k=k, bottom=bottom, top=top,
            sy=sy, ss=ss, dt_f=dt_f, storage_reference=storage_reference,
            storage_mode=storage_mode, controls=controls, allow_rejected_trial=allow_rejected_trial,
            heads_per_period=heads_per_period, period_times=period_times,
            period_infos=period_infos, counters=counters, histories=histories,
        )

    return _assemble_transient_result(
        model=self, heads_per_period=heads_per_period, head_prev=head_prev,
        accepted_periods=accepted_periods, n_periods=n_periods, period_infos=period_infos,
        last_info=last_info, period_times=period_times, total_t0=total_t0, dt_f=dt_f,
        storage_reference=storage_reference, controls=controls, counters=counters,
        active_i=active_i, bc_i=bc_i, return_info=return_info,
        histories=histories,
    )


def solve_transient_unconfined(
    context: SolverContext,
    *,
    solver: str | None = "unconfined_picard_kcycle",
    **kwargs: Any,
):
    """Dispatch to the selected unconfined multi-period backend.

    :param context: Model and formulation supplied to the solver registry.
    :param solver: Backend name; defaults to the production Picard K-cycle driver.
    :param kwargs: Physical inputs and controls of the selected period driver.
    :returns: Head history, optionally with backend-labelled diagnostics.
    :raises ValueError: If the backend does not support the selected formulation.
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
        # Explicitly selected nonlinear backends run through the alternate
        # capability-gated driver (per-timestep solves, retry, fallback,
        # budgets, full histories). FAS remains experimental; Newton is a
        # production alternative.
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

# SPDX-License-Identifier: AGPL-3.0-only
"""Experimental multi-period transient driver for nonlinear backends.

Drives backends that expose the single-timestep transient contract
(``unconfined_fas`` and ``unconfined_semismooth_newton_kcycle``) through
complete transient simulations:

* multiple timesteps within one stress period (``experimental_max_dt`` cap
  and/or retry-driven sub-stepping),
* multiple sequential stress periods with changing recharge or signed source
  fields, changing prescribed-head values, and changing timestep lengths,
* failed-timestep retry with dt reduction (the last accepted head is never
  overwritten by a rejected trial),
* per-timestep fallback to another backend (recorded, not silent),
* complete per-timestep histories, budgets, and replay-compatible
  period-granularity diagnostics.

The production Picard driver (``transient_unconfined.py``) is untouched and
remains the default; this driver runs only when an experimental backend is
explicitly selected through ``solve_transient_unconfined``.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import warp as wp

from DARCY_WARP_PACKAGE.physics.budgets_2d import (
    add_exact_storage_to_budget,
    compute_mass_balance_budget,
)
from DARCY_WARP_PACKAGE.physics.storage_2d import exact_unconfined_storage_terms

# Match the solver stack's float default (warped_darcy.py and
# nonlinear/kernels.py both default DARCY_FLOAT to float64); importing
# config.py here would pin the pytest session to config.py's float32 default
# and break other modules' kernel/array dtype agreement.
NP_FLOAT = np.float64

EXPERIMENTAL_TRANSIENT_BACKENDS = (
    "unconfined_fas",
    "unconfined_semismooth_newton_kcycle",
)

_BACKEND_CONTROL_PREFIX = {
    "unconfined_fas": "fas_",
    "unconfined_semismooth_newton_kcycle": "newton_",
}

_BUDGET_RATE_KEYS = (
    "rcha_in", "rcha_out", "chd_in", "chd_out", "ghb_in", "ghb_out",
    "storage_in", "storage_out",
)


def _merge_timestep_budgets(records: list[dict]) -> dict:
    """Merge per-timestep budget rows into one period row.

    Budget columns are rates; the period row is the dt-weighted mean rate with
    totals and percent discrepancy recomputed with the same formulas as
    ``physics.budgets_2d``.
    """
    merged = {key: 0.0 for key in _BUDGET_RATE_KEYS}
    total_weight = 0.0
    for record in records:
        if not record.get("accepted"):
            continue
        summary = record.get("budget_summary") or {}
        weight = float(record["dt"])
        total_weight += weight
        for key in _BUDGET_RATE_KEYS:
            merged[key] += float(summary.get(key, 0.0) or 0.0) * weight
    if total_weight > 0.0:
        for key in _BUDGET_RATE_KEYS:
            merged[key] /= total_weight
    total_in = merged["rcha_in"] + merged["chd_in"] + merged["ghb_in"] + merged["storage_in"]
    total_out = merged["rcha_out"] + merged["chd_out"] + merged["ghb_out"] + merged["storage_out"]
    imbalance = total_in - total_out
    denominator = abs(total_in) + abs(total_out)
    merged["total_in"] = total_in
    merged["total_out"] = total_out
    merged["in_minus_out"] = imbalance
    merged["percent_discrepancy"] = 0.0 if denominator == 0.0 else 100.0 * imbalance / denominator
    merged["throughflow"] = 0.5 * (total_in + total_out)
    merged["imbalance_fraction"] = 0.0 if (total_in + total_out) == 0.0 else imbalance / (0.5 * (total_in + total_out))
    return merged


def _timestep_budget_summary(
    model: Any,
    *,
    head: np.ndarray,
    head_prev: np.ndarray,
    dt: float,
    k: np.ndarray,
    bottom: np.ndarray,
    top: np.ndarray,
    active: np.ndarray,
    sy: float,
    ss: float,
    min_sat: float,
) -> dict:
    """Evaluate the complete timestep water budget for an accepted head.

    Used when the backend (or its per-timestep fallback) did not attach a
    budget to its info dict — e.g. a Picard fallback solve.  Mirrors the
    FAS/Newton budget construction: package budget at the accepted head plus
    exact nonlinear storage, area-integrated.
    """
    head_arr = np.asarray(head, dtype=np.float64)
    thickness = np.clip(head_arr - bottom, min_sat, np.maximum(top - bottom, min_sat))
    t_field = np.asarray(k * thickness, dtype=np.float64)
    t_field[np.asarray(active, dtype=np.int32) == 0] = 0.0
    ghb_kwargs: dict[str, Any] = {}
    if bool(getattr(model, "use_ghb", False)):
        ghb_kwargs = {
            "gh_mask": np.asarray(model.gh_mask_host, dtype=np.int32),
            "gh_head": np.asarray(model.gh_head_host, dtype=np.float64),
            "ghb_factor": np.asarray(model.ghb_factor_host, dtype=np.float64),
        }
    budget = compute_mass_balance_budget(
        T_field=t_field,
        R_field=np.asarray(model.R_field_host, dtype=np.float64),
        head=head_arr,
        active=np.asarray(active, dtype=np.int32),
        bc_mask=np.asarray(model.bc_mask_host, dtype=np.int32),
        bc_values=np.asarray(model.bc_values_host, dtype=np.float64),
        dx=float(model.dx),
        **ghb_kwargs,
    )
    storage_term, _, _ = exact_unconfined_storage_terms(
        head_new=head_arr,
        head_old=np.asarray(head_prev, dtype=np.float64),
        bottom=bottom,
        top=top,
        specific_yield=float(sy),
        specific_storage=float(ss),
        dt=float(dt),
    )
    # exact_unconfined_storage_terms is per unit plan area; budgets are
    # area-integrated.
    budget = add_exact_storage_to_budget(budget, storage_term * float(model.dx) ** 2)
    return dict(budget.iloc[0])


def solve_transient_unconfined_experimental(
    *,
    model: Any,
    backend_name: str,
    initial_head: np.ndarray,
    recharge_rates: np.ndarray,
    k_field: np.ndarray,
    zbot_field: np.ndarray,
    ztop_field: np.ndarray,
    sy: float,
    ss: float,
    dt: float | np.ndarray,
    active: np.ndarray,
    bc_mask: np.ndarray,
    bc_values: np.ndarray,
    storage_mode: str | None = None,
    storage_reference: str | None = None,
    solve_controls: dict | None = None,
    min_saturated_thickness: float = 0.1,
    return_info: bool = True,
    source_fields_per_period: np.ndarray | None = None,
    bc_values_per_period: np.ndarray | None = None,
    save_transient_diagnostics: bool = True,
):
    """Run a complete transient simulation with an experimental backend.

    Every timestep solves the authoritative nonlinear equation for
    ``head_new`` given the previous accepted head, the current dt, the current
    stress-period source and boundary data, and the current Sy/Ss.  The
    previous-head state advances only on accepted timesteps.
    """
    if backend_name not in EXPERIMENTAL_TRANSIENT_BACKENDS:
        raise ValueError(
            f"backend {backend_name!r} is not supported by the experimental "
            f"transient driver; choose one of: {', '.join(EXPERIMENTAL_TRANSIENT_BACKENDS)}."
        )

    controls = dict(solve_controls or {})
    save_diagnostics_b = bool(controls.pop("save_transient_diagnostics", save_transient_diagnostics))
    max_dt = controls.pop("experimental_max_dt", None)
    max_dt = None if max_dt is None else float(max_dt)
    if max_dt is not None and (not np.isfinite(max_dt) or max_dt <= 0.0):
        raise ValueError("experimental_max_dt must be positive and finite.")
    shrink_factor = float(controls.pop("experimental_dt_shrink_factor", 0.5))
    grow_factor = float(controls.pop("experimental_dt_grow_factor", 2.0))
    dt_min_fraction = float(controls.pop("experimental_dt_min_fraction", 0.0625))
    max_growth_steps = int(controls.pop("experimental_dt_max_growth_steps", 2))
    max_retries = int(controls.pop("experimental_max_retries", 16))
    budget_tol_pct = float(controls.pop("experimental_budget_max_percent_discrepancy", 0.1))
    retry_enabled = bool(controls.pop("experimental_retry_enabled", True))
    allow_unaccepted = bool(controls.pop("allow_unaccepted_transient_period", False))
    prefix = _BACKEND_CONTROL_PREFIX[backend_name]
    backend_kwargs = {key: value for key, value in controls.items() if str(key).startswith(prefix)}

    h0 = np.asarray(initial_head, dtype=np.float64)
    k = np.asarray(k_field, dtype=np.float64)
    bottom = np.asarray(zbot_field, dtype=np.float64)
    top = np.asarray(ztop_field, dtype=np.float64)
    active_i = np.asarray(active, dtype=np.int32)
    bc_i = np.asarray(bc_mask, dtype=np.int32)
    bc_v = np.asarray(bc_values, dtype=np.float64)
    for name, arr in (("k_field", k), ("zbot_field", bottom), ("ztop_field", top), ("active", active_i), ("bc_mask", bc_i), ("bc_values", bc_v)):
        if arr.shape != h0.shape:
            raise ValueError(f"{name} shape {arr.shape} expected {h0.shape}")

    rates = np.asarray(recharge_rates, dtype=np.float64).reshape(-1)
    source_fields = None
    if source_fields_per_period is not None:
        source_fields = np.asarray(source_fields_per_period, dtype=np.float64)
        if source_fields.ndim != 3 or source_fields.shape[1:] != h0.shape:
            raise ValueError(
                f"source_fields_per_period must have shape (n_periods, {h0.shape[0]}, {h0.shape[1]}), "
                f"got {source_fields.shape}."
            )
    n_periods = int(source_fields.shape[0]) if source_fields is not None else int(rates.size)
    if n_periods < 1:
        raise ValueError("at least one stress period is required.")
    if source_fields is None and rates.size != n_periods:
        raise ValueError("recharge_rates must contain one entry per stress period.")

    dt_array = np.asarray(dt if np.ndim(dt) else [float(dt)] * n_periods, dtype=np.float64).reshape(-1)
    if dt_array.size == 1 and n_periods > 1:
        dt_array = np.full(n_periods, float(dt_array[0]), dtype=np.float64)
    if dt_array.size != n_periods:
        raise ValueError(f"dt must be scalar or contain one entry per stress period ({n_periods}).")
    if not np.all(np.isfinite(dt_array)) or not np.all(dt_array > 0.0):
        raise ValueError("dt entries must be finite and > 0.")

    bc_sequence = None
    if bc_values_per_period is not None:
        bc_sequence = np.asarray(bc_values_per_period, dtype=np.float64)
        if bc_sequence.ndim != 3 or bc_sequence.shape[1:] != h0.shape or int(bc_sequence.shape[0]) != n_periods:
            raise ValueError(
                f"bc_values_per_period must have shape (n_periods={n_periods}, {h0.shape[0]}, {h0.shape[1]}), "
                f"got {bc_sequence.shape}."
            )

    sy_f = float(sy)
    ss_f = float(ss)
    min_sat = float(controls.get("min_saturated_thickness", min_saturated_thickness))
    thickness = np.clip(h0 - bottom, min_sat, np.maximum(top - bottom, min_sat))
    initial_T = np.asarray(k * thickness, dtype=NP_FLOAT)
    initial_T[active_i == 0] = 0.0
    recharge_field = np.zeros(h0.shape, dtype=NP_FLOAT)
    model.build_from_fields(
        T_field=initial_T,
        R_field=recharge_field,
        active=active_i,
        bc_mask=bc_i,
        bc_values=bc_v,
    )

    ny, nx = h0.shape
    heads_per_period = np.zeros((n_periods, ny, nx), dtype=np.float64)
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

    counters = {
        "R_device_updates": 0,
        "bc_device_updates": 0,
        "head_downloads": 0,
        "device_to_host_full_grid_copies": 0,
        "hierarchy_rebuilds": 0,
        "gpu_scalar_synchronizations": 0,
        "diagnostic_full_grid_arrays_saved": int(save_diagnostics_b),
        "experimental_backend_solves": 0,
        "experimental_timestep_count": 0,
        "experimental_retry_count": 0,
        "experimental_rejected_attempt_count": 0,
        "experimental_fallback_timestep_count": 0,
        "experimental_forced_accept_count": 0,
        "device_side_picard_fast_path_active": 0,
    }

    period_infos: list[dict] = []
    period_budgets: list[dict] = []
    timestep_records: list[dict] = []
    period_times = np.zeros(n_periods, dtype=np.float64)

    head_accepted = np.asarray(h0, dtype=np.float64).copy()
    simulation_time = 0.0
    total_t0 = time.perf_counter()
    last_info: dict = {}

    for period_index in range(n_periods):
        # ---------------- stress-period transition ----------------
        if source_fields is not None:
            model.update_R_in_place(source_fields[period_index])
        else:
            model.update_uniform_recharge_in_place(float(rates[period_index]))
        counters["R_device_updates"] += 1
        if bc_sequence is not None:
            model.update_bc_in_place(bc_sequence[period_index])
            counters["bc_device_updates"] += 1

        period_dt = float(dt_array[period_index])
        dt_min = max(period_dt * dt_min_fraction, 1.0e-12)
        remaining_dt = period_dt
        current_dt = min(period_dt, max_dt) if max_dt is not None else period_dt
        growth_steps = 0
        retries_for_timestep = 0
        retries_this_period = 0
        rejected_this_period = 0
        fallback_this_period = 0
        period_head_old = head_accepted.copy()
        period_record_start = len(timestep_records)
        period_t0 = time.perf_counter()
        timestep_index = 0
        period_solve_seconds = 0.0

        while remaining_dt > max(1.0e-12, period_dt * 1.0e-12):
            dt_try = min(current_dt, remaining_dt)
            if remaining_dt - dt_try < dt_min:
                dt_try = remaining_dt  # absorb sliver
            solve_t0 = time.perf_counter()
            head_trial, info = model.solve(
                formulation="unconfined",
                solver=backend_name,
                initial_head=head_accepted,
                K_field=k,
                zbot_field=bottom,
                ztop_field=top,
                transient=True,
                storage_coeff=sy_f,
                sy=sy_f,
                ss=ss_f,
                dt=dt_try,
                head_prev=head_accepted,
                return_info=True,
                **backend_kwargs,
            )
            try:
                wp.synchronize_device(model.device_str)
            except Exception:
                pass
            solve_seconds = time.perf_counter() - solve_t0
            period_solve_seconds += solve_seconds
            counters["experimental_backend_solves"] += 1
            counters["head_downloads"] += 1
            counters["device_to_host_full_grid_copies"] += 1
            info = dict(info) if isinstance(info, dict) else {}
            last_info = info

            converged = bool(info.get("converged", False))
            fallback_used = bool(info.get("fas_fallback_used", False) or info.get("newton_fallback_used", False))
            fallback_backend = info.get("fallback_backend")
            budget_summary = dict(info.get("budget_summary") or {})
            if converged and not budget_summary:
                # Some backends (e.g. a Picard fallback solve) do not attach a
                # budget; evaluate it here so acceptance and records stay
                # uniform across backends.
                budget_summary = _timestep_budget_summary(
                    model,
                    head=head_trial,
                    head_prev=head_accepted,
                    dt=dt_try,
                    k=k,
                    bottom=bottom,
                    top=top,
                    active=active_i,
                    sy=sy_f,
                    ss=ss_f,
                    min_sat=min_sat,
                )
            _pct_raw = budget_summary.get("percent_discrepancy")
            pct_discrepancy = abs(float(_pct_raw)) if _pct_raw is not None else np.inf
            budget_ok = bool(converged and pct_discrepancy <= budget_tol_pct)
            accepted = bool(converged and budget_ok)
            failure_reason = None
            if not accepted:
                failure_reason = (
                    info.get("fas_failure_reason")
                    or info.get("newton_failure_reason")
                    or ("budget_discrepancy" if converged else "nonconvergence")
                )

            cycle_history = info.get("fas_cycle_history") or []
            last_cycle = cycle_history[-1] if cycle_history else {}
            record = {
                "simulation_time": simulation_time + (dt_try if accepted else 0.0),
                "stress_period_index": int(period_index),
                "timestep_index": int(timestep_index),
                "dt": float(dt_try),
                "accepted": bool(accepted),
                "converged": bool(converged),
                "backend_attempted": str(backend_name),
                "backend_used": str(fallback_backend) if fallback_used and fallback_backend else str(backend_name),
                "fallback_used": bool(fallback_used),
                "fallback_backend": fallback_backend,
                "fallback_state": info.get("fallback_state", "not_used"),
                "failure_reason": failure_reason,
                "retry_count": int(retries_for_timestep),
                "solve_seconds": float(solve_seconds),
                "nonlinear_residual_rms": info.get("true_nonlinear_residual_rms"),
                "head_equivalent_residual_rms": info.get("head_equivalent_residual_rms"),
                "fas_cycles": info.get("fas_cycles"),
                "newton_iterations": info.get("newton_iterations"),
                "smoothing_sweeps_by_level": {
                    "pre": info.get("pre_smoothing_sweeps_by_level"),
                    "post": info.get("post_smoothing_sweeps_by_level"),
                    "coarse": info.get("coarse_smoothing_sweeps_by_level"),
                },
                "tau_norms": last_cycle.get("tau_norms"),
                "rejected_corrections": info.get("rejected_corrections"),
                "damped_corrections": info.get("damped_corrections"),
                "storage_in": budget_summary.get("storage_in"),
                "storage_out": budget_summary.get("storage_out"),
                "budget_summary": budget_summary,
                "budget_percent_discrepancy": None if not np.isfinite(pct_discrepancy) else float(pct_discrepancy),
                "previous_head": head_accepted.copy() if save_diagnostics_b else None,
                "accepted_head": np.asarray(head_trial, dtype=np.float64).copy() if (accepted and save_diagnostics_b) else None,
            }

            if accepted:
                simulation_time += dt_try
                remaining_dt = max(0.0, remaining_dt - dt_try)
                record["simulation_time"] = float(simulation_time)
                timestep_records.append(record)
                timestep_index += 1
                counters["experimental_timestep_count"] += 1
                if fallback_used:
                    fallback_this_period += 1
                    counters["experimental_fallback_timestep_count"] += 1
                head_accepted = np.asarray(head_trial, dtype=np.float64)
                retries_for_timestep = 0
                if (
                    grow_factor > 1.0
                    and growth_steps < max_growth_steps
                    and current_dt < (min(period_dt, max_dt) if max_dt is not None else period_dt) - 1.0e-12
                ):
                    cap = min(period_dt, max_dt) if max_dt is not None else period_dt
                    current_dt = min(cap, current_dt * grow_factor)
                    growth_steps += 1
                continue

            # -------- rejected attempt: never overwrite the accepted head --------
            timestep_records.append(record)
            retries_for_timestep += 1
            retries_this_period += 1
            rejected_this_period += 1
            counters["experimental_retry_count"] += 1
            counters["experimental_rejected_attempt_count"] += 1
            at_dt_min = dt_try <= dt_min + max(1.0e-12, period_dt * 1.0e-12)
            if retry_enabled and not at_dt_min and retries_for_timestep <= max_retries:
                current_dt = max(dt_min, dt_try * shrink_factor)
                growth_steps = 0
                continue
            if allow_unaccepted:
                # Forced acceptance at dt_min, explicitly flagged.
                record_forced = dict(record)
                record_forced.update(
                    {
                        "accepted": True,
                        "forced_accept": True,
                        "simulation_time": float(simulation_time + dt_try),
                        "accepted_head": np.asarray(head_trial, dtype=np.float64) if save_diagnostics_b else None,
                    }
                )
                simulation_time += dt_try
                remaining_dt = max(0.0, remaining_dt - dt_try)
                timestep_records.append(record_forced)
                timestep_index += 1
                counters["experimental_timestep_count"] += 1
                counters["experimental_forced_accept_count"] += 1
                head_accepted = np.asarray(head_trial, dtype=np.float64)
                retries_for_timestep = 0
                continue
            raise RuntimeError(
                f"{backend_name} failed stress period {period_index + 1}, timestep "
                f"{timestep_index + 1} at dt_min={dt_min:.6g} (reason: {failure_reason}; "
                f"budget discrepancy {pct_discrepancy:.6g}% vs limit {budget_tol_pct:.6g}%). "
                "The last accepted head was preserved; the failed trial head was discarded."
            )

        # ---------------- period acceptance and outputs ----------------
        period_records = timestep_records[period_record_start:]
        accepted_records = [rec for rec in period_records if rec.get("accepted")]
        period_budget = _merge_timestep_budgets(period_records)
        period_budgets.append(period_budget)
        period_time = time.perf_counter() - period_t0
        period_times[period_index] = period_time
        heads_per_period[period_index] = head_accepted

        period_converged = all(bool(rec.get("converged")) for rec in accepted_records) and not any(
            rec.get("forced_accept") for rec in accepted_records
        )
        period_info = {
            "period": int(period_index + 1),
            "solver_type": f"experimental_{backend_name}_transient",
            "solver_backend": str(backend_name),
            "experimental_backend": True,
            "converged": bool(period_converged),
            "experimental_acceptance_passed": bool(period_converged),
            "production_acceptance_passed": bool(period_converged),
            "strict_picard_convergence_passed": False,
            "practical_picard_acceptance_passed": False,
            "incremental_picard_enabled": False,
            "adaptive_dt_enabled": False,
            "experimental_timestep_count": int(timestep_index),
            "experimental_retry_count": int(retries_this_period),
            "experimental_rejected_attempt_count": int(rejected_this_period),
            "experimental_fallback_timestep_count": int(fallback_this_period),
            "experimental_substep_dts": [float(rec["dt"]) for rec in accepted_records],
            "adaptive_dt_substep_count": int(timestep_index),
            "adaptive_dt_retry_count": int(retries_this_period),
            "period_dt": float(period_dt),
            "period_total_seconds": float(period_time),
            "period_solve_seconds": float(period_solve_seconds),
            "budget_summary": period_budget,
            "budget_percent_discrepancy": float(abs(period_budget["percent_discrepancy"])),
            "final_residual": accepted_records[-1]["nonlinear_residual_rms"] if accepted_records else None,
            "true_nonlinear_residual_rms": accepted_records[-1]["nonlinear_residual_rms"] if accepted_records else None,
            "head_equivalent_residual_rms": accepted_records[-1]["head_equivalent_residual_rms"] if accepted_records else None,
            "storage_mode": "exact_nonlinear_previous_accepted_head",
            "storage_specific_storage_formulation": "exact_potential",
            "unconfined_storage_mode_2d": storage_mode,
            "storage_reference": storage_reference,
        }
        period_infos.append(period_info)

        if save_diagnostics_b:
            exact_storage_term, exact_sy_term, exact_ss_term = exact_unconfined_storage_terms(
                head_new=head_accepted,
                head_old=period_head_old,
                bottom=bottom,
                top=top,
                specific_yield=sy_f,
                specific_storage=ss_f,
                dt=period_dt,
            )
            heads_old_per_period[period_index] = period_head_old
            storage_reference_heads[period_index] = head_accepted
            storage_coeffs[period_index] = 0.0
            sy_coeffs[period_index] = 0.0
            ss_coeffs[period_index] = 0.0
            storage_terms[period_index] = exact_storage_term
            sy_terms[period_index] = exact_sy_term
            ss_terms[period_index] = exact_ss_term
            sy_crossing_terms[period_index] = exact_sy_term

    info_all = {
        "heads_per_period": heads_per_period,
        "heads_final": heads_per_period[-1],
        "period_infos": period_infos,
        "last_info": last_info,
        "period_times": period_times,
        "total_time": float(time.perf_counter() - total_t0),
        "n_periods": n_periods,
        "storage_reference": "exact_nonlinear_previous_accepted_head",
        "dt": float(np.mean(dt_array)),
        "dt_per_period": dt_array,
        "solve_controls": controls,
        "save_diagnostics": bool(save_diagnostics_b),
        "transient_replay_counters": counters,
        "experimental_backend": str(backend_name),
        "experimental_timestep_records": timestep_records,
        "experimental_period_budgets": period_budgets,
        "simulation_time": float(simulation_time),
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
    model._transient_replay_counters = dict(counters)
    return (heads_per_period, info_all) if return_info else heads_per_period

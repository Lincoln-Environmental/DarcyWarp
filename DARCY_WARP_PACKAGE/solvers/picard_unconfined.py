# SPDX-License-Identifier: AGPL-3.0-only
"""Unconfined Picard iteration using the shared geometric K-cycle backend."""

from __future__ import annotations

from typing import Any

from .base import SolverContext


def solve_unconfined_picard(*, model: Any, state: dict[str, Any]):
    """Trusted unconfined Picard branch using the extracted K-cycle backend."""
    from DARCY_WARP_PACKAGE import warped_darcy as kernel_module
    globals().update(kernel_module.__dict__)
    globals().update(state)
    self = model
    if K_field is None or zbot_field is None:
        raise ValueError("unconfined=True requires K_field and zbot_field.")
    if self.active_host is None or self.bc_mask_host is None or self.bc_values_host is None:
        raise RuntimeError("build_from_truth_inputs or build_from_fields must be called before solve.")

    ny0 = int(self.ny)
    nx0 = int(self.nx)
    shape0 = (ny0, nx0)

    K_arr = np.asarray(K_field, dtype=np.float64)
    zbot_arr = np.asarray(zbot_field, dtype=np.float64)
    if K_arr.shape != shape0:
        raise ValueError(f"K_field shape {K_arr.shape} expected {shape0}.")
    if zbot_arr.shape != shape0:
        raise ValueError(f"zbot_field shape {zbot_arr.shape} expected {shape0}.")
    if not np.all(np.isfinite(K_arr)) or np.any(K_arr < 0.0):
        raise ValueError("K_field must be finite and non-negative.")
    if not np.all(np.isfinite(zbot_arr)):
        raise ValueError("zbot_field must be finite.")

    ztop_arr = None
    if ztop_field is not None:
        ztop_arr = np.asarray(ztop_field, dtype=np.float64)
        if ztop_arr.shape != shape0:
            raise ValueError(f"ztop_field shape {ztop_arr.shape} expected {shape0}.")
        if not np.all(np.isfinite(ztop_arr)):
            raise ValueError("ztop_field must be finite.")

    min_sat = float(
        unconfined_min_sat
        if unconfined_min_sat is not None
        else (0.1 if min_saturated_thickness is None else min_saturated_thickness)
    )
    if min_sat <= 0.0 or not np.isfinite(min_sat):
        raise ValueError("min_saturated_thickness must be positive and finite.")

    max_outer = int(
        unconfined_max_picard_iter
        if unconfined_max_picard_iter is not None
        else (100 if max_outer_iterations is None else max_outer_iterations)
    )
    if max_outer < 1:
        raise ValueError("max_outer_iterations must be >= 1.")

    omega_current = float(unconfined_relax if unconfined_relax is not None else omega)
    omega_min_f = float(omega_min)
    omega_max_f = float(omega_max)
    if not (0.0 < omega_min_f <= omega_max_f):
        raise ValueError("omega_min and omega_max must satisfy 0 < omega_min <= omega_max.")
    omega_current = min(max(omega_current, omega_min_f), omega_max_f)

    hclose_f = float(
        unconfined_head_tol
        if unconfined_head_tol is not None
        else (1.0e-4 if hclose is None else hclose)
    )
    if hclose_f < 0.0 or not np.isfinite(hclose_f):
        raise ValueError("hclose must be non-negative and finite.")

    residual_floor_tol_f = float(residual_floor_tol) if residual_floor_tol is not None else None
    if residual_floor_tol_f is not None and residual_floor_tol_f < 0.0:
        raise ValueError("residual_floor_tol must be non-negative.")

    inner_head_residual_tol_f = float(
        inner_head_residual_tol if inner_head_residual_tol is not None else hclose_f
    )
    if inner_head_residual_tol_f < 0.0 or not np.isfinite(inner_head_residual_tol_f):
        raise ValueError("inner_head_residual_tol must be non-negative and finite.")

    inner_max_cycles_early = int(unconfined_inner_max_cycles_early)
    inner_max_cycles_middle = int(unconfined_inner_max_cycles_middle)
    inner_max_cycles_late = int(unconfined_inner_max_cycles_late)
    if min(inner_max_cycles_early, inner_max_cycles_middle, inner_max_cycles_late) < 1:
        raise ValueError("unconfined inner max cycles must be >= 1.")

    inner_late_dh_f = float(unconfined_inner_late_dh)
    inner_middle_dh_f = float(unconfined_inner_middle_dh)
    if inner_late_dh_f < 0.0 or inner_middle_dh_f < 0.0:
        raise ValueError("unconfined inner dh thresholds must be non-negative.")

    inner_forcing_eta_f = float(inner_forcing_eta)
    if inner_forcing_eta_f < 0.0 or inner_forcing_eta_f > 1.0:
        raise ValueError("inner_forcing_eta must be in [0, 1].")

    inner_head_residual_tol_min_f = float(
        inner_head_residual_tol_min if inner_head_residual_tol_min is not None else hclose_f
    )
    if inner_head_residual_tol_min_f < 0.0 or not np.isfinite(inner_head_residual_tol_min_f):
        raise ValueError("inner_head_residual_tol_min must be non-negative and finite.")

    inner_head_residual_tol_max_f = float(inner_head_residual_tol_max)
    if inner_head_residual_tol_max_f < inner_head_residual_tol_min_f:
        raise ValueError("inner_head_residual_tol_max must be >= inner_head_residual_tol_min.")

    inner_picard_scale_max_fraction_f = float(inner_picard_scale_max_fraction)
    if inner_picard_scale_max_fraction_f < 0.0 or inner_picard_scale_max_fraction_f > 1.0:
        raise ValueError("inner_picard_scale_max_fraction must be in [0, 1].")

    chebyshev_reset_factor_f = float(chebyshev_reset_factor)
    if chebyshev_reset_factor_f <= 1.0 or not np.isfinite(chebyshev_reset_factor_f):
        raise ValueError("chebyshev_reset_factor must be finite and > 1.")

    chebyshev_minor_increase_patience_i = int(chebyshev_minor_increase_patience)
    if chebyshev_minor_increase_patience_i < 0:
        raise ValueError("chebyshev_minor_increase_patience must be >= 0.")

    transmissivity_relaxation_enabled_b = bool(transmissivity_relaxation_enabled)
    T_relax_early_f = float(transmissivity_relaxation_early)
    T_relax_middle_f = float(transmissivity_relaxation_middle)
    T_relax_late_f = float(transmissivity_relaxation_late)
    if not all(0.0 <= v <= 1.0 for v in (T_relax_early_f, T_relax_middle_f, T_relax_late_f)):
        raise ValueError("transmissivity relaxation factors must be in [0, 1].")

    T_relax_middle_iter = int(transmissivity_relaxation_middle_iteration)
    T_relax_late_iter = int(transmissivity_relaxation_late_iteration)
    if T_relax_middle_iter < 1 or T_relax_late_iter < T_relax_middle_iter:
        raise ValueError("transmissivity relaxation iterations must satisfy 1 <= middle <= late.")

    startup_mode = str(unconfined_startup_mode).strip().lower()
    if startup_mode not in {"initial_head", "confined_pre_solve", "unconfined_pre_solve"}:
        raise ValueError(
            "unconfined_startup_mode must be 'initial_head', 'confined_pre_solve', "
            "or 'unconfined_pre_solve'."
        )
    unconfined_pre_solve_iterations_i = int(unconfined_pre_solve_iterations)
    if unconfined_pre_solve_iterations_i < 1 or not np.isfinite(unconfined_pre_solve_iterations_i):
        raise ValueError("unconfined_pre_solve_iterations must be a finite integer >= 1.")

    storage_reference_mode = str(storage_reference).strip().lower()
    if storage_reference_mode not in {"previous_period", "current_picard"}:
        raise ValueError("storage_reference must be 'previous_period' or 'current_picard'.")
    storage_mode_2d = None if unconfined_storage_mode_2d is None else str(unconfined_storage_mode_2d).strip().lower()
    current_picard_storage = bool(transient) and storage_reference_mode == "current_picard"
    if current_picard_storage:
        if storage_mode_2d != "mf6_convertible_secant_sy":
            raise ValueError(
                "current_picard storage requires unconfined_storage_mode_2d to be "
                "'mf6_convertible_secant_sy'."
            )
        if sy is None or ss is None:
            raise ValueError("current_picard storage requires sy and ss.")
        if ztop_arr is None:
            raise ValueError("current_picard storage requires ztop_field.")
        sy_f = float(sy)
        ss_f = float(ss)
        if sy_f < 0.0 or ss_f < 0.0 or not np.isfinite(sy_f) or not np.isfinite(ss_f):
            raise ValueError("sy and ss must be finite and non-negative.")
    else:
        sy_f = float("nan")
        ss_f = float("nan")

    max_update_f = float(max_head_change_per_outer_iteration)
    if max_update_f <= 0.0 or not np.isfinite(max_update_f):
        raise ValueError("max_head_change_per_outer_iteration must be positive and finite.")

    practical_picard_acceptance_enabled_b = bool(practical_picard_acceptance_enabled)
    min_practical_outer_iterations_i = int(min_practical_outer_iterations)
    if min_practical_outer_iterations_i < 1:
        raise ValueError("min_practical_outer_iterations must be >= 1.")
    practical_residual_tol_f = float(practical_residual_tol)
    practical_dh_rms_tol_f = float(practical_dh_rms_tol)
    practical_storage_diag_change_rms_tol_f = float(practical_storage_diag_change_rms_tol)
    if (
        practical_residual_tol_f < 0.0
        or practical_dh_rms_tol_f < 0.0
        or practical_storage_diag_change_rms_tol_f < 0.0
    ):
        raise ValueError("practical Picard tolerances must be non-negative.")
    save_transient_diagnostics_b = bool(save_transient_diagnostics)
    secant_sy_practical_mode = bool(
        practical_picard_acceptance_enabled_b
        and current_picard_storage
        and storage_mode_2d == "mf6_convertible_secant_sy"
    )

    initial_sat_f = float(initial_saturated_thickness)
    if initial_sat_f <= 0.0 or not np.isfinite(initial_sat_f):
        raise ValueError("initial_saturated_thickness must be positive and finite.")

    rejection_factor_f = float(chebyshev_rejection_factor)
    if rejection_factor_f <= 1.0 or not np.isfinite(rejection_factor_f):
        raise ValueError("chebyshev_rejection_factor must be finite and > 1.")

    active_mask = np.asarray(self.active_host, dtype=np.int32) != 0
    bc_mask0 = np.asarray(self.bc_mask_host, dtype=np.int32) != 0
    free_mask0 = active_mask & (~bc_mask0)
    bc_values0 = np.asarray(self.bc_values_host, dtype=NP_FLOAT)

    if initial_head is None:
        h_iter = (zbot_arr + max(initial_sat_f, min_sat)).astype(NP_FLOAT, copy=False)
    else:
        h_iter = np.asarray(initial_head, dtype=NP_FLOAT).copy()
        if h_iter.shape != shape0:
            raise ValueError(f"initial_head must have shape {shape0}, got {h_iter.shape}.")
    h_iter[bc_mask0] = bc_values0[bc_mask0]
    h_iter[~active_mask] = NP_FLOAT(0.0)
    if not np.all(np.isfinite(h_iter)):
        raise ValueError("initial head for unconfined solve must be finite.")

    kc_base_kwargs = {
        "nu_pre": int(nu_pre),
        "nu_post": int(nu_post),
        "nu_coarse": int(nu_coarse),        "omega": float(omega),
        "rel_tol": float(rel_tol),
        "abs_tol_min": float(abs_tol_min),
        "aq_thickness": aq_thickness,
        "gh_alpha": gh_alpha,
        "max_levels": int(max_levels),
        "check_every_no": int(check_every_no),
        "dh_rms_tol": dh_rms_tol,
        "dh_max_tol": dh_max_tol,
        "dh_max_factor": float(dh_max_factor),
        "min_coarse_cells": min_coarse_cells,
        "fallback_to_pcg": bool(fallback_to_pcg),
        "divergence_cycle_start": int(divergence_cycle_start),
        "divergence_residual_factor": float(divergence_residual_factor),
        "fallback_pcg_max_iter": fallback_pcg_max_iter,
        "fallback_pcg_history_every": fallback_pcg_history_every,
        "smoother": str(smoother_mode),
        "cheby_lambda_min": float(cheby_lambda_min),
        "cheby_lambda_max": float(cheby_lambda_max),
    }

    # Optional fast inner solver (steady confined face-array K-cycle) for the
    # per-outer linearised solves.  Steady only: the fast implementation
    # rejects transient solves, and the Picard T(h) update already refreshes
    # the fast face arrays in place via update_T_in_place -> _fast_faces_stale,
    # so captured graphs survive across outer iterations.
    inner_implementation_mode = (
        "classic"
        if inner_implementation is None
        else str(inner_implementation).strip().lower()
    )
    if inner_implementation_mode not in {"classic", "fast"}:
        raise ValueError(
            f"inner_implementation must be 'classic' or 'fast', got {inner_implementation!r}."
        )
    if inner_implementation_mode == "fast":
        if bool(transient):
            raise ValueError(
                "inner_implementation='fast' is steady only; use 'classic' "
                "(or the device transient fast path) for transient unconfined solves."
            )
        kc_base_kwargs["implementation"] = "fast"
        # The fast backend's Jacobi-block coarsest solve needs more sweeps
        # than the classic PCG default for equivalent contraction (the
        # validated fast configuration uses nu_coarse=10).
        kc_base_kwargs["nu_coarse"] = max(int(nu_coarse), 10)

    def _storage_from_picard_head(
            head_ref_arr: np.ndarray,
    ) -> dict[str, np.ndarray]:
        head_ref64 = np.asarray(head_ref_arr, dtype=np.float64)
        if ztop_arr is None:
            raise ValueError("ztop_field is required for current Picard storage.")
        full_thickness = np.maximum(ztop_arr - zbot_arr, min_sat)
        head_old64 = np.asarray(head_prev, dtype=np.float64)
        sat_ref_zero = np.clip(head_ref64 - zbot_arr, 0.0, full_thickness)
        sat_old_zero = np.clip(head_old64 - zbot_arr, 0.0, full_thickness)
        sat_ref_ss = np.clip(head_ref64 - zbot_arr, min_sat, full_thickness)
        sy_coeff = np.zeros(shape0, dtype=np.float64)
        ss_coeff = np.zeros(shape0, dtype=np.float64)
        dh_ref = head_ref64 - head_old64
        moving = np.abs(dh_ref) > 1.0e-12
        sy_coeff[moving] = sy_f * ((sat_ref_zero[moving] - sat_old_zero[moving]) / dh_ref[moving])
        fallback = (~moving) & (head_ref64 < ztop_arr) & (head_ref64 > zbot_arr)
        sy_coeff[fallback] = sy_f
        sy_coeff = np.clip(sy_coeff, 0.0, sy_f)
        ss_coeff[:, :] = ss_f * sat_ref_ss
        storage = sy_coeff + ss_coeff
        storage = storage.astype(NP_FLOAT, copy=False)
        storage[~free_mask0] = NP_FLOAT(0.0)
        sy_coeff = sy_coeff.astype(np.float64, copy=False)
        ss_coeff = ss_coeff.astype(np.float64, copy=False)
        sy_coeff[~free_mask0] = 0.0
        ss_coeff[~free_mask0] = 0.0
        sat_ref_zero = sat_ref_zero.astype(np.float64, copy=False)
        sat_old_zero = sat_old_zero.astype(np.float64, copy=False)
        sat_ref_ss = sat_ref_ss.astype(np.float64, copy=False)
        sat_ref_zero[~free_mask0] = 0.0
        sat_old_zero[~free_mask0] = 0.0
        sat_ref_ss[~free_mask0] = 0.0
        return {
            "storage": storage,
            "sy_coeff": sy_coeff,
            "ss_coeff": ss_coeff,
            "sat_ref_zero": sat_ref_zero,
            "sat_old_zero": sat_old_zero,
            "sat_ref_ss": sat_ref_ss,
            "full_thickness": full_thickness.astype(np.float64, copy=False),
            "head_ref": head_ref64.astype(np.float64, copy=False),
        }

    if startup_mode == "confined_pre_solve":
        sat_startup = h_iter.astype(np.float64, copy=False) - zbot_arr
        sat_startup = np.maximum(sat_startup, min_sat)
        if ztop_arr is not None:
            sat_cap = np.maximum(ztop_arr - zbot_arr, min_sat)
            sat_startup = np.minimum(sat_startup, sat_cap)
        T_startup = (K_arr * sat_startup).astype(NP_FLOAT, copy=False)
        T_startup[~active_mask] = NP_FLOAT(0.0)
        self.update_T_in_place(T_startup)
        storage_coeff_startup = (
            _storage_from_picard_head(h_iter)["storage"]
            if current_picard_storage
            else storage_coeff
        )

        h_startup = self.solve_multigrid_kcycle(
            max_cycles=int(max_cycles),
            initial_head=h_iter,
            return_info=False,
            unconfined=False,
            transient=transient,
            storage_coeff=storage_coeff_startup,
            dt=dt,
            head_prev=head_prev,
            refresh_diag_with_transient_storage=True,
            **kc_base_kwargs,
        )
        h_startup = np.asarray(h_startup, dtype=np.float64)
        h_startup = np.maximum(h_startup, zbot_arr + min_sat)
        if ztop_arr is not None:
            h_startup = np.minimum(h_startup, ztop_arr)
        h_startup[~active_mask] = 0.0
        h_startup[bc_mask0] = bc_values0[bc_mask0]
        if not np.all(np.isfinite(h_startup)):
            raise FloatingPointError("confined pre-solve produced non-finite heads.")
        h_iter = h_startup.astype(NP_FLOAT, copy=False)

    elif startup_mode == "unconfined_pre_solve":
        # Unconfined warm start: a small fixed number of Picard
        # sub-iterations that rebuild transmissivity from the current
        # head (unconfined linearisation), each solved as a linearised
        # K-cycle step. Unlike ``confined_pre_solve`` (one fixed-T
        # solve), this lets the saturated thickness relax before the
        # main transient Picard loop begins. The storage term follows
        # the same reference as the main solve (current Picard head when
        # ``storage_reference='current_picard'``).
        h_pre = np.asarray(h_iter, dtype=np.float64)
        if ztop_arr is not None:
            sat_cap_pre = np.maximum(ztop_arr - zbot_arr, min_sat)
        for _ in range(unconfined_pre_solve_iterations_i):
            sat_pre = np.maximum(h_pre - zbot_arr, min_sat)
            if ztop_arr is not None:
                sat_pre = np.minimum(sat_pre, sat_cap_pre)
            T_pre = (K_arr * sat_pre).astype(NP_FLOAT, copy=False)
            T_pre[~active_mask] = NP_FLOAT(0.0)
            self.update_T_in_place(T_pre)
            storage_coeff_pre = (
                _storage_from_picard_head(h_pre)["storage"]
                if current_picard_storage
                else storage_coeff
            )
            h_pre = self.solve_multigrid_kcycle(
                max_cycles=int(max_cycles),
                initial_head=h_pre,
                return_info=False,
                unconfined=False,
                transient=transient,
                storage_coeff=storage_coeff_pre,
                dt=dt,
                head_prev=head_prev,
                refresh_diag_with_transient_storage=True,
                **kc_base_kwargs,
            )
            h_pre = np.asarray(h_pre, dtype=np.float64)
            h_pre = np.maximum(h_pre, zbot_arr + min_sat)
            if ztop_arr is not None:
                h_pre = np.minimum(h_pre, ztop_arr)
            h_pre[~active_mask] = 0.0
            h_pre[bc_mask0] = bc_values0[bc_mask0]
            if not np.all(np.isfinite(h_pre)):
                raise FloatingPointError("unconfined pre-solve produced non-finite heads.")
        h_iter = h_pre.astype(NP_FLOAT, copy=False)

    cheb_weights = _chebyshev_update_weights(
        order=int(chebyshev_order),
        lambda_min_fraction=float(chebyshev_lambda_min_fraction),
    )
    previous_update = np.zeros(shape0, dtype=np.float64)
    previous_measure = float("inf")
    chebyshev_rejections = 0
    chebyshev_resets = 0
    inner_solve_failures = 0
    strict_inner_nonconvergence_count = 0
    unusable_inner_solve_count = 0
    practical_inner_acceptances = 0
    accepted_picard_update_count = 0
    outer_chebyshev_ready_count = 0
    outer_chebyshev_used_count = 0
    outer_chebyshev_reset_count = 0
    improvement_streak = 0
    minor_increase_count = 0
    final_residual = None
    final_h_rms_end = float("nan")
    final_inner_max_cycles = 0
    final_max_abs_head_change = float("nan")
    last_linear_info: dict = {}
    outer_history: list[dict] = []
    strict_picard_convergence_passed = False
    practical_picard_acceptance_passed = False
    production_acceptance_passed = False
    T_previous: np.ndarray | None = None
    T_relax = float("nan")
    previous_storage_diag_arr: np.ndarray | None = None
    max_storage_diag_change_max = 0.0
    max_storage_diag_change_rms = 0.0
    last_storage_coeff_array: np.ndarray | None = None
    last_sy_storage_coeff_array: np.ndarray | None = None
    last_ss_storage_coeff_array: np.ndarray | None = None
    last_storage_reference_head_array: np.ndarray | None = None

    def _to_finite(value):
        try:
            f = float(value)
            return f if np.isfinite(f) else None
        except Exception:
            return None

    for outer_idx in range(max_outer):
        if not np.isfinite(previous_measure):
            inner_max_cycles = inner_max_cycles_early
        elif previous_measure > inner_middle_dh_f:
            inner_max_cycles = inner_max_cycles_early
        elif previous_measure > inner_late_dh_f:
            inner_max_cycles = inner_max_cycles_middle
        else:
            inner_max_cycles = inner_max_cycles_late

        sat = h_iter.astype(np.float64, copy=False) - zbot_arr
        sat = np.maximum(sat, min_sat)
        if ztop_arr is not None:
            sat_cap = np.maximum(ztop_arr - zbot_arr, min_sat)
            sat = np.minimum(sat, sat_cap)
        if not np.all(np.isfinite(sat)) or np.any(sat <= 0.0):
            raise FloatingPointError("unconfined saturated thickness became invalid.")

        T_candidate = (K_arr * sat).astype(NP_FLOAT, copy=False)
        T_candidate[~active_mask] = NP_FLOAT(0.0)
        if not np.all(np.isfinite(T_candidate)):
            raise FloatingPointError("unconfined transmissivity became non-finite.")

        if transmissivity_relaxation_enabled_b and outer_idx > 0 and T_previous is not None:
            if outer_idx < T_relax_middle_iter:
                T_relax = T_relax_early_f
            elif outer_idx < T_relax_late_iter:
                T_relax = T_relax_middle_f
            else:
                T_relax = T_relax_late_f
            T_pic = (1.0 - T_relax) * T_previous + T_relax * T_candidate
        else:
            T_pic = T_candidate
            T_relax = float("nan")

        T_pic[~active_mask] = NP_FLOAT(0.0)
        if not np.all(np.isfinite(T_pic)):
            raise FloatingPointError("unconfined transmissivity became non-finite.")

        self.update_T_in_place(T_pic)
        T_previous = T_pic.copy()
        storage_sy_coeff_arr = None
        storage_ss_coeff_arr = None
        storage_reference_head_arr = None
        if current_picard_storage:
            storage_state = _storage_from_picard_head(h_iter)
            storage_coeff_inner = storage_state["storage"]
            storage_sy_coeff_arr = np.asarray(storage_state["sy_coeff"], dtype=np.float64)
            storage_ss_coeff_arr = np.asarray(storage_state["ss_coeff"], dtype=np.float64)
            storage_reference_head_arr = np.asarray(storage_state["head_ref"], dtype=np.float64)
        else:
            storage_coeff_inner = storage_coeff
        if storage_coeff_inner is not None:
            storage_inner_arr = np.asarray(storage_coeff_inner, dtype=np.float64)
            if storage_inner_arr.ndim == 0:
                storage_inner_arr = np.full(shape0, float(storage_inner_arr.reshape(())), dtype=np.float64)
            storage_inner_diag = storage_inner_arr * float(self.dx) * float(self.dx) / float(dt)
            storage_inner_free = storage_inner_diag[free_mask0]
            storage_coeff_free = storage_inner_arr[free_mask0]
            storage_diag_min = float(np.min(storage_inner_free)) if storage_inner_free.size else None
            storage_diag_max = float(np.max(storage_inner_free)) if storage_inner_free.size else None
            storage_diag_mean = float(np.mean(storage_inner_free)) if storage_inner_free.size else None
            storage_coeff_min = float(np.min(storage_coeff_free)) if storage_coeff_free.size else None
            storage_coeff_max = float(np.max(storage_coeff_free)) if storage_coeff_free.size else None
            storage_coeff_mean = float(np.mean(storage_coeff_free)) if storage_coeff_free.size else None
            if previous_storage_diag_arr is None:
                storage_diag_change_max = None
                storage_diag_change_rms = None
            else:
                storage_diag_delta = storage_inner_diag - previous_storage_diag_arr
                storage_diag_delta_free = storage_diag_delta[free_mask0]
                if storage_diag_delta_free.size > 0:
                    storage_diag_change_max = float(np.max(np.abs(storage_diag_delta_free)))
                    storage_diag_change_rms = float(
                        np.sqrt(np.mean(storage_diag_delta_free * storage_diag_delta_free))
                    )
                else:
                    storage_diag_change_max = 0.0
                    storage_diag_change_rms = 0.0
        else:
            storage_diag_min = None
            storage_diag_max = None
            storage_diag_mean = None
            storage_coeff_min = None
            storage_coeff_max = None
            storage_coeff_mean = None
            storage_diag_change_max = None
            storage_diag_change_rms = None

        head_lin, info_lin = self.solve_multigrid_kcycle(
            max_cycles=int(inner_max_cycles),
            initial_head=h_iter,
            return_info=True,
            unconfined=False,
            transient=transient,
            storage_coeff=storage_coeff_inner,
            dt=dt,
            head_prev=head_prev,
            # Rebuild hierarchy when period-dependent storage changes;
            # otherwise coarse MG levels can retain stale storage.
            refresh_diag_with_transient_storage=True,
            **kc_base_kwargs,
        )
        last_linear_info = dict(info_lin) if isinstance(info_lin, dict) else {}
        inner_converged = bool(last_linear_info.get("converged", False))

        h_lin = np.asarray(head_lin, dtype=np.float64)
        if h_lin.shape != shape0:
            raise RuntimeError(f"inner linear solve returned shape {h_lin.shape}, expected {shape0}.")

        picard_update = h_lin - h_iter.astype(np.float64, copy=False)

        # Dynamic inexact inner tolerance based on the Picard update scale.
        if np.any(free_mask0):
            picard_update_free_raw = picard_update[free_mask0]
        else:
            picard_update_free_raw = np.array([], dtype=np.float64)
        if picard_update_free_raw.size > 0 and np.all(np.isfinite(picard_update_free_raw)):
            picard_update_abs = np.abs(picard_update_free_raw)
            picard_update_max = float(np.max(picard_update_abs))
            picard_update_rms = float(np.sqrt(np.mean(picard_update_free_raw * picard_update_free_raw)))
            picard_scale = max(
                picard_update_rms,
                inner_picard_scale_max_fraction_f * picard_update_max,
            )
            inner_head_residual_tol_used = min(
                inner_head_residual_tol_max_f,
                max(inner_head_residual_tol_min_f, inner_forcing_eta_f * picard_scale),
            )
            inner_usable_fallback = False
        else:
            picard_update_max = 0.0
            picard_update_rms = 0.0
            picard_scale = 0.0
            inner_head_residual_tol_used = float(inner_head_residual_tol_min_f)
            inner_usable_fallback = True

        r_rms_end = _to_finite(last_linear_info.get("r_rms_end"))
        h_rms_end = _to_finite(last_linear_info.get("h_rms_end"))
        tol_abs_inner = _to_finite(last_linear_info.get("tol_abs"))
        dh_rms_lastcheck = _to_finite(last_linear_info.get("dh_rms_lastcheck"))
        inner_residual_converged = (
            r_rms_end is not None and tol_abs_inner is not None and r_rms_end <= tol_abs_inner
        )
        inner_head_change_converged = (
            dh_rms_lastcheck is not None and dh_rms_tol_f is not None and dh_rms_lastcheck <= dh_rms_tol_f
        )
        inner_practically_converged = (
            inner_head_change_converged
            and residual_floor_tol_f is not None
            and r_rms_end is not None
            and r_rms_end <= residual_floor_tol_f
        )
        inner_usable_for_picard = (
            inner_converged
            or inner_head_change_converged
            or (
                not inner_usable_fallback
                and h_rms_end is not None
                and np.isfinite(float(h_rms_end))
                and float(h_rms_end) <= inner_head_residual_tol_used
            )
        )

        if not inner_converged:
            strict_inner_nonconvergence_count += 1

        picard_update[bc_mask0] = 0.0
        picard_update[~active_mask] = 0.0

        chebyshev_used = False
        chebyshev_rejected = False
        chebyshev_reset = False
        clipped_update = False

        if not inner_usable_for_picard:
            unusable_inner_solve_count += 1
            inner_solve_failures += 1
            chebyshev_resets += 1
            chebyshev_reset = True
            previous_update.fill(0.0)
        else:
            if not inner_converged:
                practical_inner_acceptances += 1
            accepted_picard_update_count += 1

        outer_chebyshev_ready = (
            bool(chebyshev_enabled)
            and accepted_picard_update_count >= 2
            and len(cheb_weights) > 0
            and inner_usable_for_picard
        )
        use_cheb = outer_chebyshev_ready
        if outer_chebyshev_ready:
            outer_chebyshev_ready_count += 1
        if use_cheb:
            weight = float(cheb_weights[(outer_idx - 1) % len(cheb_weights)])
            alpha = min(max(omega_current * weight, omega_min_f), omega_max_f)
            beta = 0.2 * max(0.0, alpha - omega_current)
            proposed_update = alpha * picard_update + beta * previous_update
            chebyshev_used = True
        else:
            proposed_update = omega_current * picard_update

        clipped = np.clip(proposed_update, -max_update_f, max_update_f)
        clipped_update = bool(np.any(clipped != proposed_update))
        h_trial = h_iter.astype(np.float64, copy=False) + clipped
        h_trial[bc_mask0] = bc_values0[bc_mask0]
        h_trial[~active_mask] = 0.0

        if np.any(free_mask0):
            trial_dh = (h_trial - h_iter.astype(np.float64, copy=False))[free_mask0]
            trial_measure = float(np.max(np.abs(trial_dh)))
            trial_rms = float(np.sqrt(np.mean(trial_dh * trial_dh)))
        else:
            trial_measure = 0.0
            trial_rms = 0.0

        reject_cheb = False
        if chebyshev_used:
            if clipped_update or not np.all(np.isfinite(h_trial)):
                reject_cheb = True
            elif np.isfinite(previous_measure) and trial_measure > rejection_factor_f * previous_measure:
                reject_cheb = True

        if reject_cheb:
            chebyshev_rejected = True
            chebyshev_used = False
            chebyshev_rejections += 1
            chebyshev_resets += 1
            chebyshev_reset = True
            previous_update.fill(0.0)
            fallback_update = omega_current * picard_update
            clipped = np.clip(fallback_update, -max_update_f, max_update_f)
            clipped_update = bool(np.any(clipped != fallback_update))
            h_trial = h_iter.astype(np.float64, copy=False) + clipped
            h_trial[bc_mask0] = bc_values0[bc_mask0]
            h_trial[~active_mask] = 0.0
            if np.any(free_mask0):
                trial_dh = (h_trial - h_iter.astype(np.float64, copy=False))[free_mask0]
                trial_measure = float(np.max(np.abs(trial_dh)))
                trial_rms = float(np.sqrt(np.mean(trial_dh * trial_dh)))
            else:
                trial_measure = 0.0
                trial_rms = 0.0

        if not np.all(np.isfinite(h_trial)):
            chebyshev_resets += 1
            raise FloatingPointError("unconfined nonlinear update produced non-finite heads.")

        if np.isfinite(previous_measure) and trial_measure > rejection_factor_f * previous_measure:
            omega_current = max(omega_min_f, 0.5 * omega_current)
            improvement_streak = 0
        else:
            improvement_streak += 1
            if improvement_streak >= 3:
                omega_current = min(omega_max_f, 1.1 * omega_current)
                improvement_streak = 0

        previous_update[:, :] = clipped
        h_iter = h_trial.astype(NP_FLOAT, copy=False)
        final_max_abs_head_change = float(trial_measure)
        final_residual = last_linear_info.get("r_rms_end")
        final_h_rms_end = h_rms_end if h_rms_end is not None else float("nan")
        final_inner_max_cycles = int(inner_max_cycles)

        if clipped_update:
            chebyshev_resets += 1
            chebyshev_reset = True
            previous_update.fill(0.0)

        if bool(chebyshev_reset_on_residual_increase) and np.isfinite(previous_measure):
            if trial_measure > chebyshev_reset_factor_f * previous_measure:
                chebyshev_resets += 1
                chebyshev_reset = True
                previous_update.fill(0.0)
                minor_increase_count = 0
            elif trial_measure > previous_measure:
                minor_increase_count += 1
                if minor_increase_count > chebyshev_minor_increase_patience_i:
                    chebyshev_resets += 1
                    chebyshev_reset = True
                    previous_update.fill(0.0)
                    minor_increase_count = 0
            else:
                minor_increase_count = 0

        previous_measure = trial_measure

        if chebyshev_used:
            outer_chebyshev_used_count += 1
        if chebyshev_reset:
            outer_chebyshev_reset_count += 1

        if storage_diag_change_max is not None:
            max_storage_diag_change_max = max(max_storage_diag_change_max, float(storage_diag_change_max))
        if storage_diag_change_rms is not None:
            max_storage_diag_change_rms = max(max_storage_diag_change_rms, float(storage_diag_change_rms))

        outer_history.append(
            {
                "outer_iteration": int(outer_idx + 1),
                "inner_max_cycles_used": int(inner_max_cycles),
                "inner_converged": bool(inner_converged),
                "inner_head_change_converged": bool(inner_head_change_converged),
                "inner_usable_for_picard": bool(inner_usable_for_picard),
                "h_rms_end": float(h_rms_end) if h_rms_end is not None else None,
                "inner_head_residual_tol_used": float(inner_head_residual_tol_used),
                "picard_update_max": float(picard_update_max),
                "picard_update_rms": float(picard_update_rms),
                "picard_scale": float(picard_scale),
                "accepted_picard_update_count": int(accepted_picard_update_count),
                "omega": float(omega_current),
                "chebyshev_used": bool(chebyshev_used),
                "chebyshev_ready": bool(outer_chebyshev_ready),
                "chebyshev_rejected": bool(chebyshev_rejected),
                "chebyshev_reset": bool(chebyshev_reset),
                "trial_measure": float(trial_measure),
                "trial_rms": float(trial_rms),
                "previous_measure": float(previous_measure) if np.isfinite(previous_measure) else None,
                "clipped_update": bool(clipped_update),
                "accepted_update": bool(inner_usable_for_picard),
                "transmissivity_relaxation_used": None if np.isnan(T_relax) else float(T_relax),
                "max_abs_head_change": float(final_max_abs_head_change),
                "rms_head_change": float(trial_rms),
                "min_head": float(np.nanmin(h_iter[active_mask])) if np.any(active_mask) else float("nan"),
                "max_head": float(np.nanmax(h_iter[active_mask])) if np.any(active_mask) else float("nan"),
                "min_saturated_thickness": float(np.nanmin(sat[active_mask])) if np.any(active_mask) else float("nan"),
                "max_saturated_thickness": float(np.nanmax(sat[active_mask])) if np.any(active_mask) else float("nan"),
                "mean_saturated_thickness": float(np.nanmean(sat[free_mask0])) if np.any(free_mask0) else float("nan"),
                "min_transmissivity": float(np.nanmin(T_pic[active_mask])) if np.any(active_mask) else float("nan"),
                "max_transmissivity": float(np.nanmax(T_pic[active_mask])) if np.any(active_mask) else float("nan"),
                "storage_reference": str(storage_reference_mode),
                "unconfined_storage_mode_2d": storage_mode_2d,
                "storage_coeff_min": storage_coeff_min,
                "storage_coeff_max": storage_coeff_max,
                "storage_coeff_mean": storage_coeff_mean,
                "storage_diag_min": storage_diag_min,
                "storage_diag_max": storage_diag_max,
                "storage_diag_mean": storage_diag_mean,
                "storage_diag_change_max": storage_diag_change_max,
                "storage_diag_change_rms": storage_diag_change_rms,
                "inner_iterations": int(last_linear_info.get("n_cycles_used", 0)),
                "inner_residual": None if final_residual is None else float(final_residual),
            }
        )

        previous_storage_diag_arr = None if storage_coeff_inner is None else np.asarray(storage_inner_diag, dtype=np.float64).copy()
        if save_transient_diagnostics_b:
            last_storage_coeff_array = (
                None if storage_coeff_inner is None else np.asarray(storage_inner_arr, dtype=np.float64).copy()
            )
            last_sy_storage_coeff_array = (
                None if storage_sy_coeff_arr is None else np.asarray(storage_sy_coeff_arr, dtype=np.float64).copy()
            )
            last_ss_storage_coeff_array = (
                None if storage_ss_coeff_arr is None else np.asarray(storage_ss_coeff_arr, dtype=np.float64).copy()
            )
            last_storage_reference_head_array = (
                None
                if storage_reference_head_arr is None
                else np.asarray(storage_reference_head_arr, dtype=np.float64).copy()
            )

        head_change_converged = final_max_abs_head_change < hclose_f
        strict_picard_convergence_passed = bool(
            head_change_converged and (inner_usable_for_picard or accept_on_head_change_only)
        )
        practical_picard_acceptance_passed = False
        if secant_sy_practical_mode:
            practical_picard_acceptance_passed = bool(
                int(outer_idx + 1) >= min_practical_outer_iterations_i
                and final_residual is not None
                and np.isfinite(float(final_residual))
                and float(final_residual) <= practical_residual_tol_f
                and np.isfinite(float(trial_rms))
                and float(trial_rms) <= practical_dh_rms_tol_f
                and storage_diag_change_rms is not None
                and np.isfinite(float(storage_diag_change_rms))
                and float(storage_diag_change_rms) <= practical_storage_diag_change_rms_tol_f
            )
        production_acceptance_passed = bool(
            strict_picard_convergence_passed or practical_picard_acceptance_passed
        )
        if outer_history:
            outer_history[-1]["strict_picard_convergence_passed"] = bool(strict_picard_convergence_passed)
            outer_history[-1]["practical_picard_acceptance_passed"] = bool(practical_picard_acceptance_passed)
            outer_history[-1]["production_acceptance_passed"] = bool(production_acceptance_passed)
        # Diagnostic opt-in (default off): accept the Picard update on head
        # change alone, treating the inner-residual / inner_usable_for_picard
        # gate as a guardrail rather than a hard failure criterion. When False
        # this is identical to ``and inner_usable_for_picard``.
        if production_acceptance_passed:
            break

    final_sat = h_iter.astype(np.float64, copy=False) - zbot_arr
    final_sat = np.maximum(final_sat, min_sat)
    if ztop_arr is not None:
        final_sat = np.minimum(final_sat, np.maximum(ztop_arr - zbot_arr, min_sat))
    final_T = (K_arr * final_sat).astype(NP_FLOAT, copy=False)
    final_T[~active_mask] = NP_FLOAT(0.0)
    self.update_T_in_place(final_T)

    effectively_dry = active_mask & (h_iter.astype(np.float64, copy=False) <= zbot_arr + float(dry_cell_flag_threshold))
    info_out = dict(last_linear_info) if isinstance(last_linear_info, dict) else {}
    info_out.update(
            {
                "solver_type": "kcycle_unconfined_picard_chebyshev",
                "linear_solver_type": str(last_linear_info.get("solver_type", "kcycle")),
                "inner_implementation": str(inner_implementation_mode),
                "unconfined": True,
            "converged": bool(production_acceptance_passed),
            "outer_iterations": int(len(outer_history)),
            "chebyshev_enabled": bool(chebyshev_enabled),
            "chebyshev_order": int(chebyshev_order),
            "chebyshev_rejections": int(chebyshev_rejections),
            "chebyshev_resets": int(chebyshev_resets),
            "omega_final": float(omega_current),
            "min_saturated_thickness": float(min_sat),
            "max_head_change_per_outer_iteration": float(max_update_f),
            "final_max_abs_head_change": float(final_max_abs_head_change),
            "final_residual": None if final_residual is None else float(final_residual),
            "inner_solve_failures": int(inner_solve_failures),
            "strict_inner_nonconvergence_count": int(strict_inner_nonconvergence_count),
            "unusable_inner_solve_count": int(unusable_inner_solve_count),
            "practical_inner_acceptance_count": int(practical_inner_acceptances),
            "accepted_picard_update_count": int(accepted_picard_update_count),
            "outer_chebyshev_ready_count": int(outer_chebyshev_ready_count),
            "outer_chebyshev_used_count": int(outer_chebyshev_used_count),
            "outer_chebyshev_reset_count": int(outer_chebyshev_reset_count),
            "effectively_dry_cell_count": int(np.count_nonzero(effectively_dry)),
            "inner_forcing_eta": float(inner_forcing_eta_f),
            "inner_head_residual_tol_min": float(inner_head_residual_tol_min_f),
            "inner_head_residual_tol_max": float(inner_head_residual_tol_max_f),
            "nonlinear_convergence_basis": (
                "head_change_only"
                if bool(accept_on_head_change_only)
                else "head_change_and_inner_usable_for_picard"
            ),
            "accept_on_head_change_only": bool(accept_on_head_change_only),
            "residual_floor_tol": None if residual_floor_tol_f is None else float(residual_floor_tol_f),
            "inner_head_residual_tol": float(inner_head_residual_tol_f),
            "inner_residual_converged": bool(inner_residual_converged),
            "inner_head_change_converged": bool(inner_head_change_converged),
            "inner_practically_converged": bool(inner_practically_converged),
            "inner_usable_for_picard": bool(inner_usable_for_picard),
            "inner_h_rms_end": float(final_h_rms_end) if np.isfinite(final_h_rms_end) else None,
            "inner_max_cycles_used": int(final_inner_max_cycles),
            "outer_history": outer_history,
            "picard_converged": bool(strict_picard_convergence_passed),
            "strict_picard_convergence_passed": bool(strict_picard_convergence_passed),
            "practical_picard_acceptance_passed": bool(practical_picard_acceptance_passed),
            "production_acceptance_passed": bool(production_acceptance_passed),
            "practical_picard_acceptance_enabled": bool(secant_sy_practical_mode),
            "min_practical_outer_iterations": int(min_practical_outer_iterations_i),
            "practical_residual_tol": float(practical_residual_tol_f),
            "practical_dh_rms_tol": float(practical_dh_rms_tol_f),
            "practical_storage_diag_change_rms_tol": float(practical_storage_diag_change_rms_tol_f),
            "picard_n_iter_used": int(len(outer_history)),
            "picard_max_iter": int(max_outer),
            "picard_relax": float(omega_current),
            "picard_head_tol": float(hclose_f),
            "picard_dh_max_end": float(final_max_abs_head_change),
            "unconfined_min_sat": float(min_sat),
            "unconfined_startup_mode": str(startup_mode),
            "unconfined_pre_solve_iterations": int(unconfined_pre_solve_iterations_i),
            "storage_reference": str(storage_reference_mode),
            "unconfined_storage_mode_2d": storage_mode_2d,
            "max_storage_diag_change_max": float(max_storage_diag_change_max),
            "max_storage_diag_change_rms": float(max_storage_diag_change_rms),
            "save_transient_diagnostics": bool(save_transient_diagnostics_b),
            "diag_preconditioner_backend": self._diag_backend_env_or_default(),
            "update_T_profile_last": None if self._last_update_T_profile is None else dict(self._last_update_T_profile),
            "update_T_profile_totals": None if self._update_T_profile_totals is None else dict(self._update_T_profile_totals),
        }
    )
    if save_transient_diagnostics_b:
        info_out.update(
            {
                "storage_coeff_last_linearization_array": last_storage_coeff_array,
                "sy_storage_coeff_last_linearization_array": last_sy_storage_coeff_array,
                "ss_storage_coeff_last_linearization_array": last_ss_storage_coeff_array,
                "storage_reference_head_last_linearization_array": last_storage_reference_head_array,
            }
        )
    return (h_iter, info_out) if return_info else h_iter


class UnconfinedPicardKCycleBackend:
    """Trusted Picard implementation; linear work is supplied by K-cycle."""

    name = "unconfined_picard_kcycle"

    def solve(self, context: SolverContext, **kwargs: Any):
        kwargs["unconfined"] = True
        from DARCY_WARP_PACKAGE.solvers.multigrid_kcycle import (
            solve_multigrid_kcycle_backend,
        )

        return solve_multigrid_kcycle_backend(model=context.model, **kwargs)

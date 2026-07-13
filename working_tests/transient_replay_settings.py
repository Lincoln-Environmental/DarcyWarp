from __future__ import annotations

import math

import numpy as np

PRODUCTION_RUN_MODE = "production"
BENCHMARK_RUN_MODE = "benchmark"
DIAGNOSTICS_RUN_MODE = "diagnostics"
RUN_MODES = (PRODUCTION_RUN_MODE, BENCHMARK_RUN_MODE, DIAGNOSTICS_RUN_MODE)

MASS_BALANCE_EXCELLENT_PCT = 0.001
MASS_BALANCE_GOOD_PCT = 0.01
MASS_BALANCE_ACCEPTABLE_PCT = 0.1
MASS_BALANCE_STARTUP_WARN_PCT = 0.2
MASS_BALANCE_STARTUP_PERIOD = 1

HEAD_ACCURACY_CRITERIA = {
    "final_rmse_max": 0.001,
    "final_max_abs_diff_max": 0.005,
    "worst_period_rmse_max": 0.005,
    "worst_period_max_abs_diff_max": 0.02,
    "all_period_percent_within_0_01m_min": 99.9,
}

PRODUCTION_RUNTIME_TARGET_S = 30.0
PRODUCTION_RUNTIME_STRETCH_TARGET_S = 20.0

DEFAULT_MIN_SAT = 0.1
STORAGE_REFERENCE_PREVIOUS_PERIOD = "previous_period"
STORAGE_REFERENCE_CURRENT_PICARD = "current_picard"
STORAGE_REFERENCE_MODES = {
    STORAGE_REFERENCE_PREVIOUS_PERIOD,
    STORAGE_REFERENCE_CURRENT_PICARD,
}
STORAGE_TOP_THRESHOLD_GE = "ge"
STORAGE_TOP_THRESHOLD_GT = "gt"
STORAGE_TOP_THRESHOLD_MODES = {
    STORAGE_TOP_THRESHOLD_GE,
    STORAGE_TOP_THRESHOLD_GT,
}
STORAGE_ACTIVE_SET_NONE = "none"
STORAGE_ACTIVE_SET_HYSTERESIS = "hysteresis"
STORAGE_ACTIVE_SET_FREEZE_WHEN_STABLE = "freeze_when_stable"
STORAGE_ACTIVE_SET_PREDICTOR_CORRECTOR = "predictor_corrector"
STORAGE_ACTIVE_SET_STRATEGIES = {
    STORAGE_ACTIVE_SET_NONE,
    STORAGE_ACTIVE_SET_HYSTERESIS,
    STORAGE_ACTIVE_SET_FREEZE_WHEN_STABLE,
    STORAGE_ACTIVE_SET_PREDICTOR_CORRECTOR,
}

VALIDATED_METHOD_SETTINGS = {
    "unconfined_storage_mode": "mf6_convertible_secant_sy",
    "storage_reference": STORAGE_REFERENCE_CURRENT_PICARD,
    "storage_top_threshold": STORAGE_TOP_THRESHOLD_GE,
    "storage_active_set_strategy": STORAGE_ACTIVE_SET_NONE,
    "unconfined_startup_mode": "confined_pre_solve",
    "warm_start": "unconfined_steady_mf6",
}


def default_solve_controls() -> dict:
    """Default Warp solve controls for the transient replay (kcycle)."""
    return {
        "max_cycles": 200,
        "max_levels": 4,
        "min_coarse_cells": 500,
        "nu_pre": 1,
        "nu_post": 1,
        "nu_coarse": 1,
        "check_every_no": 1,
        "max_outer_iterations": 100,
        "hclose": 1.0e-4,
        "rel_tol": 5.0e-7,
        "abs_tol_min": 5.0e-7,
        "dh_rms_tol": 1.0e-4,
        "residual_floor_tol": 1.0e-4,
        "smoother": "chebyshev",
        "omega": 0.7,
        "omega_min": 0.1,
        "omega_max": 0.9,
        "chebyshev_enabled": True,
        "chebyshev_order": 3,
        "cheby_lambda_min": 0.1,
        "cheby_lambda_max": 2.0,
        "chebyshev_reset_factor": 1.2,
        "chebyshev_rejection_factor": 1.2,
        "inner_forcing_eta": 0.10,
        "inner_head_residual_tol_min": 2.5e-6,
        "inner_head_residual_tol_max": 2.0e-4,
        "inner_picard_scale_max_fraction": 0.10,
        "transmissivity_relaxation_enabled": False,
        "unconfined_startup_mode": "confined_pre_solve",
        "unconfined_pre_solve_iterations": 3,
        "min_saturated_thickness": DEFAULT_MIN_SAT,
        "initial_saturated_thickness": 100.0,
        "max_head_change_per_outer_iteration": 10.0,
        "storage_active_set_strategy": STORAGE_ACTIVE_SET_NONE,
        "storage_hysteresis_eps": 0.0,
        "storage_freeze_after_stable_iterations": 0,
        "storage_freeze_after_outer": None,
        "storage_switch_fraction_tol": 0.0,
        "predictor_max_outer_iterations": 5,
        "corrector_max_outer_iterations": 100,
        "predictor_corrector_corrector_strategy": STORAGE_ACTIVE_SET_NONE,
        "practical_picard_acceptance_enabled": True,
        "strict_head_residual_tol": 1.0e-6,
        "min_practical_outer_iterations": 8,
        "practical_head_residual_tol": 1.0e-5,
        "practical_residual_tol": 1.0e-5,
        "practical_dh_rms_tol": 3.0e-3,
        "practical_storage_diag_change_rms_tol": 30.0,
        "unconfined_inner_max_cycles_early": 10,
        "unconfined_inner_max_cycles_middle": 20,
        "unconfined_inner_max_cycles_late": 40,
        "unconfined_inner_middle_dh": 1.0,
        "unconfined_inner_late_dh": 1.0e-2,
        "adaptive_unconfined_inner_enabled": True,
        "adaptive_inner_initial_block_cycles": 5,
        "adaptive_inner_min_block_cycles": 5,
        "adaptive_inner_max_block_cycles": 20,
        "adaptive_inner_min_total_cycles": 5,
        "adaptive_inner_eta_initial": 0.05,
        "adaptive_inner_eta_min": 0.005,
        "adaptive_inner_eta_max": 0.10,
        "adaptive_inner_eta_gamma": 0.25,
        "adaptive_inner_eta_power": 1.5,
        "adaptive_inner_good_contraction_ratio": 0.40,
        "adaptive_inner_weak_contraction_ratio": 0.90,
        "adaptive_inner_stall_contraction_ratio": 0.9995,
        "adaptive_inner_divergence_contraction_ratio": 1.10,
        "adaptive_inner_stall_patience": 8,
        "adaptive_inner_minimum_usable_reduction_ratio": 0.10,
        "adaptive_inner_residual_floor": 1.0e-12,
        "adaptive_inner_relative_flow_residual_target": 1.0e-4,
        "adaptive_inner_save_block_history": False,
        "allow_unaccepted_transient_period": False,
        "use_device_transient_fast_path": True,
    }


def production_secant_sy_settings() -> dict:
    return {
        "solve_controls": default_solve_controls(),
        "unconfined_storage_mode": "mf6_convertible_secant_sy",
        "storage_reference": STORAGE_REFERENCE_CURRENT_PICARD,
        "storage_top_threshold": STORAGE_TOP_THRESHOLD_GE,
        "storage_active_set_strategy": STORAGE_ACTIVE_SET_NONE,
        "storage_freeze_after_outer": None,
        "warm_start_mode": "unconfined_steady_mf6",
    }


def secant_sy_freeze_settings(freeze_after_outer: int) -> dict:
    freeze_after_outer_i = int(freeze_after_outer)
    if freeze_after_outer_i < 1:
        raise ValueError("freeze_after_outer must be >= 1.")
    settings = production_secant_sy_settings()
    settings["storage_freeze_after_outer"] = freeze_after_outer_i
    return settings


def default_run_config(
    *,
    run_mode: str = PRODUCTION_RUN_MODE,
    device: str = "auto",
    compute_mass_balance: bool = True,
    profile_performance: bool = False,
    save_heavy_diagnostics: bool = False,
    run_replay_matrix: bool = False,
) -> dict:
    mode = str(run_mode).strip().lower()
    if mode not in RUN_MODES:
        raise ValueError(f"run_mode must be one of {RUN_MODES}.")
    return {
        "run_mode": mode,
        "device": str(device),
        "compute_mass_balance": bool(compute_mass_balance),
        "profile_performance": bool(profile_performance),
        "save_heavy_diagnostics": bool(save_heavy_diagnostics),
        "run_replay_matrix": bool(run_replay_matrix),
    }


def representative_recharge_rate(recharge_rates: np.ndarray) -> float:
    rates = np.asarray(recharge_rates, dtype=np.float64).reshape(-1)
    if rates.size == 0:
        raise ValueError("recharge_rates must contain at least one value.")
    if not np.all(np.isfinite(rates)):
        raise ValueError("recharge_rates must be finite.")
    return float(np.mean(rates))


def classify_period_mass_balance(percent_discrepancy: float, is_startup_period: bool) -> str:
    value = abs(float(percent_discrepancy or 0.0))
    if value < MASS_BALANCE_EXCELLENT_PCT:
        return "excellent"
    if value < MASS_BALANCE_GOOD_PCT:
        return "good"
    if is_startup_period:
        if value < MASS_BALANCE_STARTUP_WARN_PCT:
            return "startup_warning"
        return "fail"
    if value < MASS_BALANCE_ACCEPTABLE_PCT:
        return "acceptable"
    return "fail"


def classify_replay_mass_balance(
    mass_balance: dict,
    startup_period: int = MASS_BALANCE_STARTUP_PERIOD,
) -> dict:
    rows = mass_balance.get("per_period") or []
    cumulative = mass_balance.get("cumulative") or {}
    cumulative_pct = float(cumulative.get("percent_discrepancy", 0.0) or 0.0)
    warnings: list[str] = []
    failures: list[str] = []
    if not rows:
        return {
            "mass_balance_class": "fail",
            "mass_balance_passed": False,
            "cumulative_percent_discrepancy": cumulative_pct,
            "startup_percent_discrepancy": None,
            "worst_nonstartup_percent_discrepancy": None,
            "warnings": warnings,
            "failures": ["mass balance has no per-period rows"],
        }
    startup_row = next((row for row in rows if int(row.get("period", 0)) == int(startup_period)), None)
    startup_pct = abs(float(startup_row.get("percent_discrepancy", 0.0) or 0.0)) if startup_row is not None else 0.0
    nonstartup_values = [
        abs(float(row.get("percent_discrepancy", 0.0) or 0.0))
        for row in rows
        if int(row.get("period", 0)) != int(startup_period)
    ]
    worst_nonstartup = max(nonstartup_values) if nonstartup_values else 0.0
    cum_abs = abs(cumulative_pct)
    failed = False
    if cum_abs >= MASS_BALANCE_ACCEPTABLE_PCT:
        failed = True
        failures.append(
            f"cumulative percent discrepancy {cum_abs:.5g}% >= {MASS_BALANCE_ACCEPTABLE_PCT}%"
        )
    if nonstartup_values and worst_nonstartup >= MASS_BALANCE_GOOD_PCT:
        failed = True
        failures.append(
            f"a non-startup period has percent discrepancy {worst_nonstartup:.5g}% >= "
            f"{MASS_BALANCE_GOOD_PCT}%"
        )
    if startup_pct >= MASS_BALANCE_STARTUP_WARN_PCT:
        failed = True
        failures.append(
            f"startup period {startup_period} percent discrepancy {startup_pct:.5g}% >= "
            f"{MASS_BALANCE_STARTUP_WARN_PCT}%"
        )
    if failed:
        return {
            "mass_balance_class": "fail",
            "mass_balance_passed": False,
            "cumulative_percent_discrepancy": cumulative_pct,
            "startup_percent_discrepancy": startup_pct,
            "worst_nonstartup_percent_discrepancy": worst_nonstartup,
            "warnings": warnings,
            "failures": failures,
        }
    startup_elevated = (
        startup_pct >= MASS_BALANCE_GOOD_PCT
        and cum_abs < MASS_BALANCE_ACCEPTABLE_PCT
        and (not nonstartup_values or worst_nonstartup < MASS_BALANCE_GOOD_PCT)
    )
    if startup_elevated:
        warnings.append(
            f"Period {startup_period} has a slightly elevated mass-balance discrepancy "
            f"({startup_pct:.5g}%) during the confined_pre_solve / warm-start startup transient. "
            "Cumulative closure is acceptable and non-startup periods close tightly."
        )
        return {
            "mass_balance_class": "startup_warning",
            "mass_balance_passed": True,
            "cumulative_percent_discrepancy": cumulative_pct,
            "startup_percent_discrepancy": startup_pct,
            "worst_nonstartup_percent_discrepancy": worst_nonstartup,
            "warnings": warnings,
            "failures": failures,
        }
    worst_overall = max(abs(float(row.get("percent_discrepancy", 0.0) or 0.0)) for row in rows)
    if worst_overall < MASS_BALANCE_EXCELLENT_PCT:
        overall_class = "excellent"
    elif worst_overall < MASS_BALANCE_GOOD_PCT:
        overall_class = "good"
    else:
        overall_class = "acceptable"
    return {
        "mass_balance_class": overall_class,
        "mass_balance_passed": True,
        "cumulative_percent_discrepancy": cumulative_pct,
        "startup_percent_discrepancy": startup_pct,
        "worst_nonstartup_percent_discrepancy": worst_nonstartup,
        "warnings": warnings,
        "failures": failures,
    }


def annotate_mass_balance_classification(
    mass_balance: dict,
    startup_period: int = MASS_BALANCE_STARTUP_PERIOD,
) -> dict:
    classification = classify_replay_mass_balance(mass_balance, startup_period=startup_period)
    mass_balance["mass_balance_class"] = classification["mass_balance_class"]
    mass_balance["mass_balance_passed"] = classification["mass_balance_passed"]
    mass_balance["cumulative_percent_discrepancy"] = classification["cumulative_percent_discrepancy"]
    mass_balance["startup_percent_discrepancy"] = classification["startup_percent_discrepancy"]
    mass_balance["worst_nonstartup_percent_discrepancy"] = classification["worst_nonstartup_percent_discrepancy"]
    mass_balance["mass_balance_warnings"] = classification["warnings"]
    mass_balance["mass_balance_failures"] = classification["failures"]
    for row in mass_balance.get("per_period") or []:
        row["mass_balance_class"] = classify_period_mass_balance(
            row.get("percent_discrepancy", 0.0),
            is_startup_period=(int(row.get("period", 0)) == int(startup_period)),
        )
    for variant_key in ("mass_balance_linearized", "mass_balance_volume_sy"):
        variant = mass_balance.get(variant_key) or {}
        for row in variant.get("per_period") or []:
            row["mass_balance_class"] = classify_period_mass_balance(
                row.get("percent_discrepancy", 0.0),
                is_startup_period=(int(row.get("period", 0)) == int(startup_period)),
            )
    return mass_balance

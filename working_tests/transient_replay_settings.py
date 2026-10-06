from __future__ import annotations

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

from DARCY_WARP_PACKAGE.solvers.transient_config import (
    DEFAULT_MIN_SAT, STORAGE_REFERENCE_CURRENT_PICARD,
    default_solve_controls, production_secant_sy_settings,
)

VALIDATED_METHOD_SETTINGS = {
    "unconfined_storage_mode": "mf6_convertible_secant_sy",
    "storage_reference": STORAGE_REFERENCE_CURRENT_PICARD,
    "unconfined_startup_mode": "confined_pre_solve",
    "warm_start": "unconfined_steady_mf6",
}


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

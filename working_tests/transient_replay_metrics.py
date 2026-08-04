from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _head_metrics(warp_heads: np.ndarray, mf6_heads: np.ndarray, active: np.ndarray) -> dict:
    warp_heads = np.asarray(warp_heads, dtype=np.float64)
    mf6_heads = np.asarray(mf6_heads, dtype=np.float64)
    mask = (np.asarray(active) != 0) & np.isfinite(warp_heads) & np.isfinite(mf6_heads)
    diff = warp_heads - mf6_heads
    diff_masked = diff[mask]
    abs_diff = np.abs(diff_masked)
    return {
        "rmse": float(np.sqrt(np.mean(diff_masked * diff_masked))) if diff_masked.size else None,
        "max_abs_diff": float(np.max(abs_diff)) if abs_diff.size else None,
        "mean_bias_warp_minus_mf6": float(np.mean(diff_masked)) if diff_masked.size else None,
        "percent_within_0_01m": float(np.mean(abs_diff <= 0.01) * 100.0) if abs_diff.size else None,
        "percent_within_0_1m": float(np.mean(abs_diff <= 0.1) * 100.0) if abs_diff.size else None,
        "n_active": int(mask.sum()),
    }


def _field_stats(field: np.ndarray | None, active: np.ndarray) -> dict | None:
    if field is None:
        return None
    arr = np.asarray(field, dtype=np.float64)
    mask = np.asarray(active, dtype=np.int32) != 0
    vals = arr[mask]
    if vals.size == 0:
        return None
    return {
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
    }


def _sat_ref_summary(
    field: np.ndarray | None,
    source: str | None,
    active: np.ndarray,
) -> dict | None:
    stats = _field_stats(field, active)
    if stats is None or source is None:
        return None
    return {"source": source, **stats}


def compare_transient(
    warp_result: dict,
    mf6_heads_per_period: np.ndarray,
    mf6_heads_final: np.ndarray,
    active: np.ndarray,
) -> dict:
    warp_hpp = np.asarray(warp_result["heads_per_period"], dtype=np.float64)
    mf6_hpp = np.asarray(mf6_heads_per_period, dtype=np.float64)
    if warp_hpp.shape != mf6_hpp.shape:
        raise ValueError(f"per-period head shape mismatch: warp {warp_hpp.shape}, mf6 {mf6_hpp.shape}")
    per_period = [_head_metrics(warp_hpp[i], mf6_hpp[i], active) for i in range(warp_hpp.shape[0])]
    final = _head_metrics(warp_result["heads_final"], mf6_heads_final, active)
    max_abs_values = [m["max_abs_diff"] for m in per_period if m["max_abs_diff"] is not None]
    worst_period = int(np.argmax(max_abs_values)) if max_abs_values else None
    return {
        "per_period": per_period,
        "final": final,
        "worst_period_index_zero_based": worst_period,
        "worst_period_number_one_based": (None if worst_period is None else int(worst_period + 1)),
        "worst_period": (None if worst_period is None else int(worst_period + 1)),
    }


def save_summary(path: str | Path, summary: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(summary, f, indent=4, default=str)
    return path


def _scalar(artifact: dict, name: str) -> float | None:
    if name not in artifact:
        return None
    try:
        return float(np.asarray(artifact[name]).reshape(()))
    except (TypeError, ValueError):
        return None


def _summarize_last_info(info: dict) -> dict:
    if not isinstance(info, dict):
        return {}
    keys = (
        "converged",
        "outer_iterations",
        "formulation",
        "transient",
        "final_max_abs_head_change",
        "final_rms_head_change",
        "final_flow_residual_rms",
        "final_head_residual_rms",
        "final_residual",
        "chebyshev_rejections",
        "chebyshev_resets",
        "accepted_picard_update_count",
        "strict_inner_nonconvergence_count",
        "unusable_inner_solve_count",
        "practical_inner_acceptance_count",
        "effectively_dry_cell_count",
        "strict_picard_convergence_passed",
        "practical_picard_acceptance_passed",
        "production_acceptance_passed",
        "practical_picard_acceptance_enabled",
        "min_practical_outer_iterations",
        "practical_residual_tol",
        "practical_dh_rms_tol",
        "practical_storage_diag_change_rms_tol",
        "max_storage_diag_change_max",
        "max_storage_diag_change_rms",
        "storage_diag_change_max",
        "storage_diag_change_rms",
        "storage_specific_storage_formulation",
        "coarse_operator_mode",
        "coarse_krylov_method",
        "fine_operator_residual_checked",
        "total_inner_kcycles",
        "maximum_inner_kcycles_in_one_outer_iteration",
    )
    out = {}
    for key in keys:
        if key in info:
            value = info[key]
            if isinstance(value, (int, np.integer)):
                out[key] = int(value)
            elif isinstance(value, (float, np.floating)):
                out[key] = float(value)
            else:
                out[key] = value
    return out


def _summary_value(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value_f = float(value)
        return value_f if np.isfinite(value_f) else None
    if value is None:
        return None
    return value


def _summarize_period_infos(period_infos: list[dict]) -> dict:
    if not isinstance(period_infos, list):
        return {"n_periods": 0, "all_converged": False, "periods": []}
    keys = (
        "converged",
        "outer_iterations",
        "final_max_abs_head_change",
        "final_rms_head_change",
        "final_flow_residual_rms",
        "final_head_residual_rms",
        "final_residual",
        "picard_max_iter",
        "picard_dh_max_end",
        "strict_inner_nonconvergence_count",
        "unusable_inner_solve_count",
        "practical_inner_acceptance_count",
        "accepted_picard_update_count",
        "strict_picard_convergence_passed",
        "practical_picard_acceptance_passed",
        "production_acceptance_passed",
        "practical_picard_acceptance_enabled",
        "min_practical_outer_iterations",
        "practical_residual_tol",
        "practical_dh_rms_tol",
        "practical_storage_diag_change_rms_tol",
        "inner_usable_for_picard",
        "inner_h_rms_end",
        "inner_max_cycles_used",
        "total_inner_kcycles",
        "maximum_inner_kcycles_in_one_outer_iteration",
        "effectively_dry_cell_count",
        "max_storage_diag_change_max",
        "max_storage_diag_change_rms",
        "storage_diag_change_max",
        "storage_diag_change_rms",
        "storage_specific_storage_formulation",
        "coarse_operator_mode",
        "coarse_krylov_method",
        "fine_operator_residual_checked",
        # Device fast path / implementation-activity reporting (consumed by the
        # sanity matrix gates; without these the matrix silently reads zeros).
        "solver_type",
        "device_side_picard_fast_path_active",
        "transient_face_operator",
        "transient_face_graphs",
        "transient_mixed_precision",
        "transient_face_kcycle_graph_count",
        "transient_face_refresh_graph_count",
        "transient_face_kcycle_graph_fallback_count",
        "transient_face_refresh_graph_fallback_count",
        "adaptive_dt_enabled",
        "adaptive_dt_retry_count",
        "adaptive_dt_substep_count",
        "T_update_seconds",
        "storage_kernel_seconds",
        "fine_m_inv_refresh_seconds",
        "dynamic_coarse_refresh_seconds",
        "rhs_assembly_seconds",
        "inner_solver_seconds",
        "outer_convergence_check_seconds",
        "final_nonlinear_residual_check_seconds",
        "head_download_seconds",
        "period_total_seconds",
    )
    periods = []
    first_nonconverged_period = None
    for period_index, info in enumerate(period_infos, start=1):
        period_summary = {"period": int(period_index)}
        if isinstance(info, dict):
            for key in keys:
                if key in info:
                    period_summary[key] = _summary_value(info[key])
        production_accepted = bool(period_summary.get("production_acceptance_passed", period_summary.get("converged", False)))
        strict_converged = bool(period_summary.get("strict_picard_convergence_passed", False))
        practical_accepted = bool(period_summary.get("practical_picard_acceptance_passed", False) or production_accepted)
        period_summary["converged"] = production_accepted
        period_summary["strict_converged"] = strict_converged
        period_summary["practical_accepted"] = practical_accepted
        if not production_accepted and first_nonconverged_period is None:
            first_nonconverged_period = int(period_index)
        period_summary["diagnosis"] = "stable"
        periods.append(period_summary)
    strict_fail_period = None
    practical_fail_period = None
    for period in periods:
        if strict_fail_period is None and not bool(period.get("strict_converged", False)):
            strict_fail_period = int(period["period"])
        if practical_fail_period is None and not bool(period.get("practical_accepted", False)):
            practical_fail_period = int(period["period"])
    return {
        "n_periods": int(len(periods)),
        "all_converged": first_nonconverged_period is None and len(periods) > 0,
        "strict_all_converged": strict_fail_period is None and len(periods) > 0,
        "practical_all_accepted": practical_fail_period is None and len(periods) > 0,
        "production_accepted": first_nonconverged_period is None and len(periods) > 0,
        "first_nonconverged_period": first_nonconverged_period,
        "first_strict_nonconverged_period": strict_fail_period,
        "first_practical_nonaccepted_period": practical_fail_period,
        "periods": periods,
    }


def _summarize_period_head_stats(
    *,
    heads_per_period: np.ndarray,
    active: np.ndarray,
    bc_mask: np.ndarray,
) -> list[dict]:
    heads = np.asarray(heads_per_period, dtype=np.float64)
    active_i = np.asarray(active, dtype=np.int32)
    bc_i = np.asarray(bc_mask, dtype=np.int32)
    free = (active_i != 0) & (bc_i == 0)
    summaries: list[dict] = []
    for period_index in range(heads.shape[0]):
        period_heads = heads[period_index]
        mask = free & np.isfinite(period_heads)
        values = period_heads[mask]
        if values.size == 0:
            summaries.append(
                {
                    "period": int(period_index + 1),
                    "n_free": 0,
                    "min": None,
                    "max": None,
                    "mean": None,
                    "std": None,
                }
            )
            continue
        summaries.append(
            {
                "period": int(period_index + 1),
                "n_free": int(values.size),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
        )
    return summaries

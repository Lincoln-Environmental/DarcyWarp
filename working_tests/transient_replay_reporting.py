from __future__ import annotations

import math

import numpy as np
from working_tests.transient_replay_settings import (
    HEAD_ACCURACY_CRITERIA,
    PRODUCTION_RUNTIME_STRETCH_TARGET_S,
    PRODUCTION_RUNTIME_TARGET_S,
    VALIDATED_METHOD_SETTINGS,
)


def evaluate_head_accuracy(comparison: dict) -> dict:
    per_period = comparison.get("per_period") or []
    final = comparison.get("final") or {}
    final_rmse = float(final.get("rmse", math.inf) or math.inf)
    final_max_abs = float(final.get("max_abs_diff", math.inf) or math.inf)
    worst_rmse = max((float(row.get("rmse", 0.0) or 0.0) for row in per_period), default=math.inf)
    worst_max_abs = max((float(row.get("max_abs_diff", 0.0) or 0.0) for row in per_period), default=math.inf)
    min_percent_within = min((float(row.get("percent_within_0_01m", 0.0) or 0.0) for row in per_period), default=0.0)
    criteria = HEAD_ACCURACY_CRITERIA
    checks = {
        "final_rmse_lt_0p001": final_rmse < criteria["final_rmse_max"],
        "final_max_abs_lt_0p005": final_max_abs < criteria["final_max_abs_diff_max"],
        "worst_period_rmse_lt_0p005": worst_rmse < criteria["worst_period_rmse_max"],
        "worst_period_max_abs_lt_0p02": worst_max_abs < criteria["worst_period_max_abs_diff_max"],
        "all_period_percent_within_0p01m_ge_99p9": min_percent_within >= criteria["all_period_percent_within_0_01m_min"],
    }
    return {
        "passed": all(checks.values()),
        "final_rmse": final_rmse,
        "final_max_abs_diff": final_max_abs,
        "worst_period_rmse": worst_rmse,
        "worst_period_max_abs_diff": worst_max_abs,
        "all_period_percent_within_0_01m_min": min_percent_within,
        "criteria": criteria,
        "checks": checks,
    }


def evaluate_method_settings(
    *,
    unconfined_storage_mode: str,
    storage_reference: str,
    unconfined_startup_mode: str,
    warm_start: str,
) -> dict:
    actual = {
        "unconfined_storage_mode": str(unconfined_storage_mode),
        "storage_reference": str(storage_reference),
        "unconfined_startup_mode": str(unconfined_startup_mode),
        "warm_start": str(warm_start),
    }
    mismatches = {
        key: {"expected": VALIDATED_METHOD_SETTINGS[key], "actual": actual[key]}
        for key in VALIDATED_METHOD_SETTINGS
        if actual[key] != VALIDATED_METHOD_SETTINGS[key]
    }
    return {"passed": not mismatches, "settings": actual, "mismatches": mismatches}


def build_production_acceptance(
    *,
    method_settings: dict,
    head_accuracy: dict,
    mass_balance: dict,
    period_convergence: dict,
) -> dict:
    warnings: list[str] = []
    failures: list[str] = []
    method_valid = bool(method_settings["passed"])
    head_passed = bool(head_accuracy["passed"])
    mass_passed = bool(mass_balance.get("mass_balance_passed", False))
    strict_passed = bool(period_convergence.get("strict_all_converged", False))
    practical_passed = bool(period_convergence.get("practical_all_accepted", False))
    if not method_valid:
        failures.append("method settings do not match the validated secant-Sy method")
    if not head_passed:
        failures.append("head-accuracy target not met")
    if not mass_passed:
        failures.append(f"mass balance failed (class={mass_balance.get('mass_balance_class')})")
    if not practical_passed:
        failures.append("practical Picard production acceptance failed for at least one period")
    if not strict_passed:
        first_strict_fail = period_convergence.get("first_strict_nonconverged_period")
        warnings.append(
            "strict Picard convergence failed"
            + (f" first in period {first_strict_fail}" if first_strict_fail else "")
            + "; practical production acceptance is the production gate"
        )
    for warning in (mass_balance.get("mass_balance_warnings") or []):
        warnings.append(warning)
    production_passed = method_valid and head_passed and mass_passed and practical_passed
    return {
        "method_settings_valid": method_valid,
        "head_accuracy_passed": head_passed,
        "mass_balance_passed": mass_passed,
        "strict_picard_convergence_passed": strict_passed,
        "practical_picard_acceptance_passed": practical_passed,
        "production_acceptance_passed": production_passed,
        "warnings": warnings,
        "failures": failures,
    }


def build_performance_summary(
    *,
    timing: dict,
    period_convergence: dict,
    solve_settings: dict,
    mass_balance_runtime: float | None,
    profile: dict | None,
) -> dict:
    warp_total_time = float(timing.get("warp_total_time", 0.0) or 0.0)
    mf6_transient_total_time = timing.get("mf6_transient_total_time")
    mf6_including_warm_start = timing.get("mf6_engine_time_including_warm_start")

    def _speedup(reference):
        try:
            reference_f = float(reference)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(reference_f) or reference_f <= 0.0 or warp_total_time <= 0.0:
            return None
        return reference_f / warp_total_time

    periods = period_convergence.get("periods") or []
    outer_iterations = [int(p.get("outer_iterations", 0) or 0) for p in periods if isinstance(p, dict)]
    total_outer_iterations = int(sum(outer_iterations)) if outer_iterations else 0
    period_1_outer_iterations = int(outer_iterations[0]) if outer_iterations else 0
    summary = {
        "warp_total_time": warp_total_time,
        "mf6_transient_total_time": mf6_transient_total_time,
        "mf6_engine_time_including_warm_start": mf6_including_warm_start,
        "speedup_vs_mf6_transient": _speedup(mf6_transient_total_time),
        "speedup_vs_mf6_including_warm_start": _speedup(mf6_including_warm_start),
        "period_1_runtime": timing.get("warp_period_1_time"),
        "period_runtime_mean": timing.get("warp_period_time_mean"),
        "period_runtime_mean_excluding_period_1": timing.get("warp_period_time_mean_excluding_period_1"),
        "period_runtime_max": timing.get("warp_period_time_max"),
        "period_runtime_sum": timing.get("warp_period_time_sum"),
        "total_outer_iterations": total_outer_iterations,
        "period_1_outer_iterations": period_1_outer_iterations,
        "selected_practical_stopping_setting": {
            "practical_picard_acceptance_enabled": solve_settings.get("practical_picard_acceptance_enabled"),
            "min_practical_outer_iterations": solve_settings.get("min_practical_outer_iterations"),
            "practical_residual_tol": solve_settings.get("practical_residual_tol"),
            "practical_dh_rms_tol": solve_settings.get("practical_dh_rms_tol"),
            "practical_storage_diag_change_rms_tol": solve_settings.get("practical_storage_diag_change_rms_tol"),
            "max_outer_iterations": solve_settings.get("max_outer_iterations"),
        },
        "selected_inner_solve_profile": {
            "nu_pre": solve_settings.get("nu_pre"),
            "nu_post": solve_settings.get("nu_post"),
            "nu_coarse": solve_settings.get("nu_coarse"),
            "omega": solve_settings.get("omega"),
            "max_cycles": solve_settings.get("max_cycles"),
            "smoother": solve_settings.get("smoother"),
        },
        "mass_balance_runtime": mass_balance_runtime,
        "runtime_target_s": PRODUCTION_RUNTIME_TARGET_S,
        "runtime_stretch_target_s": PRODUCTION_RUNTIME_STRETCH_TARGET_S,
        "runtime_target_met": warp_total_time <= PRODUCTION_RUNTIME_TARGET_S if warp_total_time > 0.0 else False,
    }
    if profile is not None:
        summary["profile_available"] = True
        summary["profile"] = profile
    else:
        summary["profile_available"] = False
        summary["profile_reason"] = "category timing not yet instrumented"
    return summary


def _fmt_optional(value, spec: str = ".3g") -> str:
    if value is None:
        return "n/a"
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def print_mass_balance_table(mass_balance: dict) -> None:
    rows = mass_balance.get("per_period") or []
    if not rows:
        print("\nMass balance table, Warp: not_available")
        return
    print("\nMass balance table, Warp, preferred storage budget")
    print(
        "period recharge_in recharge_out chd_in chd_out ghb_in ghb_out "
        "storage_in storage_out total_in total_out discrepancy_pct"
    )
    for row in rows:
        print(
            f"{int(row.get('period', 0)):>6d} "
            f"{float(row.get('recharge_in', 0.0) or 0.0):>11.6g} "
            f"{float(row.get('recharge_out', 0.0) or 0.0):>12.6g} "
            f"{float(row.get('chd_in', 0.0) or 0.0):>7.6g} "
            f"{float(row.get('chd_out', 0.0) or 0.0):>8.6g} "
            f"{float(row.get('ghb_in', 0.0) or 0.0):>7.6g} "
            f"{float(row.get('ghb_out', 0.0) or 0.0):>8.6g} "
            f"{float(row.get('storage_in', 0.0) or 0.0):>10.6g} "
            f"{float(row.get('storage_out', 0.0) or 0.0):>11.6g} "
            f"{float(row.get('total_in', 0.0) or 0.0):>8.6g} "
            f"{float(row.get('total_out', 0.0) or 0.0):>9.6g} "
            f"{float(row.get('percent_discrepancy', 0.0) or 0.0):>15.6g}"
        )


def print_cumulative_mass_balance(mass_balance: dict) -> None:
    cumulative = mass_balance.get("cumulative") or {}
    if not cumulative:
        print("\nCumulative mass balance: not_available")
        return
    print("\nCumulative mass balance")
    keys = (
        "recharge_in_total",
        "recharge_out_total",
        "chd_in_total",
        "chd_out_total",
        "ghb_in_total",
        "ghb_out_total",
        "storage_in_total",
        "storage_out_total",
        "total_in_total",
        "total_out_total",
        "in_minus_out_total",
        "percent_discrepancy",
    )
    for key in keys:
        value = cumulative.get(key)
        print(f"  {key}: {_fmt_optional(value, '.8g')}")


def _print_production_report(*, summary: dict) -> None:
    acceptance = summary.get("production_acceptance", {}) or {}
    performance = summary.get("performance", {}) or {}
    head = summary.get("head_accuracy", {}) or {}
    mass_balance = summary.get("mass_balance", {}) or {}
    print("\nProduction acceptance")
    print(
        f"  method settings valid: "
        f"{'PASS: method settings valid' if acceptance.get('method_settings_valid') else 'FAIL: method settings invalid'}"
    )
    print(
        f"  head accuracy target:  "
        f"{'PASS: head accuracy target met' if acceptance.get('head_accuracy_passed') else 'FAIL: head accuracy target not met'} "
        f"(final rmse={_fmt_optional(head.get('final_rmse'))}, "
        f"max_abs={_fmt_optional(head.get('final_max_abs_diff'))}, "
        f"worst rmse={_fmt_optional(head.get('worst_period_rmse'))}, "
        f"worst max_abs={_fmt_optional(head.get('worst_period_max_abs_diff'))}, "
        f"min %within0.01m={_fmt_optional(head.get('all_period_percent_within_0_01m_min'), '.4g')})"
    )
    if acceptance.get("strict_picard_convergence_passed"):
        print("  strict Picard convergence: PASS")
    else:
        first_strict = (summary.get("period_convergence") or {}).get("first_strict_nonconverged_period")
        suffix = f" first in period {first_strict}" if first_strict else ""
        print(f"  strict Picard convergence: WARNING: strict Picard convergence failed but practical acceptance passed{suffix}")
    print(
        f"  practical production convergence: "
        f"{'PASS: practical production convergence accepted' if acceptance.get('practical_picard_acceptance_passed') else 'FAIL: practical production convergence not accepted'}"
    )
    mass_class = mass_balance.get("mass_balance_class")
    if acceptance.get("mass_balance_passed"):
        if mass_class == "startup_warning":
            print(f"  mass balance: PASS: mass balance acceptable with startup warning (class={mass_class})")
        else:
            print(f"  mass balance: PASS (class={mass_class})")
    else:
        print(f"  mass balance: FAIL (class={mass_class})")
    print(
        f"  production accepted: "
        f"{'PASS: production run accepted' if acceptance.get('production_acceptance_passed') else 'FAIL: production run rejected'}"
    )
    for warning in acceptance.get("warnings", []):
        print(f"  WARNING: {warning}")
    for failure in acceptance.get("failures", []):
        print(f"  FAILURE: {failure}")
    print("\nPerformance summary")
    print(f"  Warp total time: {_fmt_optional(performance.get('warp_total_time'))} s")
    print(f"  Period 1 time: {_fmt_optional(performance.get('period_1_runtime'))} s")
    print(
        f"  Mean period time excluding period 1: "
        f"{_fmt_optional(performance.get('period_runtime_mean_excluding_period_1'))} s"
    )
    speedup_transient = performance.get("speedup_vs_mf6_transient")
    speedup_warm = performance.get("speedup_vs_mf6_including_warm_start")
    print(f"  Speedup vs MF6 transient: {_fmt_optional(speedup_transient)}x")
    print(f"  Speedup vs MF6 including warm start: {_fmt_optional(speedup_warm)}x")
    print(
        f"  total outer iterations: {performance.get('total_outer_iterations')} "
        f"(period 1: {performance.get('period_1_outer_iterations')})"
    )
    if performance.get("runtime_target_met"):
        print(f"  runtime target (<{performance.get('runtime_target_s')}s): PASS: performance target met")
    else:
        print(f"  runtime target (<{performance.get('runtime_target_s')}s): WARNING: runtime target not met")
    if not performance.get("profile_available"):
        print(f"  WARNING: detailed profiling not implemented ({performance.get('profile_reason')})")
    else:
        profile = performance.get("profile") or {}
        totals = profile.get("totals") or {}
        fractions = profile.get("fractions_of_period_total") or {}
        print("  profile totals:")
        for key in (
            "T_update_seconds",
            "storage_kernel_seconds",
            "fine_m_inv_refresh_seconds",
            "dynamic_coarse_refresh_seconds",
            "rhs_assembly_seconds",
            "inner_solver_seconds",
            "outer_convergence_check_seconds",
            "final_nonlinear_residual_check_seconds",
            "head_download_seconds",
        ):
            value = totals.get(key)
            fraction = fractions.get(key)
            print(
                f"    {key}: {_fmt_optional(value)} s "
                f"({_fmt_optional(None if fraction is None else 100.0 * float(fraction), '.3g')}%)"
            )
        print(f"    mass_balance_runtime: {_fmt_optional(profile.get('mass_balance_runtime'))} s")
    print("\nMass balance summary")
    print(f"  cumulative discrepancy: {_fmt_optional(mass_balance.get('cumulative_percent_discrepancy'), '.6g')}%")
    print(f"  max period discrepancy: {_fmt_optional(mass_balance.get('max_abs_percent_discrepancy'), '.6g')}%")
    worst_period = mass_balance.get("worst_period") or {}
    print(f"  worst period: {worst_period.get('period')}")
    print(f"  class: {mass_class}")
    print(f"  passed: {mass_balance.get('mass_balance_passed')}")

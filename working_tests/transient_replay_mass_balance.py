from __future__ import annotations

from pathlib import Path

import numpy as np

from working_tests.transient_replay_storage import _specific_storage_potential
from working_tests.transient_replay_settings import (
    MASS_BALANCE_ACCEPTABLE_PCT,
    MASS_BALANCE_EXCELLENT_PCT,
    MASS_BALANCE_GOOD_PCT,
    MASS_BALANCE_STARTUP_PERIOD,
    MASS_BALANCE_STARTUP_WARN_PCT,
    STORAGE_ACTIVE_SET_NONE,
    STORAGE_REFERENCE_CURRENT_PICARD,
)


def transmissivity_from_period_head(
    *,
    formulation: str,
    head: np.ndarray,
    k: np.ndarray,
    top: np.ndarray,
    bottom: np.ndarray,
    active: np.ndarray,
    min_sat: float,
) -> np.ndarray:
    formulation_mode = str(formulation).strip().lower()
    if formulation_mode == "confined":
        thickness = np.maximum(np.asarray(top, dtype=np.float64) - np.asarray(bottom, dtype=np.float64), float(min_sat))
        transmissivity = np.asarray(k, dtype=np.float64) * thickness
        transmissivity[np.asarray(active, dtype=np.int32) == 0] = 0.0
        return transmissivity.astype(np.float64, copy=False)
    thickness = np.maximum(np.asarray(head, dtype=np.float64) - np.asarray(bottom, dtype=np.float64), float(min_sat))
    full_thickness = np.maximum(np.asarray(top, dtype=np.float64) - np.asarray(bottom, dtype=np.float64), float(min_sat))
    t = np.asarray(k, dtype=np.float64) * np.minimum(thickness, full_thickness)
    t[np.asarray(active, dtype=np.int32) == 0] = 0.0
    return t.astype(np.float64, copy=False)


def _split_signed_flux(total: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(total, dtype=np.float64)
    return float(np.sum(np.maximum(arr, 0.0))), float(np.sum(np.maximum(-arr, 0.0)))


def compute_boundary_flux_terms(
    *,
    T_field: np.ndarray,
    recharge_rate: float,
    head: np.ndarray,
    active: np.ndarray,
    bc_mask: np.ndarray,
    bc_values: np.ndarray,
    dx: float,
) -> dict:
    T = np.asarray(T_field, dtype=np.float64)
    h = np.asarray(head, dtype=np.float64)
    active_i = np.asarray(active, dtype=np.int32) != 0
    bc_i = np.asarray(bc_mask, dtype=np.int32) != 0
    bc_v = np.asarray(bc_values, dtype=np.float64)
    free = active_i & (~bc_i)
    dx_f = float(dx)
    tiny = 1.0e-12
    h_use = np.asarray(h, dtype=np.float64).copy()
    h_use[~active_i] = 0.0
    h_use[bc_i] = bc_v[bc_i]
    recharge_cell = np.zeros_like(h_use, dtype=np.float64)
    recharge_cell[free] = float(recharge_rate) * dx_f * dx_f
    recharge_in, recharge_out = _split_signed_flux(recharge_cell)

    chd_in = 0.0
    chd_out = 0.0

    act_l = active_i[:, :-1]
    act_r = active_i[:, 1:]
    conn_e = act_l & act_r
    T_l = T[:, :-1]
    T_r = T[:, 1:]
    denom_e = T_l + T_r
    cond_e = np.zeros((T.shape[0], T.shape[1] - 1), dtype=np.float64)
    valid_e = conn_e & (T_l > 0.0) & (T_r > 0.0) & (denom_e > tiny)
    cond_e[valid_e] = 2.0 * T_l[valid_e] * T_r[valid_e] / denom_e[valid_e]
    q_l_to_r = cond_e * (h_use[:, :-1] - h_use[:, 1:])
    bc_l = bc_i[:, :-1]
    bc_r = bc_i[:, 1:]
    dom_l = conn_e & (~bc_l) & bc_r
    dom_r = conn_e & bc_l & (~bc_r)
    q_int_to_bc_l = np.where(dom_l, q_l_to_r, 0.0)
    q_int_to_bc_r = np.where(dom_r, -q_l_to_r, 0.0)
    chd_out += float(np.sum(np.maximum(q_int_to_bc_l, 0.0)))
    chd_in += float(np.sum(-np.minimum(q_int_to_bc_l, 0.0)))
    chd_out += float(np.sum(np.maximum(q_int_to_bc_r, 0.0)))
    chd_in += float(np.sum(-np.minimum(q_int_to_bc_r, 0.0)))

    act_t = active_i[:-1, :]
    act_b = active_i[1:, :]
    conn_s = act_t & act_b
    T_t = T[:-1, :]
    T_b = T[1:, :]
    denom_s = T_t + T_b
    cond_s = np.zeros((T.shape[0] - 1, T.shape[1]), dtype=np.float64)
    valid_s = conn_s & (T_t > 0.0) & (T_b > 0.0) & (denom_s > tiny)
    cond_s[valid_s] = 2.0 * T_t[valid_s] * T_b[valid_s] / denom_s[valid_s]
    q_t_to_b = cond_s * (h_use[:-1, :] - h_use[1:, :])
    bc_t = bc_i[:-1, :]
    bc_b = bc_i[1:, :]
    dom_t = conn_s & (~bc_t) & bc_b
    dom_b = conn_s & bc_t & (~bc_b)
    q_int_to_bc_t = np.where(dom_t, q_t_to_b, 0.0)
    q_int_to_bc_b = np.where(dom_b, -q_t_to_b, 0.0)
    chd_out += float(np.sum(np.maximum(q_int_to_bc_t, 0.0)))
    chd_in += float(np.sum(-np.minimum(q_int_to_bc_t, 0.0)))
    chd_out += float(np.sum(np.maximum(q_int_to_bc_b, 0.0)))
    chd_in += float(np.sum(-np.minimum(q_int_to_bc_b, 0.0)))

    ghb_in = 0.0
    ghb_out = 0.0
    total_in = recharge_in + chd_in + ghb_in
    total_out = recharge_out + chd_out + ghb_out
    in_minus_out = total_in - total_out
    throughflow = 0.5 * (total_in + total_out)
    denom = abs(total_in) + abs(total_out)
    percent_discrepancy = 0.0 if denom == 0.0 else 100.0 * in_minus_out / denom
    imbalance_fraction = 0.0 if throughflow == 0.0 else in_minus_out / throughflow
    return {
        "recharge_in": recharge_in,
        "recharge_out": recharge_out,
        "chd_in": chd_in,
        "chd_out": chd_out,
        "ghb_in": ghb_in,
        "ghb_out": ghb_out,
        "total_in_without_storage": total_in,
        "total_out_without_storage": total_out,
        "in_minus_out_without_storage": in_minus_out,
        "percent_discrepancy_without_storage": percent_discrepancy,
        "throughflow_without_storage": throughflow,
        "imbalance_fraction_without_storage": imbalance_fraction,
    }


def finalize_mass_balance_row(
    *,
    period_number: int,
    base_terms: dict,
    storage_release: np.ndarray,
) -> dict:
    storage_in, storage_out = _split_signed_flux(storage_release)
    total_in = float(base_terms["recharge_in"] + base_terms["chd_in"] + base_terms["ghb_in"] + storage_in)
    total_out = float(base_terms["recharge_out"] + base_terms["chd_out"] + base_terms["ghb_out"] + storage_out)
    in_minus_out = total_in - total_out
    throughflow = 0.5 * (total_in + total_out)
    denom = abs(total_in) + abs(total_out)
    percent_discrepancy = 0.0 if denom == 0.0 else 100.0 * in_minus_out / denom
    imbalance_fraction = 0.0 if throughflow == 0.0 else in_minus_out / throughflow
    return {
        "period": int(period_number),
        "recharge_in": float(base_terms["recharge_in"]),
        "recharge_out": float(base_terms["recharge_out"]),
        "chd_in": float(base_terms["chd_in"]),
        "chd_out": float(base_terms["chd_out"]),
        "ghb_in": float(base_terms["ghb_in"]),
        "ghb_out": float(base_terms["ghb_out"]),
        "storage_in": float(storage_in),
        "storage_out": float(storage_out),
        "total_in": float(total_in),
        "total_out": float(total_out),
        "in_minus_out": float(in_minus_out),
        "percent_discrepancy": float(percent_discrepancy),
        "throughflow": float(throughflow),
        "imbalance_fraction": float(imbalance_fraction),
    }


def summarize_mass_balance_rows(rows: list[dict]) -> tuple[dict, dict]:
    if not rows:
        return {}, {}
    keys = (
        "recharge_in",
        "recharge_out",
        "chd_in",
        "chd_out",
        "ghb_in",
        "ghb_out",
        "storage_in",
        "storage_out",
        "total_in",
        "total_out",
        "in_minus_out",
    )
    cumulative = {"n_periods": int(len(rows))}
    for key in keys:
        cumulative[f"{key}_total"] = float(sum(float(row.get(key, 0.0) or 0.0) for row in rows))
    total_in = float(cumulative["total_in_total"])
    total_out = float(cumulative["total_out_total"])
    in_minus_out = float(cumulative["in_minus_out_total"])
    throughflow = 0.5 * (total_in + total_out)
    denom = abs(total_in) + abs(total_out)
    cumulative["throughflow"] = float(throughflow)
    cumulative["percent_discrepancy"] = 0.0 if denom == 0.0 else float(100.0 * in_minus_out / denom)
    cumulative["imbalance_fraction"] = 0.0 if throughflow == 0.0 else float(in_minus_out / throughflow)

    def _worst_key(row: dict) -> float:
        return abs(float(row.get("percent_discrepancy", 0.0) or 0.0))

    return cumulative, dict(max(rows, key=_worst_key))


def storage_budget_arrays_from_warp_result(warp_result: dict) -> dict[str, np.ndarray]:
    heads_new = np.asarray(warp_result["heads_per_period"], dtype=np.float64)
    warm_start_head = np.asarray(warp_result["warm_start_head"], dtype=np.float64)
    n_periods = int(heads_new.shape[0])
    ny, nx = heads_new.shape[1:]
    heads_old = warp_result.get("heads_old_per_period")
    if heads_old is None:
        heads_old_arr = np.zeros_like(heads_new)
        previous = warm_start_head
        for period_index in range(n_periods):
            heads_old_arr[period_index] = previous
            previous = heads_new[period_index]
    else:
        heads_old_arr = np.asarray(heads_old, dtype=np.float64)
    base_coeff = warp_result.get("storativity")
    if base_coeff is None:
        base_coeff_arr = np.zeros((ny, nx), dtype=np.float64)
    else:
        base_coeff_arr = np.asarray(base_coeff, dtype=np.float64)

    def _period_or_repeat(key: str, default: np.ndarray) -> np.ndarray:
        value = warp_result.get(key)
        if value is None:
            if default.ndim == 2:
                return np.repeat(default[None, :, :], n_periods, axis=0)
            return np.asarray(default, dtype=np.float64)
        return np.asarray(value, dtype=np.float64)

    storage_reference_heads = _period_or_repeat("storage_reference_heads_per_period", heads_new)
    storage_coeffs = _period_or_repeat("storage_coeffs_per_period", base_coeff_arr)
    sy_storage_coeffs = _period_or_repeat("sy_storage_coeffs_per_period", np.zeros_like(base_coeff_arr))
    ss_storage_coeffs = _period_or_repeat(
        "ss_storage_coeffs_per_period",
        storage_coeffs[0] if storage_coeffs.ndim == 3 else storage_coeffs,
    )
    storage_terms = warp_result.get("storage_terms_per_period")
    if storage_terms is None:
        delta = heads_new - heads_old_arr
        storage_terms_arr = storage_coeffs * delta / float(np.asarray(warp_result["dt"]).reshape(()))
    else:
        storage_terms_arr = np.asarray(storage_terms, dtype=np.float64)
    sy_storage_terms = warp_result.get("sy_storage_terms_per_period")
    if sy_storage_terms is None:
        delta = heads_new - heads_old_arr
        sy_storage_terms_arr = sy_storage_coeffs * delta / float(np.asarray(warp_result["dt"]).reshape(()))
    else:
        sy_storage_terms_arr = np.asarray(sy_storage_terms, dtype=np.float64)
    ss_storage_terms = warp_result.get("ss_storage_terms_per_period")
    if ss_storage_terms is None:
        delta = heads_new - heads_old_arr
        ss_storage_terms_arr = ss_storage_coeffs * delta / float(np.asarray(warp_result["dt"]).reshape(()))
    else:
        ss_storage_terms_arr = np.asarray(ss_storage_terms, dtype=np.float64)
    sy_crossing_volume_terms = _period_or_repeat(
        "sy_crossing_volume_terms_per_period",
        np.zeros((ny, nx), dtype=np.float64),
    )
    return {
        "heads_old_per_period": heads_old_arr,
        "heads_new_per_period": heads_new,
        "storage_reference_heads_per_period": storage_reference_heads,
        "storage_coeffs_per_period": storage_coeffs,
        "sy_storage_coeffs_per_period": sy_storage_coeffs,
        "ss_storage_coeffs_per_period": ss_storage_coeffs,
        "storage_terms_per_period": storage_terms_arr,
        "sy_storage_terms_per_period": sy_storage_terms_arr,
        "ss_storage_terms_per_period": ss_storage_terms_arr,
        "sy_crossing_volume_terms_per_period": sy_crossing_volume_terms,
    }


def save_warp_storage_budget_terms(
    *,
    path: Path,
    warp_result: dict,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = storage_budget_arrays_from_warp_result(warp_result=warp_result)
    np.savez_compressed(
        path,
        heads_old_per_period=np.asarray(arrays["heads_old_per_period"], dtype=np.float64),
        heads_new_per_period=np.asarray(arrays["heads_new_per_period"], dtype=np.float64),
        storage_reference_heads_per_period=np.asarray(arrays["storage_reference_heads_per_period"], dtype=np.float64),
        storage_coeffs_per_period=np.asarray(arrays["storage_coeffs_per_period"], dtype=np.float64),
        sy_storage_coeffs_per_period=np.asarray(arrays["sy_storage_coeffs_per_period"], dtype=np.float64),
        ss_storage_coeffs_per_period=np.asarray(arrays["ss_storage_coeffs_per_period"], dtype=np.float64),
        storage_terms_per_period=np.asarray(arrays["storage_terms_per_period"], dtype=np.float64),
        sy_storage_terms_per_period=np.asarray(arrays["sy_storage_terms_per_period"], dtype=np.float64),
        ss_storage_terms_per_period=np.asarray(arrays["ss_storage_terms_per_period"], dtype=np.float64),
        sy_crossing_volume_terms_per_period=np.asarray(arrays["sy_crossing_volume_terms_per_period"], dtype=np.float64),
        storage_reference=np.asarray(warp_result.get("storage_reference", STORAGE_REFERENCE_CURRENT_PICARD)),
        unconfined_storage_mode=np.asarray(
            "none" if warp_result["unconfined_storage_mode"] is None else warp_result["unconfined_storage_mode"]
        ),
        storage_top_threshold=np.asarray(warp_result.get("storage_top_threshold", "ge")),
        storage_active_set_strategy=np.asarray(warp_result.get("storage_active_set_strategy", STORAGE_ACTIVE_SET_NONE)),
        dt=np.asarray(warp_result["dt"], dtype=np.float64),
        warm_start_head=np.asarray(warp_result["warm_start_head"], dtype=np.float64),
    )
    return path


def compute_replay_mass_balance(
    *,
    spatial: dict,
    recharge_rates: np.ndarray,
    sy: float,
    dt: float,
    formulation: str,
    unconfined_storage_mode: str | None,
    warp_result: dict,
    min_sat: float,
    ss: float = 0.0,
) -> dict:
    active = np.asarray(spatial["active"], dtype=np.int32)
    bc_mask = np.asarray(spatial["bc_mask"], dtype=np.int32)
    bc_values = np.asarray(spatial["bc_values"], dtype=np.float64)
    k = np.asarray(spatial["k"], dtype=np.float64)
    top = np.asarray(spatial["top"], dtype=np.float64)
    bottom = np.asarray(spatial["bottom"], dtype=np.float64)
    dx = float(spatial["dx"])
    area = dx * dx
    heads_new = np.asarray(warp_result["heads_per_period"], dtype=np.float64)
    arrays = storage_budget_arrays_from_warp_result(warp_result=warp_result)
    heads_old = np.asarray(arrays["heads_old_per_period"], dtype=np.float64)
    storage_coeffs = np.asarray(arrays["storage_coeffs_per_period"], dtype=np.float64)
    ss_coeffs = np.asarray(arrays["ss_storage_coeffs_per_period"], dtype=np.float64)
    n_periods = int(heads_new.shape[0])
    recharge_series = np.asarray(recharge_rates, dtype=np.float64).reshape(-1)
    linearized_rows: list[dict] = []
    volume_sy_rows: list[dict] = []
    full_thickness = np.maximum(top - bottom, 0.0)
    for period_index in range(n_periods):
        head_new = heads_new[period_index]
        head_old = heads_old[period_index]
        T_period = transmissivity_from_period_head(
            formulation=formulation,
            head=head_new,
            k=k,
            top=top,
            bottom=bottom,
            active=active,
            min_sat=min_sat,
        )
        base_terms = compute_boundary_flux_terms(
            T_field=T_period,
            recharge_rate=float(recharge_series[period_index]),
            head=head_new,
            active=active,
            bc_mask=bc_mask,
            bc_values=bc_values,
            dx=dx,
        )
        delta_head = head_new - head_old
        storage_release_linearized = -np.asarray(storage_coeffs[period_index], dtype=np.float64) * delta_head * area / float(dt)
        sat_old = np.clip(head_old - bottom, 0.0, full_thickness)
        sat_new = np.clip(head_new - bottom, 0.0, full_thickness)
        sy_storage_release_volume = -float(sy) * (sat_new - sat_old) * area / float(dt)
        phi_old = _specific_storage_potential(
            head=head_old,
            bottom=bottom,
            top=top,
            ss=float(ss),
        )
        phi_new = _specific_storage_potential(
            head=head_new,
            bottom=bottom,
            top=top,
            ss=float(ss),
        )
        ss_storage_release_volume = -(phi_new - phi_old) * area / float(dt)
        ss_storage_release_linearized = -np.asarray(ss_coeffs[period_index], dtype=np.float64) * delta_head * area / float(dt)
        storage_release_volume = sy_storage_release_volume + ss_storage_release_volume
        linearized_row = finalize_mass_balance_row(
            period_number=period_index + 1,
            base_terms=base_terms,
            storage_release=storage_release_linearized,
        )
        linearized_row["storage_release_total"] = float(np.sum(storage_release_linearized))
        linearized_rows.append(linearized_row)
        volume_sy_row = finalize_mass_balance_row(
            period_number=period_index + 1,
            base_terms=base_terms,
            storage_release=storage_release_volume,
        )
        volume_sy_row["storage_release_total"] = float(np.sum(storage_release_volume))
        volume_sy_row["sy_storage_release_volume_total"] = float(np.sum(sy_storage_release_volume))
        volume_sy_row["ss_storage_release_volume_total"] = float(np.sum(ss_storage_release_volume))
        volume_sy_row["ss_storage_release_linearized_total"] = float(np.sum(ss_storage_release_linearized))
        volume_sy_rows.append(volume_sy_row)
    linearized_cumulative, linearized_worst = summarize_mass_balance_rows(linearized_rows)
    volume_sy_cumulative, volume_sy_worst = summarize_mass_balance_rows(volume_sy_rows)
    preferred_storage_budget = (
        "volume_sy" if str(unconfined_storage_mode).strip().lower() == "mf6_convertible_secant_sy" else "linearized"
    )
    preferred_rows = volume_sy_rows if preferred_storage_budget == "volume_sy" else linearized_rows
    preferred_cumulative = volume_sy_cumulative if preferred_storage_budget == "volume_sy" else linearized_cumulative
    preferred_worst = volume_sy_worst if preferred_storage_budget == "volume_sy" else linearized_worst
    max_abs_percent_discrepancy = max(
        abs(float(row.get("percent_discrepancy", 0.0) or 0.0)) for row in preferred_rows
    ) if preferred_rows else None
    max_abs_in_minus_out = max(
        abs(float(row.get("in_minus_out", 0.0) or 0.0)) for row in preferred_rows
    ) if preferred_rows else None
    return {
        "warp_mass_balance_available": True,
        "warp_storage_budget_available": True,
        "mf6_mass_balance_available": False,
        "mf6_storage_budget_available": False,
        "storage_budget_comparison_available": False,
        "preferred_storage_budget": preferred_storage_budget,
        "per_period": preferred_rows,
        "cumulative": preferred_cumulative,
        "worst_period": preferred_worst,
        "max_abs_percent_discrepancy": max_abs_percent_discrepancy,
        "max_abs_in_minus_out": max_abs_in_minus_out,
        "mass_balance_linearized": {
            "per_period": linearized_rows,
            "cumulative": linearized_cumulative,
            "worst_period": linearized_worst,
        },
        "mass_balance_volume_sy": {
            "per_period": volume_sy_rows,
            "cumulative": volume_sy_cumulative,
            "worst_period": volume_sy_worst,
        },
        "status_thresholds": {
            "excellent_percent_discrepancy": MASS_BALANCE_EXCELLENT_PCT,
            "good_percent_discrepancy": MASS_BALANCE_GOOD_PCT,
            "warning_percent_discrepancy": MASS_BALANCE_STARTUP_WARN_PCT,
        },
    }


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
        failures.append(f"cumulative percent discrepancy {cum_abs:.5g}% >= {MASS_BALANCE_ACCEPTABLE_PCT}%")
    if nonstartup_values and worst_nonstartup >= MASS_BALANCE_GOOD_PCT:
        failed = True
        failures.append(
            f"a non-startup period has percent discrepancy {worst_nonstartup:.5g}% >= {MASS_BALANCE_GOOD_PCT}%"
        )
    if startup_pct >= MASS_BALANCE_STARTUP_WARN_PCT:
        failed = True
        failures.append(
            f"startup period {startup_period} percent discrepancy {startup_pct:.5g}% >= {MASS_BALANCE_STARTUP_WARN_PCT}%"
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
            f"Period {startup_period} has a slightly elevated mass-balance discrepancy ({startup_pct:.5g}%) "
            "during the confined_pre_solve / warm-start startup transient."
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

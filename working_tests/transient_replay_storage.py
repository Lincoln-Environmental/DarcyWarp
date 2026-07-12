from __future__ import annotations

import numpy as np

from working_tests.transient_artifacts import (
    UNCONFINED_STORAGE_INTEGRATED_SY_SS,
    UNCONFINED_STORAGE_MF6_CONVERTIBLE,
    UNCONFINED_STORAGE_MF6_CONVERTIBLE_CROSSING_VOLUME_SY,
    UNCONFINED_STORAGE_MF6_CONVERTIBLE_SECANT_SY,
    UNCONFINED_STORAGE_MF6_CONVERTIBLE_TOP_SWITCH,
    UNCONFINED_STORAGE_MODES,
    UNCONFINED_STORAGE_PHREATIC_ONLY,
)
from working_tests.transient_replay_settings import (
    DEFAULT_MIN_SAT,
    STORAGE_TOP_THRESHOLD_GE,
    STORAGE_TOP_THRESHOLD_GT,
    STORAGE_TOP_THRESHOLD_MODES,
)


def _initial_transmissivity(
    k: np.ndarray,
    initial_head: np.ndarray,
    top: np.ndarray,
    bottom: np.ndarray,
    active: np.ndarray,
    min_sat: float = DEFAULT_MIN_SAT,
) -> np.ndarray:
    thickness = np.maximum(
        np.asarray(initial_head, dtype=np.float64) - np.asarray(bottom, dtype=np.float64),
        float(min_sat),
    )
    full_thickness = np.maximum(
        np.asarray(top, dtype=np.float64) - np.asarray(bottom, dtype=np.float64),
        float(min_sat),
    )
    t = np.asarray(k, dtype=np.float64) * np.minimum(thickness, full_thickness)
    t[np.asarray(active, dtype=np.int32) == 0] = 0.0
    return t.astype(np.float64, copy=False)


def _confined_transmissivity(
    k: np.ndarray,
    top: np.ndarray,
    bottom: np.ndarray,
    active: np.ndarray,
    min_sat: float = DEFAULT_MIN_SAT,
) -> np.ndarray:
    thickness = np.maximum(
        np.asarray(top, dtype=np.float64) - np.asarray(bottom, dtype=np.float64),
        float(min_sat),
    )
    transmissivity = np.asarray(k, dtype=np.float64) * thickness
    transmissivity[np.asarray(active, dtype=np.int32) == 0] = 0.0
    return transmissivity.astype(np.float64, copy=False)


def _confined_storage_coeff(
    ss: float,
    top: np.ndarray,
    bottom: np.ndarray,
    active: np.ndarray,
    min_sat: float = DEFAULT_MIN_SAT,
) -> np.ndarray:
    thickness = np.maximum(
        np.asarray(top, dtype=np.float64) - np.asarray(bottom, dtype=np.float64),
        float(min_sat),
    )
    storage = float(ss) * thickness
    storage[np.asarray(active, dtype=np.int32) == 0] = 0.0
    return storage.astype(np.float64, copy=False)


def _specific_storage_potential(
    *,
    head: np.ndarray,
    bottom: np.ndarray,
    top: np.ndarray,
    ss: float,
) -> np.ndarray:
    head_arr = np.asarray(head, dtype=np.float64)
    bottom_arr = np.asarray(bottom, dtype=np.float64)
    top_arr = np.asarray(top, dtype=np.float64)
    thickness = np.maximum(top_arr - bottom_arr, 0.0)
    rel = head_arr - bottom_arr
    phi = np.zeros_like(head_arr, dtype=np.float64)
    partial = (rel > 0.0) & (rel < thickness)
    full = rel >= thickness
    phi[partial] = 0.5 * float(ss) * rel[partial] * rel[partial]
    phi[full] = (
        0.5 * float(ss) * thickness[full] * thickness[full]
        + float(ss) * thickness[full] * (rel[full] - thickness[full])
    )
    return phi


def build_unconfined_storativity(
    *,
    sy: float,
    active: np.ndarray,
    bc_mask: np.ndarray,
    ss: float = 0.0,
    head_ref: np.ndarray | None = None,
    head_old: np.ndarray | None = None,
    bottom: np.ndarray | None = None,
    top: np.ndarray | None = None,
    min_sat: float = DEFAULT_MIN_SAT,
    include_specific_storage: bool = False,
    storage_mode: str | None = None,
    storage_top_threshold: str = STORAGE_TOP_THRESHOLD_GE,
    secant_eps: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray | None]:
    mode = (
        UNCONFINED_STORAGE_MF6_CONVERTIBLE
        if storage_mode is None and include_specific_storage
        else (
            UNCONFINED_STORAGE_PHREATIC_ONLY
            if storage_mode is None
            else str(storage_mode).strip().lower()
        )
    )
    if mode not in UNCONFINED_STORAGE_MODES:
        raise ValueError(f"storage_mode must be one of {sorted(UNCONFINED_STORAGE_MODES)}.")
    threshold_mode = str(storage_top_threshold).strip().lower()
    if threshold_mode not in STORAGE_TOP_THRESHOLD_MODES:
        raise ValueError(f"storage_top_threshold must be one of {sorted(STORAGE_TOP_THRESHOLD_MODES)}.")

    active_i = np.asarray(active, dtype=np.int32)
    bcm = np.asarray(bc_mask, dtype=np.int32)
    free = (active_i != 0) & (bcm == 0)
    components = compute_unconfined_storage_components(
        sy=float(sy),
        ss=float(ss),
        head_ref=head_ref,
        head_old=head_old,
        bottom=bottom,
        top=top,
        active=active_i,
        bc_mask=bcm,
        min_sat=float(min_sat),
        storage_mode=mode,
        storage_top_threshold=threshold_mode,
        secant_eps=float(secant_eps),
    )
    storativity = np.asarray(components["storage_coeff"], dtype=np.float64)
    sat_ref = np.asarray(components["sat_ref_ss"], dtype=np.float64) if components["sat_ref_ss"] is not None else None
    return (
        storativity.astype(np.float64, copy=False),
        None if sat_ref is None else sat_ref.astype(np.float64, copy=False),
    )


def compute_unconfined_storage_components(
    *,
    sy: float,
    ss: float,
    head_ref: np.ndarray | None,
    head_old: np.ndarray | None,
    bottom: np.ndarray | None,
    top: np.ndarray | None,
    active: np.ndarray,
    bc_mask: np.ndarray,
    min_sat: float,
    storage_mode: str,
    storage_top_threshold: str = STORAGE_TOP_THRESHOLD_GE,
    secant_eps: float = 1.0e-12,
    above_top_mask_override: np.ndarray | None = None,
) -> dict[str, np.ndarray | None]:
    mode = str(storage_mode).strip().lower()
    threshold_mode = str(storage_top_threshold).strip().lower()
    if mode not in UNCONFINED_STORAGE_MODES:
        raise ValueError(f"storage_mode must be one of {sorted(UNCONFINED_STORAGE_MODES)}.")
    if threshold_mode not in STORAGE_TOP_THRESHOLD_MODES:
        raise ValueError(f"storage_top_threshold must be one of {sorted(STORAGE_TOP_THRESHOLD_MODES)}.")

    active_i = np.asarray(active, dtype=np.int32)
    bc_i = np.asarray(bc_mask, dtype=np.int32)
    free = (active_i != 0) & (bc_i == 0)
    shape = active_i.shape

    storage_coeff = np.zeros(shape, dtype=np.float64)
    sy_coeff = np.zeros(shape, dtype=np.float64)
    ss_coeff = np.zeros(shape, dtype=np.float64)
    sat_old_zero = np.zeros(shape, dtype=np.float64)
    sat_ref_zero = np.zeros(shape, dtype=np.float64)
    sat_ref_ss = np.zeros(shape, dtype=np.float64)
    full_thickness = None
    raw_above_top = None
    effective_above_top = None

    if mode == UNCONFINED_STORAGE_PHREATIC_ONLY:
        sy_coeff[free] = float(sy)
        storage_coeff[free] = float(sy)
        return {
            "storage_coeff": storage_coeff,
            "sy_coeff": sy_coeff,
            "ss_coeff": ss_coeff,
            "sat_old_zero": sat_old_zero,
            "sat_ref_zero": sat_ref_zero,
            "sat_ref_ss": None,
            "full_thickness": None,
            "raw_above_top": None,
            "effective_above_top": None,
        }

    if head_ref is None or bottom is None or top is None:
        raise ValueError(f"storage_mode='{mode}' requires head_ref, bottom, and top.")

    head_ref_arr = np.asarray(head_ref, dtype=np.float64)
    bottom_arr = np.asarray(bottom, dtype=np.float64)
    top_arr = np.asarray(top, dtype=np.float64)
    full_thickness = np.maximum(top_arr - bottom_arr, 0.0)
    sat_ref_zero = np.clip(head_ref_arr - bottom_arr, 0.0, full_thickness)
    sat_ref_ss = np.clip(head_ref_arr - bottom_arr, 0.0, full_thickness)

    if head_old is None:
        head_old_arr = head_ref_arr
    else:
        head_old_arr = np.asarray(head_old, dtype=np.float64)
    sat_old_zero = np.clip(head_old_arr - bottom_arr, 0.0, full_thickness)
    dh_ref = head_ref_arr - head_old_arr
    phi_ref = _specific_storage_potential(head=head_ref_arr, bottom=bottom_arr, top=top_arr, ss=float(ss))
    phi_old = _specific_storage_potential(head=head_old_arr, bottom=bottom_arr, top=top_arr, ss=float(ss))
    ss_secant = np.zeros(shape, dtype=np.float64)
    ss_moving = np.abs(dh_ref) > float(secant_eps)
    ss_secant[ss_moving] = (phi_ref[ss_moving] - phi_old[ss_moving]) / dh_ref[ss_moving]
    ss_secant[~ss_moving] = float(ss) * sat_ref_ss[~ss_moving]
    ss_secant = np.maximum(ss_secant, 0.0)

    if mode in {UNCONFINED_STORAGE_INTEGRATED_SY_SS, UNCONFINED_STORAGE_MF6_CONVERTIBLE}:
        sy_coeff[free] = float(sy)
        ss_coeff[free] = ss_secant[free]
    elif mode == UNCONFINED_STORAGE_MF6_CONVERTIBLE_TOP_SWITCH:
        raw_above_top = top_switch_above_mask(
            head_ref=head_ref_arr,
            top=top_arr,
            threshold_mode=threshold_mode,
            active=active_i,
            bc_mask=bc_i,
        )
        effective_above_top = (
            np.asarray(above_top_mask_override, dtype=bool)
            if above_top_mask_override is not None
            else raw_above_top.copy()
        )
        effective_above_top = effective_above_top & free
        sy_coeff[free] = float(sy)
        sy_coeff[effective_above_top] = 0.0
        ss_coeff[free] = ss_secant[free]
        ss_coeff[effective_above_top] = float(ss) * full_thickness[effective_above_top]
    elif mode == UNCONFINED_STORAGE_MF6_CONVERTIBLE_SECANT_SY:
        sy_coeff_calc = np.zeros(shape, dtype=np.float64)
        moving = np.abs(dh_ref) > float(secant_eps)
        sy_coeff_calc[moving] = float(sy) * ((sat_ref_zero[moving] - sat_old_zero[moving]) / dh_ref[moving])
        fallback_below_top = (
            (np.abs(dh_ref) <= float(secant_eps)) & (head_ref_arr < top_arr) & (head_ref_arr > bottom_arr)
        )
        sy_coeff_calc[fallback_below_top] = float(sy)
        sy_coeff_calc = np.clip(sy_coeff_calc, 0.0, float(sy))
        sy_coeff[free] = sy_coeff_calc[free]
        ss_coeff[free] = ss_secant[free]
    elif mode == UNCONFINED_STORAGE_MF6_CONVERTIBLE_CROSSING_VOLUME_SY:
        sy_coeff_calc = np.zeros(shape, dtype=np.float64)
        moving = np.abs(dh_ref) > float(secant_eps)
        sy_coeff_calc[moving] = float(sy) * ((sat_ref_zero[moving] - sat_old_zero[moving]) / dh_ref[moving])
        fallback_below_top = (
            (np.abs(dh_ref) <= float(secant_eps)) & (head_ref_arr < top_arr) & (head_ref_arr > bottom_arr)
        )
        sy_coeff_calc[fallback_below_top] = float(sy)
        sy_coeff_calc = np.clip(sy_coeff_calc, 0.0, float(sy))
        sy_coeff[free] = sy_coeff_calc[free]
        ss_coeff[free] = ss_secant[free]
    else:
        raise ValueError(f"unsupported storage_mode '{mode}'.")

    storage_coeff = sy_coeff + ss_coeff
    storage_coeff[~free] = 0.0
    sy_coeff[~free] = 0.0
    ss_coeff[~free] = 0.0
    sat_old_zero[~free] = 0.0
    sat_ref_zero[~free] = 0.0
    sat_ref_ss[~free] = 0.0
    return {
        "storage_coeff": storage_coeff,
        "sy_coeff": sy_coeff,
        "ss_coeff": ss_coeff,
        "sat_old_zero": sat_old_zero,
        "sat_ref_zero": sat_ref_zero,
        "sat_ref_ss": sat_ref_ss,
        "full_thickness": full_thickness,
        "raw_above_top": raw_above_top,
        "effective_above_top": effective_above_top,
    }


def top_switch_above_mask(
    *,
    head_ref: np.ndarray,
    top: np.ndarray,
    threshold_mode: str = STORAGE_TOP_THRESHOLD_GE,
    active: np.ndarray | None = None,
    bc_mask: np.ndarray | None = None,
) -> np.ndarray:
    head_ref_arr = np.asarray(head_ref, dtype=np.float64)
    top_arr = np.asarray(top, dtype=np.float64)
    mode = str(threshold_mode).strip().lower()
    if mode == STORAGE_TOP_THRESHOLD_GE:
        above = head_ref_arr >= top_arr
    elif mode == STORAGE_TOP_THRESHOLD_GT:
        above = head_ref_arr > top_arr
    else:
        raise ValueError(f"threshold_mode must be one of {sorted(STORAGE_TOP_THRESHOLD_MODES)}.")
    if active is not None:
        above = above & (np.asarray(active, dtype=np.int32) != 0)
    if bc_mask is not None:
        above = above & (np.asarray(bc_mask, dtype=np.int32) == 0)
    return above


def apply_top_switch_hysteresis(
    *,
    raw_above_top: np.ndarray,
    head_ref: np.ndarray,
    top: np.ndarray,
    previous_above_top: np.ndarray | None,
    hysteresis_eps: float,
    active: np.ndarray | None = None,
    bc_mask: np.ndarray | None = None,
) -> np.ndarray:
    raw_mask = np.asarray(raw_above_top, dtype=bool)
    if previous_above_top is None:
        out = raw_mask.copy()
    else:
        head_ref_arr = np.asarray(head_ref, dtype=np.float64)
        top_arr = np.asarray(top, dtype=np.float64)
        prev_mask = np.asarray(previous_above_top, dtype=bool)
        eps = max(float(hysteresis_eps), 0.0)
        out = raw_mask.copy()
        out[prev_mask & (head_ref_arr >= (top_arr - eps))] = True
        out[(~prev_mask) & (head_ref_arr < (top_arr + eps))] = False
    if active is not None:
        out = out & (np.asarray(active, dtype=np.int32) != 0)
    if bc_mask is not None:
        out = out & (np.asarray(bc_mask, dtype=np.int32) == 0)
    return out


def should_freeze_top_switch(
    *,
    changed_fraction: float,
    stable_iteration_count: int,
    fraction_tol: float,
    freeze_after_stable_iterations: int,
) -> bool:
    if int(freeze_after_stable_iterations) <= 0:
        return False
    return (
        float(changed_fraction) <= float(fraction_tol)
        and int(stable_iteration_count) >= int(freeze_after_stable_iterations)
    )

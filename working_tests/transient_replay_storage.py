from __future__ import annotations

import numpy as np

from working_tests.transient_artifacts import (
    UNCONFINED_STORAGE_MF6_CONVERTIBLE_SECANT_SY,
    UNCONFINED_STORAGE_MODES,
)
from working_tests.transient_replay_settings import DEFAULT_MIN_SAT


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
    secant_eps: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray | None]:
    mode = UNCONFINED_STORAGE_MF6_CONVERTIBLE_SECANT_SY
    if storage_mode is not None and str(storage_mode).strip().lower() != mode:
        raise ValueError(f"storage_mode must be '{mode}'.")

    active_i = np.asarray(active, dtype=np.int32)
    bcm = np.asarray(bc_mask, dtype=np.int32)
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
        secant_eps=float(secant_eps),
    )
    storativity = np.asarray(components["storage_coeff"], dtype=np.float64)
    sat_ref = np.asarray(components["sat_ref_ss"], dtype=np.float64)
    return (
        storativity.astype(np.float64, copy=False),
        sat_ref.astype(np.float64, copy=False),
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
    secant_eps: float = 1.0e-12,
) -> dict[str, np.ndarray | None]:
    mode = str(storage_mode).strip().lower()
    if mode != UNCONFINED_STORAGE_MF6_CONVERTIBLE_SECANT_SY:
        raise ValueError(f"storage_mode must be '{UNCONFINED_STORAGE_MF6_CONVERTIBLE_SECANT_SY}'.")

    if head_ref is None or bottom is None or top is None:
        raise ValueError("secant-Sy storage requires head_ref, bottom, and top.")

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
        "raw_above_top": None,
        "effective_above_top": None,
    }

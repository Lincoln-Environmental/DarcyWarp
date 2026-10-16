# SPDX-License-Identifier: AGPL-3.0-only
"""Pure host-side unconfined storage relations shared by 2D solvers."""

from __future__ import annotations

import numpy as np


def specific_storage_potential(
    *,
    head: np.ndarray,
    bottom: np.ndarray,
    top: np.ndarray,
    specific_storage: float,
) -> np.ndarray:
    """Specific-storage potential per unit plan area for a convertible layer."""
    head_arr = np.asarray(head, dtype=np.float64)
    bottom_arr = np.asarray(bottom, dtype=np.float64)
    top_arr = np.asarray(top, dtype=np.float64)
    thickness = np.maximum(top_arr - bottom_arr, 0.0)
    rel = head_arr - bottom_arr
    ss = float(specific_storage)
    phi = np.zeros(np.broadcast_shapes(head_arr.shape, bottom_arr.shape, top_arr.shape), dtype=np.float64)
    rel_b = np.broadcast_to(rel, phi.shape)
    thickness_b = np.broadcast_to(thickness, phi.shape)
    partial = (rel_b > 0.0) & (rel_b < thickness_b)
    full = rel_b >= thickness_b
    phi[partial] = 0.5 * ss * rel_b[partial] * rel_b[partial]
    phi[full] = (
        0.5 * ss * thickness_b[full] * thickness_b[full]
        + ss * thickness_b[full] * (rel_b[full] - thickness_b[full])
    )
    return phi


def secant_specific_yield_coeff(
    *,
    head_ref: np.ndarray,
    head_old: np.ndarray,
    bottom: np.ndarray,
    top: np.ndarray,
    specific_yield: float,
    secant_eps: float = 1.0e-12,
) -> np.ndarray:
    """Return the MF6-compatible secant specific-yield coefficient."""
    head_ref_arr = np.asarray(head_ref, dtype=np.float64)
    head_old_arr = np.asarray(head_old, dtype=np.float64)
    bottom_arr = np.asarray(bottom, dtype=np.float64)
    top_arr = np.asarray(top, dtype=np.float64)
    thickness = np.maximum(top_arr - bottom_arr, 0.0)
    sat_ref = np.clip(head_ref_arr - bottom_arr, 0.0, thickness)
    sat_old = np.clip(head_old_arr - bottom_arr, 0.0, thickness)
    dh = head_ref_arr - head_old_arr
    coeff = np.zeros_like(np.broadcast_to(dh, np.broadcast_shapes(dh.shape, thickness.shape)), dtype=np.float64)
    moving = np.abs(dh) > float(secant_eps)
    coeff[moving] = float(specific_yield) * ((sat_ref[moving] - sat_old[moving]) / dh[moving])
    fallback = (~moving) & (head_ref_arr > bottom_arr) & (head_ref_arr < top_arr)
    coeff[fallback] = float(specific_yield)
    return np.clip(coeff, 0.0, float(specific_yield))


def secant_specific_storage_coeff(
    *,
    head_ref: np.ndarray,
    head_old: np.ndarray,
    bottom: np.ndarray,
    top: np.ndarray,
    specific_storage: float,
    secant_eps: float = 1.0e-12,
) -> np.ndarray:
    """Return the MF6-compatible secant specific-storage coefficient."""
    head_ref_arr = np.asarray(head_ref, dtype=np.float64)
    head_old_arr = np.asarray(head_old, dtype=np.float64)
    bottom_arr = np.asarray(bottom, dtype=np.float64)
    top_arr = np.asarray(top, dtype=np.float64)
    thickness = np.maximum(top_arr - bottom_arr, 0.0)
    dh = head_ref_arr - head_old_arr
    phi_ref = specific_storage_potential(
        head=head_ref_arr,
        bottom=bottom_arr,
        top=top_arr,
        specific_storage=float(specific_storage),
    )
    phi_old = specific_storage_potential(
        head=head_old_arr,
        bottom=bottom_arr,
        top=top_arr,
        specific_storage=float(specific_storage),
    )
    coeff = np.zeros_like(phi_ref, dtype=np.float64)
    moving = np.abs(dh) > float(secant_eps)
    coeff[moving] = (phi_ref[moving] - phi_old[moving]) / dh[moving]
    fallback = ~moving
    if np.any(fallback):
        saturated_thickness = np.clip(head_ref_arr - bottom_arr, 0.0, thickness)
        coeff[fallback] = float(specific_storage) * saturated_thickness[fallback]
    return np.maximum(coeff, 0.0)


def exact_unconfined_storage_terms(
    *,
    head_new: np.ndarray,
    head_old: np.ndarray,
    bottom: np.ndarray,
    top: np.ndarray,
    specific_yield: float,
    specific_storage: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return total, Sy, and Ss storage terms per unit plan area and time."""
    head_new_arr = np.asarray(head_new, dtype=np.float64)
    head_old_arr = np.asarray(head_old, dtype=np.float64)
    bottom_arr = np.asarray(bottom, dtype=np.float64)
    top_arr = np.asarray(top, dtype=np.float64)
    thickness = np.maximum(top_arr - bottom_arr, 0.0)
    sat_new = np.clip(head_new_arr - bottom_arr, 0.0, thickness)
    sat_old = np.clip(head_old_arr - bottom_arr, 0.0, thickness)
    sy_term = float(specific_yield) * (sat_new - sat_old) / float(dt)
    ss_term = (
        specific_storage_potential(
            head=head_new_arr,
            bottom=bottom_arr,
            top=top_arr,
            specific_storage=float(specific_storage),
        )
        - specific_storage_potential(
            head=head_old_arr,
            bottom=bottom_arr,
            top=top_arr,
            specific_storage=float(specific_storage),
        )
    ) / float(dt)
    return sy_term + ss_term, sy_term, ss_term

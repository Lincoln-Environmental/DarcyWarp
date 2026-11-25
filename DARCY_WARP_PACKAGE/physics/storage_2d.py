# SPDX-License-Identifier: AGPL-3.0-or-later
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
    """Evaluate specific-storage potential per unit plan area.

    :param head: Hydraulic head field in length units.
    :param bottom: Cell-bottom elevation field in length units.
    :param top: Cell-top elevation field in length units.
    :param specific_storage: Specific storage in inverse-length units.
    :returns: FP64 stored-water potential in length units, zero below the bottom,
        quadratic within the saturated thickness and linear above the cell top.
    """
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
    """Linearise drainable storage between two water-table positions.

    :param head_ref: Reference hydraulic head field for the current iterate.
    :param head_old: Hydraulic head at the beginning of the timestep.
    :param bottom: Cell-bottom elevation field.
    :param top: Cell-top elevation field.
    :param specific_yield: Dimensionless drainable porosity.
    :param secant_eps: Absolute head-difference threshold, in length units,
        below which the local derivative replaces the secant.
    :returns: FP64 dimensionless coefficient bounded by zero and specific_yield.
    """
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
    """Linearise elastic storage between the old head and reference head.

    :param head_ref: Reference hydraulic head field for the current iterate.
    :param head_old: Hydraulic head at the beginning of the timestep.
    :param bottom: Cell-bottom elevation field.
    :param top: Cell-top elevation field.
    :param specific_storage: Specific storage in inverse-length units.
    :param secant_eps: Absolute head-difference threshold, in length units,
        below which the local saturated-thickness derivative is used.
    :returns: Nonnegative FP64 dimensionless effective storage coefficient.
    """
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
    """Evaluate the exact change in convertible-layer storage over one step.

    :param head_new: Accepted end-of-step hydraulic head field.
    :param head_old: Hydraulic head field at the start of the timestep.
    :param bottom: Cell-bottom elevation field.
    :param top: Cell-top elevation field.
    :param specific_yield: Dimensionless drainable porosity.
    :param specific_storage: Specific storage in inverse-length units.
    :param dt: Positive timestep duration in model time units.
    :returns: Total, specific-yield and specific-storage rates per plan area
        (length/time), positive for an increase in stored water.
    """
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

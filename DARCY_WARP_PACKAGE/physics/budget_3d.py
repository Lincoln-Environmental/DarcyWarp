# SPDX-License-Identifier: AGPL-3.0-only
"""Boundary flux diagnostics for structured 3D finite-volume fields."""

from __future__ import annotations

import numpy as np


def boundary_interface_flux(
    head: np.ndarray,
    boundary_mask: np.ndarray,
    active: np.ndarray,
    tx_p: np.ndarray,
    tx_m: np.ndarray,
    ty_p: np.ndarray,
    ty_m: np.ndarray,
    tz_p: np.ndarray,
    tz_m: np.ndarray,
) -> float:
    """Return flow from selected boundary cells into other active cells.

    Each internal face is counted exactly once.  Positive values mean flow
    leaves the selected boundary cells and enters the rest of the active model.
    The function is agnostic to what the mask represents (river, coast, drain,
    or another named fixed-head boundary).
    """

    arrays = {
        "head": head,
        "boundary_mask": boundary_mask,
        "active": active,
        "tx_p": tx_p,
        "tx_m": tx_m,
        "ty_p": ty_p,
        "ty_m": ty_m,
        "tz_p": tz_p,
        "tz_m": tz_m,
    }
    shape = np.asarray(head).shape
    if len(shape) != 3:
        raise ValueError(f"3D fields are required; got head shape {shape}.")
    for name, value in arrays.items():
        if np.asarray(value).shape != shape:
            raise ValueError(f"{name} shape {np.asarray(value).shape} does not match {shape}.")

    h = np.asarray(head, dtype=np.float64)
    active_raw = np.asarray(active)
    boundary_raw = np.asarray(boundary_mask)
    if not np.all(np.isfinite(active_raw)) or not np.all(np.isfinite(boundary_raw)):
        raise ValueError("active and boundary_mask must contain finite values.")
    act = active_raw != 0
    bnd = (boundary_raw != 0) & act
    if not np.all(np.isfinite(h)):
        raise ValueError("head must contain finite values.")
    conductances = tuple(np.asarray(arr, dtype=np.float64) for arr in (tx_p, tx_m, ty_p, ty_m, tz_p, tz_m))
    if any(not np.all(np.isfinite(arr)) for arr in conductances):
        raise ValueError("face conductances must contain finite values.")
    if any(np.any(arr < 0.0) for arr in conductances):
        raise ValueError("face conductances must be non-negative.")
    tx_p, tx_m, ty_p, ty_m, tz_p, tz_m = conductances
    other = act & ~bnd
    total = 0.0

    face = bnd[:, :, :-1] & other[:, :, 1:]
    total += float(np.sum(np.asarray(tx_p, dtype=np.float64)[:, :, :-1][face] * (h[:, :, :-1][face] - h[:, :, 1:][face])))
    face = bnd[:, :, 1:] & other[:, :, :-1]
    total += float(np.sum(np.asarray(tx_m, dtype=np.float64)[:, :, 1:][face] * (h[:, :, 1:][face] - h[:, :, :-1][face])))

    face = bnd[:, :-1, :] & other[:, 1:, :]
    total += float(np.sum(np.asarray(ty_p, dtype=np.float64)[:, :-1, :][face] * (h[:, :-1, :][face] - h[:, 1:, :][face])))
    face = bnd[:, 1:, :] & other[:, :-1, :]
    total += float(np.sum(np.asarray(ty_m, dtype=np.float64)[:, 1:, :][face] * (h[:, 1:, :][face] - h[:, :-1, :][face])))

    face = bnd[:-1, :, :] & other[1:, :, :]
    total += float(np.sum(np.asarray(tz_p, dtype=np.float64)[:-1, :, :][face] * (h[:-1, :, :][face] - h[1:, :, :][face])))
    face = bnd[1:, :, :] & other[:-1, :, :]
    total += float(np.sum(np.asarray(tz_m, dtype=np.float64)[1:, :, :][face] * (h[1:, :, :][face] - h[:-1, :, :][face])))
    return total


def named_boundary_interface_fluxes(
    head: np.ndarray,
    named_masks: dict[str, np.ndarray],
    active: np.ndarray,
    tx_p: np.ndarray,
    tx_m: np.ndarray,
    ty_p: np.ndarray,
    ty_m: np.ndarray,
    tz_p: np.ndarray,
    tz_m: np.ndarray,
) -> dict[str, float]:
    """Evaluate ``boundary_interface_flux`` for each named mask."""

    return {
        name: boundary_interface_flux(
            head=head,
            boundary_mask=mask,
            active=active,
            tx_p=tx_p,
            tx_m=tx_m,
            ty_p=ty_p,
            ty_m=ty_m,
            tz_p=tz_p,
            tz_m=tz_m,
        )
        for name, mask in named_masks.items()
    }

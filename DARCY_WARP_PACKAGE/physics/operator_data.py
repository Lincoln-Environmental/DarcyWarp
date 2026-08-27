# SPDX-License-Identifier: AGPL-3.0-only
"""Typed, zero-copy descriptions of a structured 2D Darcy operator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class GridSpec:
    """Grid metadata shared by every solver backend."""

    nx: int
    ny: int
    dx: float
    device: str


@dataclass(frozen=True, slots=True)
class OperatorFields:
    """Borrowed host/device operator arrays; no array is copied or owned here."""

    transmissivity: Any
    recharge: Any
    head: Any
    rhs: Any


@dataclass(frozen=True, slots=True)
class BoundaryFields:
    """Borrowed active, fixed-head, and GHB operator data."""

    active: Any
    dirichlet_mask: Any
    dirichlet_values: Any
    ghb_mask: Any
    ghb_factor: Any


@dataclass(frozen=True, slots=True)
class StorageState:
    """Borrowed transient storage data for the currently staged operator."""

    diagonal: Any
    active: bool


def normalize_scalar_or_grid_to_shape(
    value: float | np.ndarray,
    *,
    shape: tuple[int, int],
    name: str,
) -> tuple[np.ndarray, str]:
    """Normalize scalar or grid parameters without changing their values."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        scalar = float(arr)
        if not np.isfinite(scalar):
            raise ValueError(f"{name} scalar must be finite.")
        return np.full(shape, scalar, dtype=np.float64), "scalar"
    if arr.ndim == 2 and arr.shape == shape:
        return arr.astype(np.float64, copy=False), "grid"
    if arr.ndim == 3 and arr.shape[0] == 1 and tuple(arr.shape[1:]) == shape:
        return np.asarray(arr[0], dtype=np.float64), "grid"
    raise ValueError(f"{name} must be a scalar or shape {shape}. Got {arr.shape}.")


def compute_ghb_factor_from_raw_fields(
    *,
    gh_mask: np.ndarray,
    gh_width: np.ndarray,
    gh_alpha: float | np.ndarray,
    aq_thickness: float | np.ndarray,
    dx: float,
    active: np.ndarray | None = None,
    bc_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    """Compute the 2D GHB conductance factor used by the discrete operator."""
    gh_mask_i = np.asarray(gh_mask, dtype=np.int32)
    gh_width_f = np.asarray(gh_width, dtype=np.float64)
    shape = tuple(gh_mask_i.shape)
    if gh_width_f.shape != shape:
        raise ValueError(f"gh_width shape {gh_width_f.shape} must match gh_mask shape {shape}.")
    gh_alpha_grid, gh_alpha_mode = normalize_scalar_or_grid_to_shape(
        gh_alpha, shape=shape, name="gh_alpha"
    )
    aq_thickness_grid, aq_mode = normalize_scalar_or_grid_to_shape(
        aq_thickness, shape=shape, name="aq_thickness"
    )
    gh_on = gh_mask_i != 0
    if active is not None:
        gh_on &= np.asarray(active, dtype=np.int32) != 0
    if bc_mask is not None:
        gh_on &= np.asarray(bc_mask, dtype=np.int32) == 0
    gh_on &= np.isfinite(gh_width_f) & (gh_width_f > 0.0)
    if np.any(gh_on):
        thick_used = np.asarray(aq_thickness_grid[gh_on], dtype=np.float64)
        alpha_used = np.asarray(gh_alpha_grid[gh_on], dtype=np.float64)
        if np.any((~np.isfinite(thick_used)) | (thick_used <= 0.0)):
            raise ValueError("aq_thickness must be finite and > 0 on active GHB cells.")
        if np.any((~np.isfinite(alpha_used)) | (alpha_used <= 0.0)):
            raise ValueError("gh_alpha must be finite and > 0 on active GHB cells.")
    ghb_factor = np.zeros(shape, dtype=np.float64)
    if np.any(gh_on):
        ghb_factor[gh_on] = (
            gh_alpha_grid[gh_on] * gh_width_f[gh_on] * float(dx) / aq_thickness_grid[gh_on]
        )
    return ghb_factor, gh_alpha_grid, aq_thickness_grid, {
        "gh_alpha": gh_alpha_mode,
        "aq_thickness": aq_mode,
    }

# SPDX-License-Identifier: AGPL-3.0-only
"""Physical rediscretization rules for the experimental 2D FAS hierarchy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class FASPhysicalLevel2D:
    level_id: int
    nx: int
    ny: int
    dx: float
    conductivity: np.ndarray
    top: np.ndarray
    bottom: np.ndarray
    has_top: bool
    active: np.ndarray
    active_fraction: np.ndarray
    dirichlet_mask: np.ndarray
    dirichlet_values: np.ndarray
    source_rate: np.ndarray
    ghb_mask: np.ndarray
    ghb_factor: np.ndarray
    ghb_external_head: np.ndarray
    sy: np.ndarray
    ss: np.ndarray
    previous_head: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    @property
    def area(self) -> float:
        return float(self.dx) * float(self.dx)


def _as_field(value: Any, *, shape: tuple[int, int], dtype: Any) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 0:
        return np.full(shape, array.reshape(()), dtype=dtype)
    if array.shape != shape:
        raise ValueError(f"field shape {array.shape} expected {shape}.")
    return np.ascontiguousarray(array)


def make_fine_physical_level(
    *,
    conductivity: Any,
    top: Any | None,
    bottom: Any,
    active: Any,
    dirichlet_mask: Any,
    dirichlet_values: Any,
    source_rate: Any,
    ghb_mask: Any,
    ghb_factor: Any,
    ghb_external_head: Any,
    sy: Any,
    ss: Any,
    previous_head: Any,
    dx: float,
) -> FASPhysicalLevel2D:
    bottom_array = np.asarray(bottom, dtype=np.float64)
    if bottom_array.ndim != 2:
        raise ValueError("bottom must be a two-dimensional field.")
    shape = bottom_array.shape
    top_array = np.zeros(shape, dtype=np.float64) if top is None else _as_field(top, shape=shape, dtype=np.float64)
    active_array = _as_field(active, shape=shape, dtype=np.int32)
    return FASPhysicalLevel2D(
        level_id=0,
        nx=shape[1],
        ny=shape[0],
        dx=float(dx),
        conductivity=_as_field(conductivity, shape=shape, dtype=np.float64),
        top=top_array,
        bottom=np.ascontiguousarray(bottom_array),
        has_top=top is not None,
        active=active_array,
        active_fraction=(active_array != 0).astype(np.float64),
        dirichlet_mask=_as_field(dirichlet_mask, shape=shape, dtype=np.int32),
        dirichlet_values=_as_field(dirichlet_values, shape=shape, dtype=np.float64),
        source_rate=_as_field(source_rate, shape=shape, dtype=np.float64),
        ghb_mask=_as_field(ghb_mask, shape=shape, dtype=np.int32),
        ghb_factor=_as_field(ghb_factor, shape=shape, dtype=np.float64),
        ghb_external_head=_as_field(ghb_external_head, shape=shape, dtype=np.float64),
        sy=_as_field(sy, shape=shape, dtype=np.float64),
        ss=_as_field(ss, shape=shape, dtype=np.float64),
        previous_head=_as_field(previous_head, shape=shape, dtype=np.float64),
    )


def _pad_to_even(array: np.ndarray, even_shape: tuple[int, int]) -> np.ndarray:
    """Zero-pad a 2D field to even dimensions (padded cells are inactive)."""
    pad_y = even_shape[0] - array.shape[0]
    pad_x = even_shape[1] - array.shape[1]
    if pad_y == 0 and pad_x == 0:
        return array
    return np.pad(array, ((0, pad_y), (0, pad_x)), mode="constant", constant_values=0)


def coarsen_physical_level(fine: FASPhysicalLevel2D, *, min_sat: float) -> FASPhysicalLevel2D:
    """Rediscretize one level with explicit intensive/extensive conventions.

    Vectorized 2x2 block reduction: fine fields are zero-padded to even
    dimensions (padded cells are inactive and drop out of every masked
    reduction), reshaped to (ny_c, 2, nx_c, 2) blocks and reduced over the
    block axes in row-major order — the same summation order and semantics as
    the original per-block loop.
    """
    ny_c = (fine.ny + 1) // 2
    nx_c = (fine.nx + 1) // 2
    shape_c = (ny_c, nx_c)
    even = (2 * ny_c, 2 * nx_c)

    def padded(field: np.ndarray) -> np.ndarray:
        return _pad_to_even(np.asarray(field), even).reshape(ny_c, 2, nx_c, 2)

    def reduce_sum(blocks: np.ndarray) -> np.ndarray:
        return blocks.sum(axis=(1, 3))

    active_b = padded(fine.active != 0)
    count = reduce_sum(active_b)
    fraction_c = count.astype(np.float64) / 4.0
    active_c = (count > 0).astype(np.int32)
    safe_count = np.maximum(count, 1)

    def active_mean(field: np.ndarray) -> np.ndarray:
        total = reduce_sum(padded(field) * active_b)
        return np.where(count > 0, total / safe_count, 0.0)

    K_b = padded(fine.conductivity)
    any_zero_K = reduce_sum(active_b & (K_b <= 0.0)) > 0
    # The active-weighted harmonic mean is zero if any participating
    # active cell is impermeable; silently dropping K=0 would create an
    # artificial coarse connection.
    inv_sum = reduce_sum(np.where(active_b & (K_b > 0.0), 1.0 / np.maximum(K_b, 1.0e-300), 0.0))
    K_c = np.where((count > 0) & ~any_zero_K, count / np.maximum(inv_sum, 1.0e-300), 0.0)

    top_c = active_mean(fine.top) if fine.has_top else np.zeros(shape_c, dtype=np.float64)
    bottom_c = active_mean(fine.bottom)
    sy_c = active_mean(fine.sy)
    ss_c = active_mean(fine.ss)
    previous_c = active_mean(fine.previous_head)

    # Source is an extensive transfer: coarse_rate * (2dx)^2 equals
    # the sum of active/free fine rates * dx^2, including odd edge blocks.
    free_b = active_b & ~padded(fine.dirichlet_mask != 0)
    source_c = reduce_sum(padded(fine.source_rate) * free_b) / 4.0

    bc_b = active_b & padded(fine.dirichlet_mask != 0)
    bc_count = reduce_sum(bc_b)
    bc_c = (bc_count > 0).astype(np.int32)
    bc_values_c = np.where(bc_count > 0, reduce_sum(padded(fine.dirichlet_values) * bc_b) / np.maximum(bc_count, 1), 0.0)

    gh_b = active_b & padded(fine.ghb_mask != 0) & (padded(fine.ghb_factor) > 0.0)
    if fine.has_top:
        sat_ref_b = np.maximum(padded(fine.top) - padded(fine.bottom), float(min_sat))
    else:
        sat_ref_b = np.ones((ny_c, 2, nx_c, 2), dtype=np.float64)
    conductance_b = np.where(gh_b, K_b * sat_ref_b * padded(fine.ghb_factor), 0.0)
    aggregate = reduce_sum(conductance_b)
    coarse_sat_ref = np.maximum(top_c - bottom_c, float(min_sat)) if fine.has_top else 1.0
    coarse_T_ref = K_c * coarse_sat_ref
    gh_ok = (aggregate > 0.0) & (coarse_T_ref > 0.0)
    gh_mask_c = gh_ok.astype(np.int32)
    gh_factor_c = np.where(gh_ok, aggregate / np.maximum(coarse_T_ref, 1.0e-300), 0.0)
    gh_head_c = np.where(
        aggregate > 0.0,
        reduce_sum(conductance_b * padded(fine.ghb_external_head)) / np.maximum(aggregate, 1.0e-300),
        0.0,
    )

    return FASPhysicalLevel2D(
        level_id=fine.level_id + 1,
        nx=nx_c,
        ny=ny_c,
        dx=float(fine.dx) * 2.0,
        conductivity=K_c,
        top=top_c,
        bottom=bottom_c,
        has_top=fine.has_top,
        active=active_c,
        active_fraction=fraction_c,
        dirichlet_mask=bc_c,
        dirichlet_values=bc_values_c,
        source_rate=source_c,
        ghb_mask=gh_mask_c,
        ghb_factor=gh_factor_c,
        ghb_external_head=gh_head_c,
        sy=sy_c,
        ss=ss_c,
        previous_head=previous_c,
    )


def build_fas_physical_hierarchy(
    fine: FASPhysicalLevel2D,
    *,
    max_levels: int,
    min_coarse_cells: int,
    min_sat: float,
) -> list[FASPhysicalLevel2D]:
    levels = [fine]
    while len(levels) < int(max_levels):
        current = levels[-1]
        if current.nx <= 2 or current.ny <= 2:
            break
        candidate_nx = (current.nx + 1) // 2
        candidate_ny = (current.ny + 1) // 2
        if candidate_nx * candidate_ny < int(min_coarse_cells):
            break
        levels.append(coarsen_physical_level(current, min_sat=min_sat))
    return levels


__all__ = [
    "FASPhysicalLevel2D",
    "build_fas_physical_hierarchy",
    "coarsen_physical_level",
    "make_fine_physical_level",
]

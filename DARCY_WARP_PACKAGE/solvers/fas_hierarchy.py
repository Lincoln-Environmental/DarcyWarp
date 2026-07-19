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


def _blocks(shape: tuple[int, int]):
    ny, nx = shape
    for cj in range((ny + 1) // 2):
        for ci in range((nx + 1) // 2):
            yield cj, ci, slice(2 * cj, min(2 * cj + 2, ny)), slice(2 * ci, min(2 * ci + 2, nx))


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


def coarsen_physical_level(fine: FASPhysicalLevel2D, *, min_sat: float) -> FASPhysicalLevel2D:
    """Rediscretize one level with explicit intensive/extensive conventions."""
    ny_c = (fine.ny + 1) // 2
    nx_c = (fine.nx + 1) // 2
    shape_c = (ny_c, nx_c)
    zeros = np.zeros(shape_c, dtype=np.float64)
    K_c = zeros.copy()
    top_c = zeros.copy()
    bottom_c = zeros.copy()
    active_c = np.zeros(shape_c, dtype=np.int32)
    fraction_c = zeros.copy()
    bc_c = np.zeros(shape_c, dtype=np.int32)
    bc_values_c = zeros.copy()
    source_c = zeros.copy()
    gh_mask_c = np.zeros(shape_c, dtype=np.int32)
    gh_factor_c = zeros.copy()
    gh_head_c = zeros.copy()
    sy_c = zeros.copy()
    ss_c = zeros.copy()
    previous_c = zeros.copy()

    for cj, ci, js, is_ in _blocks(fine.shape):
        active_block = fine.active[js, is_] != 0
        count = int(np.count_nonzero(active_block))
        fraction_c[cj, ci] = float(count) / 4.0
        if count == 0:
            continue
        active_c[cj, ci] = 1
        K_values = fine.conductivity[js, is_][active_block]
        # The active-weighted harmonic mean is zero if any participating
        # active cell is impermeable; silently dropping K=0 would create an
        # artificial coarse connection.
        if np.all(K_values > 0.0):
            K_c[cj, ci] = float(count) / float(np.sum(1.0 / K_values))
        top_c[cj, ci] = float(np.mean(fine.top[js, is_][active_block])) if fine.has_top else 0.0
        bottom_c[cj, ci] = float(np.mean(fine.bottom[js, is_][active_block]))
        sy_c[cj, ci] = float(np.mean(fine.sy[js, is_][active_block]))
        ss_c[cj, ci] = float(np.mean(fine.ss[js, is_][active_block]))
        previous_c[cj, ci] = float(np.mean(fine.previous_head[js, is_][active_block]))

        # Source is an extensive transfer: coarse_rate * (2dx)^2 equals
        # the sum of active/free fine rates * dx^2, including odd edge blocks.
        free_block = active_block & (fine.dirichlet_mask[js, is_] == 0)
        source_c[cj, ci] = float(np.sum(fine.source_rate[js, is_][free_block])) / 4.0

        bc_block = active_block & (fine.dirichlet_mask[js, is_] != 0)
        if np.any(bc_block):
            bc_c[cj, ci] = 1
            bc_values_c[cj, ci] = float(np.mean(fine.dirichlet_values[js, is_][bc_block]))

        gh_block = active_block & (fine.ghb_mask[js, is_] != 0) & (fine.ghb_factor[js, is_] > 0.0)
        if np.any(gh_block):
            if fine.has_top:
                sat_ref = np.maximum(fine.top[js, is_] - fine.bottom[js, is_], float(min_sat))
            else:
                sat_ref = np.ones_like(fine.bottom[js, is_])
            conductance = fine.conductivity[js, is_] * sat_ref * fine.ghb_factor[js, is_]
            conductance = np.where(gh_block, conductance, 0.0)
            aggregate = float(np.sum(conductance))
            coarse_sat_ref = (
                max(top_c[cj, ci] - bottom_c[cj, ci], float(min_sat))
                if fine.has_top else 1.0
            )
            coarse_T_ref = K_c[cj, ci] * coarse_sat_ref
            if aggregate > 0.0 and coarse_T_ref > 0.0:
                gh_mask_c[cj, ci] = 1
                gh_factor_c[cj, ci] = aggregate / coarse_T_ref
                gh_head_c[cj, ci] = float(
                    np.sum(conductance * fine.ghb_external_head[js, is_]) / aggregate
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

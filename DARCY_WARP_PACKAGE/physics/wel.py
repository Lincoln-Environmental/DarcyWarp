# SPDX-License-Identifier: AGPL-3.0-or-later
"""Explicit sparse volumetric well entries (MODFLOW-style prescribed-rate WEL).

Rates follow the MF6 sign convention: negative rates abstract water, positive
rates inject water and zero rates leave the entry inactive. Wells are always
specified as volumetric rates [L³/T] irrespective of model dimensionality; the
receiving solver converts them for numerical assembly. Entries are individually
retained so diagnostics never lose opposing wells that share a cell.
"""
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from DARCY_WARP_PACKAGE.physics.inputs_3d import floating_field


@dataclass(frozen=True)
class WellEntries:
    """Owned sparse well specification for one grid.

    :param cells: Integer cell indices of shape (n, ndim) in C order, one row
        per entry, ordered like the owning grid (e.g. (row, col) in 2D and
        (layer, row, col) in 3D).
    :param rates: Signed volumetric entry rates [L³/T] of shape (n,).
    """

    cells: np.ndarray
    rates: np.ndarray

    def __len__(self) -> int:
        return int(self.rates.shape[0])


def well_entries(*cells, rates) -> list[tuple[tuple[int, ...], float]]:
    """Build explicit entry pairs from parallel cell/rate sequences.

    :param cells: Cell index tuples, one per well, e.g. ``(0, 2)`` or ``(1, 3, 4)``.
    :param rates: One signed volumetric rate [L³/T] per cell.
    :return: Entry pairs accepted by every WEL interface.
    """
    indices = tuple(tuple(int(value) for value in cell) for cell in cells)
    values = floating_field(value=rates, name="wel_rates", dtype=np.float64)
    values = np.atleast_1d(values)
    if values.ndim != 1 or values.shape[0] != len(indices):
        raise ValueError("Supply exactly one volumetric rate per well entry")
    return [(cell, float(rate)) for cell, rate in zip(indices, values)]


def _entry_cells(entry, dimensionality):
    """Validate one cell index tuple against the requested dimensionality."""
    if isinstance(entry, (int, np.integer)):
        raise ValueError("Well cells must be explicit index tuples, not scalars")
    cell = tuple(entry)
    if len(cell) != dimensionality or any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in cell
    ):
        raise ValueError(
            f"Each well cell must be an integer index tuple of length {dimensionality}"
        )
    return tuple(int(value) for value in cell)


def validate_well_entries(*, entries, shape, active, dtype=np.float64):
    """Validate sparse well entries and take ownership of their values.

    :param entries: None, a sequence of ``(cell, rate)`` pairs, a mapping from
        cell tuples to rates, or an existing :class:`WellEntries` value.
    :param shape: Owning grid shape; entry indices must match its dimensionality.
    :param active: Binary active-cell grid with ``shape``. Wells on inactive
        cells are rejected instead of being silently discarded.
    :param dtype: Floating-point storage type of the owned rates.
    :return: Owned :class:`WellEntries`, or None when no wells are supplied.
    :raises ValueError: If indices, dimensionality, rates or cells are invalid.
    """
    if entries is None:
        return None
    shape = tuple(int(size) for size in shape)
    if isinstance(entries, WellEntries):
        if entries.cells.shape[1:] != (len(shape),):
            raise ValueError(f"Well cells must have {len(shape)} indices per entry")
        pairs = list(zip(map(tuple, entries.cells.tolist()), entries.rates.tolist()))
    elif isinstance(entries, Mapping):
        pairs = list(entries.items())
    elif isinstance(entries, (str, bytes)) or not isinstance(entries, Iterable):
        raise ValueError("wel_entries must be None, entry pairs or a cell-to-rate mapping")
    else:
        pairs = list(entries)
    if not pairs:
        return None

    parsed = []
    for item in pairs:
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 2:
            raise ValueError("Each well entry must be a (cell, rate) pair")
        cell, rate = item
        parsed.append((_entry_cells(cell, len(shape)), rate))

    cells = np.asarray([cell for cell, unused in parsed], dtype=np.int64)
    rates = np.asarray([rate for unused, rate in parsed], dtype=np.float64)
    if not np.all(np.isfinite(rates)):
        raise ValueError("Well rates must be finite")
    for axis, size in enumerate(shape):
        if np.any((cells[:, axis] < 0) | (cells[:, axis] >= size)):
            raise ValueError(f"Well index {axis} outside grid dimension {size}")
    active_grid = np.asarray(active)
    if active_grid.shape != shape:
        raise ValueError(f"active shape {active_grid.shape} does not match {shape}")
    if np.any(active_grid[tuple(cells.T)] == 0):
        raise ValueError("Well entries on inactive cells are rejected; remove them explicitly")
    return WellEntries(cells=cells.astype(np.int64), rates=rates.astype(dtype))


def well_cell_volume_rates(*, entries, shape, dtype=np.float64):
    """Accumulate entry rates into a dense volumetric cell source.

    :param entries: Validated :class:`WellEntries` or None.
    :param shape: Owning grid shape.
    :param dtype: Floating-point storage type.
    :return: Owned dense signed rates [L³/T], or None without wells.
    """
    if entries is None or not len(entries):
        return None
    source = np.zeros(tuple(int(size) for size in shape), dtype=dtype)
    np.add.at(source, tuple(entries.cells.T), entries.rates)
    if not np.all(np.isfinite(source)):
        raise ValueError("Accumulated well rates are not representable")
    return source


def combine_volumetric_sources(*, rhs, wells, dtype=np.float64):
    """Add sparse well rates to ordinary volumetric sources without aliasing.

    :param rhs: Dense volumetric cell rates [L³/T], full grid or None.
    :param wells: Dense accumulated well rates from well_cell_volume_rates, or None.
    :param dtype: Floating-point storage type.
    :return: Owned dense combined rates [L³/T], or None without any source.
    """
    if rhs is None and wells is None:
        return None
    if rhs is None:
        return np.array(wells, dtype=dtype, copy=True)
    if wells is None:
        return np.array(rhs, dtype=dtype, copy=True)
    if rhs.shape != wells.shape:
        raise ValueError(f"Well source shape {wells.shape} does not match {rhs.shape}")
    combined = np.array(rhs, dtype=dtype, copy=True)
    combined += wells
    return combined


def well_budget(*, rates):
    """Gross-preserving physical budget of signed well rates.

    :param rates: One owned entry-rate array [L³/T], or None.
    :return: Mapping with ``inflow``, ``outflow`` and ``net`` rates [L³/T].
        Inflow and outflow keep opposing wells in one cell physically separate.
    """
    if rates is None or not len(rates):
        return dict(inflow=0., outflow=0., net=0.)
    values = np.asarray(rates, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("Well rates must be finite")
    inflow = float(np.sum(np.maximum(values, 0.)))
    outflow = float(np.sum(np.maximum(-values, 0.)))
    return dict(inflow=inflow, outflow=outflow, net=inflow - outflow)

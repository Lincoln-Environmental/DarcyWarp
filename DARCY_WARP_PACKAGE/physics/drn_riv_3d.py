# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reference DRN/RIV physics for structured 3D groundwater cells.

Rates are positive out of the aquifer. Conductance is integrated [L2/T].
The two packages retain separate fluxes even when they occupy the same cell.
Sparse entry lists allow multiple reaches or drains in one cell; each entry's
physical flux is evaluated independently before the cell contributions sum.
"""
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from DARCY_WARP_PACKAGE.physics.inputs_3d import binary_mask, floating_field
from DARCY_WARP_PACKAGE.validation.head_checks import invalid_head_values


@dataclass(frozen=True)
class BoundaryTerms3D:
    """Physical cell discharge and its piecewise-linear operator terms.

    :param discharge: Signed aquifer outflow [L3/T].
    :param diagonal: Derivative of outward discharge with respect to head [L2/T].
    :param rhs: Boundary forcing [L3/T] added to the groundwater RHS.
    """

    discharge: np.ndarray
    diagonal: np.ndarray
    rhs: np.ndarray


@dataclass(frozen=True)
class BoundaryEntryList:
    """Owned sparse package entries grouped per groundwater cell in CSR order.

    :param cells: Integer (m, 3) entry cells ``(layer, row, column)`` sorted in
        C order so equal cells are contiguous.
    :param conductance: Integrated entry conductances [L2/T] of shape (m,).
    :param first_level: Entry activation elevation (DRN) [L] of shape (m,).
    :param second_level: Entry stage (RIV) [L] of shape (m,).
    :param third_level: Entry bed bottom (RIV) [L] of shape (m,).
    :param offsets: CSR cell offsets of shape (n + 1) into the entry arrays.
    :param flat_cells: C-order flat cell index of each entry, shape (m,).
    """

    cells: np.ndarray
    conductance: np.ndarray
    first_level: np.ndarray | None
    second_level: np.ndarray | None
    third_level: np.ndarray | None
    offsets: np.ndarray
    flat_cells: np.ndarray

    def __len__(self) -> int:
        return int(self.conductance.shape[0])

    def __iter__(self):
        """Expose canonical records for inspection without surrendering ownership."""
        for index, cell in enumerate(self.cells):
            prefix = (tuple(int(value) for value in cell),
                      float(self.conductance[index]))
            if self.first_level is not None:
                yield (*prefix, float(self.first_level[index]))
            else:
                yield (*prefix, float(self.second_level[index]),
                       float(self.third_level[index]))


def _entry_cell_index(cell, shape):
    """Validate one integer entry cell tuple against the grid shape."""
    try:
        values = tuple(cell)
    except TypeError as error:
        raise ValueError('Entry cells must be integer index tuples') from error
    if len(values) != len(shape) or any(isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer)) for value in values):
        raise ValueError(f"Each entry cell must be an integer index tuple of length {len(shape)}")
    index = tuple(int(value) for value in values)
    for axis, (value, size) in enumerate(zip(index, shape)):
        if value < 0 or value >= size:
            raise ValueError(f"Entry index {axis} outside grid dimension {size}")
    return index


def _entry_records(entries, shape, level_count):
    """Normalise entry inputs into (cell, conductance, levels) records."""
    if isinstance(entries, BoundaryEntryList):
        cells = np.asarray(entries.cells)
        if cells.ndim != 2 or cells.shape[1:] != (len(shape),):
            raise ValueError(f"Entry cells must have {len(shape)} indices per entry")
        if (np.issubdtype(cells.dtype, np.bool_)
                or not np.issubdtype(cells.dtype, np.integer)):
            raise ValueError("Entry cells must use an integer coordinate dtype")
        count = cells.shape[0]
        conductance = np.asarray(entries.conductance)
        if conductance.ndim != 1 or conductance.shape[0] != count:
            raise ValueError("Entry conductance must have one value per cell")
        if level_count == 1:
            raw_levels = (entries.first_level,)
            if entries.second_level is not None or entries.third_level is not None:
                raise ValueError("DRN entries carry exactly one elevation array")
        else:
            raw_levels = (entries.second_level, entries.third_level)
            if entries.first_level is not None:
                raise ValueError("RIV entries carry exactly stage and bottom arrays")
        levels = []
        for level in raw_levels:
            if level is None:
                raise ValueError("Entry level arrays are incomplete")
            values = np.asarray(level)
            if values.ndim != 1 or values.shape[0] != count:
                raise ValueError("Entry levels must have one value per cell")
            levels.append(values)
        flat_cells = np.asarray(entries.flat_cells)
        if flat_cells.ndim != 1 or flat_cells.shape[0] != count:
            raise ValueError("Entry flat-cell indices must match the entry count")
        validated_cells = [_entry_cell_index(cell, shape) for cell in cells.tolist()]
        return [(cell, float(conductance[index]),
                 *[float(level[index]) for level in levels])
                for index, cell in enumerate(validated_cells)]
    if isinstance(entries, Mapping):
        records = []
        for cell, values in entries.items():
            values = tuple(np.atleast_1d(np.asarray(values, dtype=np.float64)).ravel())
            records.append((_entry_cell_index(cell, shape), *values))
        return records
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Iterable):
        raise ValueError("entries must be a sequence of (cell, conductance, levels) records")
    records = []
    for item in entries:
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != level_count + 2:
            raise ValueError(f"Each entry must carry a cell, conductance and {level_count} level(s)")
        records.append((_entry_cell_index(item[0], shape),
                        *[float(value) for value in item[1:]]))
    return records


def validate_boundary_entries(*, entries, package, shape, active, fixed):
    """Validate sparse package entries and group them per cell in CSR order.

    :param entries: Sequence of ``(cell, conductance, levels...)`` records, a
        cell-to-values mapping, or an existing :class:`BoundaryEntryList`.
    :param package: ``"DRN"`` or ``"RIV"``; selects the two required levels.
    :param shape: Owning (nz, ny, nx) grid shape.
    :param active: Binary active-cell grid.
    :param fixed: Binary prescribed-head grid.
    :return: Owned :class:`BoundaryEntryList`, or None without entries.
    :raises ValueError: If cells, values, levels or cells' usability are invalid.
    """
    if entries is None:
        return None
    shape = tuple(int(size) for size in shape)
    level_count = 1 if package == "DRN" else 2
    records = _entry_records(entries, shape, level_count)
    if not records:
        return None
    cells = np.asarray([record[0] for record in records], dtype=np.int64)
    conductance = np.asarray([record[1] for record in records], dtype=np.float64)
    levels = [np.asarray([record[2 + axis] for record in records], dtype=np.float64)
              for axis in range(level_count)]
    if not np.all(np.isfinite(conductance)) or np.any(conductance < 0.):
        raise ValueError(f"{package} entry conductance must be finite and nonnegative")
    for level in levels:
        if not np.all(np.isfinite(level)):
            raise ValueError(f"{package} entry levels must be finite")
        if np.any(invalid_head_values(head=level[conductance > 0.])):
            raise ValueError(f"Invalid conducting {package} entry level")
    if package == "RIV" and np.any((conductance > 0.) & (levels[0] < levels[1])):
        raise ValueError("RIV entry stage must be at or above its bed bottom")
    active_grid = binary_mask(value=active, name="active", shape=shape)
    fixed_grid = binary_mask(value=fixed, name="fixed", shape=shape)
    selected = active_grid[tuple(cells.T)] != 0
    if np.any(~selected):
        raise ValueError(f"{package} entries on inactive cells are rejected")
    if np.any(fixed_grid[tuple(cells.T)] != 0):
        raise ValueError(f"{package} entries cannot occupy prescribed-head cells")
    order = np.lexsort(tuple(cells[:, axis] for axis in reversed(range(len(shape)))))
    cells = cells[order]
    conductance = conductance[order]
    levels = [level[order] for level in levels]
    flat = np.ravel_multi_index(tuple(cells.T), shape)
    counts = np.bincount(flat, minlength=int(np.prod(shape)))
    offsets = np.zeros(len(counts) + 1, dtype=np.int32)
    np.cumsum(counts, out=offsets[1:])
    if package == "DRN":
        first, second, third = levels[0], None, None
    else:
        first, second, third = None, levels[0], levels[1]
    return BoundaryEntryList(cells=cells.astype(np.int64), conductance=conductance,
        first_level=first, second_level=second, third_level=third,
        offsets=offsets, flat_cells=flat.astype(np.int64))


def csr_from_boundary_field(*, conductance, levels, shape):
    """Convert owned dense single-entry fields into CSR entry form.

    :param conductance: Dense conductance grid; every conducting cell becomes
        exactly one entry, so dense models keep identical physics.
    :param levels: Dense level grids matching the package (elevation for DRN,
        stage and bottom for RIV).
    :param shape: Owning grid shape.
    :return: Owned :class:`BoundaryEntryList` with one entry per conducting cell.
    """
    c = np.broadcast_to(np.asarray(conductance, dtype=np.float64), shape)
    flat_c = c.ravel()
    flat = np.flatnonzero(flat_c > 0.)
    if flat.size == 0:
        return None
    dense_levels = [np.broadcast_to(np.asarray(level, dtype=np.float64), shape).ravel()[flat]
                    for level in levels]
    cells = np.stack(np.unravel_index(flat, shape), axis=1)
    counts = np.bincount(flat, minlength=int(np.prod(shape)))
    offsets = np.zeros(len(counts) + 1, dtype=np.int32)
    np.cumsum(counts, out=offsets[1:])
    if len(levels) == 1:
        first, second, third = dense_levels[0], None, None
    else:
        first, second, third = None, dense_levels[0], dense_levels[1]
    return BoundaryEntryList(cells=cells.astype(np.int64), conductance=flat_c[flat].copy(),
        first_level=first, second_level=second, third_level=third,
        offsets=offsets, flat_cells=flat.astype(np.int64))


def csr_entry_terms(*, head_flat, entries, package, shape):
    """Evaluate every entry's flux independently and sum the cell contributions.

    :param head_flat: C-order flat groundwater heads.
    :param entries: Owned :class:`BoundaryEntryList`.
    :param package: ``"DRN"`` or ``"RIV"`` flux law.
    :param shape: Owning (nz, ny, nx) grid shape.
    :return: ``(entry_discharge, cell_terms)`` with signed per-entry discharge
        [L3/T] and a :class:`BoundaryTerms3D` of summed cell contributions.
    """
    n = int(np.prod(shape))
    m = len(entries)
    entry_discharge = np.zeros(m, dtype=np.float64)
    entry_tangent = np.zeros(m, dtype=np.float64)
    entry_rhs = np.zeros(m, dtype=np.float64)
    h = head_flat[entries.flat_cells]
    c = entries.conductance
    enabled = c > 0.
    if package == "DRN":
        threshold = entries.first_level
        active = enabled & (h > threshold)
        entry_discharge[active] = c[active] * (h[active] - threshold[active])
        entry_tangent[active] = c[active]
        entry_rhs[active] = c[active] * threshold[active]
    else:
        stage, bottom = entries.second_level, entries.third_level
        coupled = enabled & (h > bottom)
        reference = np.maximum(h, bottom)
        entry_discharge[enabled] = c[enabled] * (reference[enabled] - stage[enabled])
        entry_tangent[coupled] = c[coupled]
        entry_rhs[coupled] = c[coupled] * stage[coupled]
        disconnected = enabled & ~coupled
        entry_rhs[disconnected] = c[disconnected] * (stage[disconnected] - bottom[disconnected])
    discharge = np.zeros(n, dtype=np.float64)
    diagonal = np.zeros(n, dtype=np.float64)
    rhs = np.zeros(n, dtype=np.float64)
    np.add.at(discharge, entries.flat_cells, entry_discharge)
    np.add.at(diagonal, entries.flat_cells, entry_tangent)
    np.add.at(rhs, entries.flat_cells, entry_rhs)
    if not (np.all(np.isfinite(entry_discharge)) and np.all(np.isfinite(diagonal))
            and np.all(np.isfinite(rhs))):
        raise ValueError("Boundary entry terms are not representable")
    terms = BoundaryTerms3D(discharge.reshape(shape), diagonal.reshape(shape), rhs.reshape(shape))
    return entry_discharge, terms


def _owned_fields(*, package, conductance, levels, active, fixed, mask):
    """Validate and own one integrated boundary of each type per cell.

    :param package: Package name used in validation errors.
    :param conductance: Scalar or full-grid integrated conductance.
    :param levels: Mapping of elevation field names to scalars or full grids.
    :param active: Binary active-cell grid.
    :param fixed: Binary prescribed-head grid.
    :param mask: Optional binary package mask.
    :return: Conductance and a dictionary of owned elevation fields.
    """
    shape = np.asarray(active).shape
    if len(shape) != 3 or any(size == 0 for size in shape):
        raise ValueError("Boundary fields require a non-empty (nz, ny, nx) grid")
    active = binary_mask(value=active, name="active", shape=shape)
    fixed = binary_mask(value=fixed, name="fixed", shape=shape)

    if conductance is None and all(value is None for value in levels.values()) and mask is None:
        return None, {name: None for name in levels}
    if conductance is None or any(value is None for value in levels.values()):
        raise ValueError(f"{package} requires conductance and all elevation fields")

    fields = {}
    for name, value in (("conductance", conductance), *levels.items()):
        field = floating_field(value=value, name=f"{package}_{name}")
        if field.ndim != 0 and field.shape != shape:
            raise ValueError(f"{package}_{name} must be scalar or have shape {shape}")
        fields[name] = np.broadcast_to(field, shape).copy()

    c = fields.pop("conductance")
    if np.any(c < 0.0):
        raise ValueError(f"{package} conductance must be nonnegative")
    selected = c > 0.0 if mask is None else binary_mask(
        value=mask, name=f"{package}_mask", shape=shape
    ) != 0
    if np.any(selected & ((active == 0) | (fixed != 0))):
        raise ValueError(f"{package} cells must be active and cannot have prescribed heads")
    if mask is not None and np.asarray(conductance).ndim != 0 and np.any((c != 0.0) & ~selected):
        raise ValueError(f"Nonzero {package} conductance outside mask would be lost")
    c[~selected] = 0.0
    for name, field in fields.items():
        field[~selected] = 0.0
        if np.any(invalid_head_values(head=field[c > 0.0])):
            raise ValueError(f"Invalid conducting {package} {name}")
    return c, fields


def validate_drn_fields(*, conductance, elevation, active, fixed, mask=None):
    """Validate and copy structured 3D drain inputs.

    :param conductance: Integrated drain conductance [L2/T], scalar or full grid.
    :param elevation: Drain activation elevation [L], scalar or full grid.
    :param active: Binary active-cell grid (nz, ny, nx).
    :param fixed: Binary prescribed-head grid.
    :param mask: Optional binary drain mask; zero conductance disables exchange.
    :return: Owned conductance and elevation arrays, or (None, None).
    """
    c, fields = _owned_fields(package="DRN", conductance=conductance,
                              levels={"elevation": elevation}, active=active,
                              fixed=fixed, mask=mask)
    e = fields["elevation"]
    if c is not None:
        with np.errstate(over="ignore", invalid="ignore"):
            _require_finite(c * e)
    return c, e


def validate_riv_fields(*, conductance, stage, bottom, active, fixed, mask=None):
    """Validate and copy structured 3D river inputs.

    :param conductance: Integrated riverbed conductance [L2/T].
    :param stage: River water surface elevation [L].
    :param bottom: Bottom elevation of the riverbed layer [L].
    :param active: Binary active-cell grid (nz, ny, nx).
    :param fixed: Binary prescribed-head grid.
    :param mask: Optional binary river mask; zero conductance disables exchange.
    :return: Owned conductance, stage and bottom arrays, or three None values.
    """
    c, fields = _owned_fields(package="RIV", conductance=conductance,
                              levels={"stage": stage, "bottom": bottom},
                              active=active, fixed=fixed, mask=mask)
    s, b = fields["stage"], fields["bottom"]
    if c is not None:
        if np.any((c > 0.0) & (s < b)):
            raise ValueError("RIV stage must be at or above riverbed bottom")
        with np.errstate(over="ignore", invalid="ignore"):
            _require_finite(c * s, c * (s - b))
    return c, s, b


def _require_finite(*fields):
    """Reject unrepresentable boundary terms instead of silently clipping.

    :param fields: Arrays of assembled boundary quantities.
    """
    if any(not np.all(np.isfinite(field)) for field in fields):
        raise ValueError("Boundary discharge or operator terms are not representable")


def _empty_terms(*, head, conductance):
    """Allocate terms and identify conducting cells without reading inactive heads.

    :param head: Current groundwater head grid.
    :param conductance: Validated conductance array or None.
    :return: Head, conducting mask and zero-initialised boundary terms.
    """
    h = np.asarray(head, dtype=np.float64)
    if h.ndim != 3:
        raise ValueError("Head must be a three-dimensional grid")
    enabled = np.zeros(h.shape, dtype=bool)
    if conductance is not None:
        if conductance.shape != h.shape:
            raise ValueError("Conductance must match head shape")
        enabled = conductance > 0.0
    if np.any(invalid_head_values(head=h[enabled])):
        raise ValueError("Invalid head in a conducting boundary cell")
    terms = BoundaryTerms3D(np.zeros_like(h), np.zeros_like(h), np.zeros_like(h))
    return h, enabled, terms


def drn_terms(*, head, conductance, elevation):
    """Evaluate DRN outflow and the exact active-branch diagonal/RHS.

    :param head: Current groundwater head [L], full 3D grid.
    :param conductance: Conductance from validate_drn_fields.
    :param elevation: Elevation from validate_drn_fields.
    :return: BoundaryTerms3D; zero derivative at the activation threshold.
    """
    h, enabled, terms = _empty_terms(head=head, conductance=conductance)
    if conductance is None:
        return terms
    flowing = enabled & (h > elevation)
    with np.errstate(over="ignore", invalid="ignore"):
        terms.discharge[flowing] = conductance[flowing] * (h[flowing] - elevation[flowing])
        terms.diagonal[flowing] = conductance[flowing]
        terms.rhs[flowing] = conductance[flowing] * elevation[flowing]
    _require_finite(terms.discharge, terms.diagonal, terms.rhs)
    return terms


def riv_terms(*, head, conductance, stage, bottom):
    """Evaluate RIV outflow, including constant leakage below the bed bottom.

    :param head: Current groundwater head [L], full 3D grid.
    :param conductance: Conductance from validate_riv_fields.
    :param stage: Stage from validate_riv_fields.
    :param bottom: Bottom from validate_riv_fields.
    :return: BoundaryTerms3D; zero derivative at and below the bottom.
    """
    h, enabled, terms = _empty_terms(head=head, conductance=conductance)
    if conductance is None:
        return terms
    coupled = enabled & (h > bottom)
    disconnected = enabled & ~coupled
    with np.errstate(over="ignore", invalid="ignore"):
        terms.discharge[enabled] = conductance[enabled] * (
            np.maximum(h[enabled], bottom[enabled]) - stage[enabled]
        )
        terms.diagonal[coupled] = conductance[coupled]
        terms.rhs[coupled] = conductance[coupled] * stage[coupled]
        terms.rhs[disconnected] = conductance[disconnected] * (
            stage[disconnected] - bottom[disconnected]
        )
    _require_finite(terms.discharge, terms.diagonal, terms.rhs)
    return terms

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
    indices = []
    for cell in cells:
        try:
            dimensionality = len(cell)
        except TypeError as error:
            raise ValueError("Well cells must be explicit index tuples") from error
        if dimensionality not in (2, 3):
            raise ValueError("Well cells must have two or three indices")
        indices.append(_entry_cells(entry=cell, dimensionality=dimensionality))
    if indices and any(len(cell) != len(indices[0]) for cell in indices):
        raise ValueError("Well cells must have consistent dimensionality")
    values = floating_field(value=rates, name="wel_rates", dtype=np.float64)
    values = np.atleast_1d(values)
    if values.ndim != 1 or values.shape[0] != len(indices):
        raise ValueError("Supply exactly one volumetric rate per well entry")
    return [(cell, float(rate)) for cell, rate in zip(indices, values)]


def _entry_cells(entry, dimensionality):
    """Validate one cell index tuple against the requested dimensionality."""
    if isinstance(entry, (int, np.integer)):
        raise ValueError("Well cells must be explicit index tuples, not scalars")
    try:
        cell = tuple(entry)
    except TypeError as error:
        raise ValueError('Well cells must be explicit integer index tuples') from error
    if len(cell) != dimensionality or any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) for value in cell
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
        scalar = floating_field(value=rate, name="wel_rate", dtype=dtype)
        if scalar.ndim != 0:
            raise ValueError("Each well entry requires exactly one scalar rate")
        indices = _entry_cells(entry=cell, dimensionality=len(shape))
        if any(index < 0 or index >= size for index, size in zip(indices, shape)):
            raise ValueError('Well cell outside the groundwater grid')
        parsed.append((indices, float(scalar)))

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
    cells, rates, unused_gross = sparse_well_cell_rates(entries=entries, shape=shape, dtype=dtype)
    source.ravel()[cells] = rates
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
    with np.errstate(over="ignore", invalid="ignore"):
        combined += wells
    if not np.all(np.isfinite(combined)):
        raise ValueError("Combined volumetric sources are not representable")
    return combined


def signed_rate_budget(*, rates):
    """Gross-preserving budget of a signed rate array.

    :param rates: Owned signed rates [L³/T] (well entries, recharge or raw
        volumetric sources), or None.
    :return: Mapping with ``inflow``, ``outflow`` and ``net`` rates. Inflow
        and outflow keep opposing contributions physically separate instead
        of cancelling them.
    """
    if rates is None or not len(rates):
        return dict(inflow=0., outflow=0., net=0.)
    values = np.asarray(rates, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("Well rates must be finite")
    inflow = float(np.sum(np.maximum(values, 0.)))
    outflow = float(np.sum(np.maximum(-values, 0.)))
    if inflow > 0. and outflow > 0.:
        from math import fsum
        net = fsum(values.ravel())
    else:
        net = inflow - outflow
    return dict(inflow=inflow, outflow=outflow, net=net)


@dataclass(frozen=True)
class AcceptedWellState:
    """Privately owned accepted 2D source outputs.

    :param head_bytes: Immutable accepted head values, without a state digest.
    :param shape: Groundwater head grid shape.
    :param rate_bytes: Immutable volumetric well entry rates.
    :param budgets: Immutable package budget tuples (name, inflow, outflow, net).
    :param dt: Accepted duration, or None for steady flow.
    :param cell_bytes: Immutable original WEL cell indices.
    :param head_owner: Optional independently owned resident accepted head.
    :param equation_identifier: Unique associated resident equation identifier.
    :param source_inputs: Owned raw/recharge inputs and actual physical multipliers.
    """

    head_bytes: bytes | None
    shape: tuple
    rate_bytes: bytes
    budgets: tuple
    dt: float | None
    cell_bytes: bytes
    head_owner: object | None = None
    equation_identifier: str | None = None
    source_inputs: tuple = ()


def validate_well_schedule(*, wel_rates, periods, shape, active, dtype):
    """Validate every scheduled WEL update before changing the model.

    :param wel_rates: None or one sparse entry specification per period.
    :param periods: Required number of periods.
    :param shape: Owning groundwater grid shape.
    :param active: Active-cell field.
    :param dtype: Destination precision.
    :return: Owned period entries, or None when built wells remain in effect.
    :raises ValueError: If a period or any entry is invalid.
    """
    if wel_rates is None:
        return None
    if not isinstance(wel_rates, (list, tuple)) or len(wel_rates) != periods:
        raise ValueError('wel_rates requires one sparse well specification per period')
    return [validate_well_entries(entries=entries, shape=shape, active=active, dtype=dtype)
            for entries in wel_rates]


def source_budgets_2d(*, model, recharge_field=None):
    """Separate physical recharge, WEL and compensating CHD exchange.

    :param model: Built 2D model owning recharge separately from WEL depth rates.
    :param recharge_field: Optional explicit mixed-path recharge depth field.
    :return: Gross volumetric rate budgets with the existing recharge convention.
    """
    active = np.asarray(model.active_host) != 0
    fixed = np.asarray(model.bc_mask_host) != 0
    recharge_input = model._recharge_input_host if recharge_field is None else recharge_field
    recharge = np.broadcast_to(np.asarray(recharge_input, dtype=np.float64), active.shape)[active & ~fixed]
    area = model.dx * model.dx
    entries = model.wel_entries
    rates = np.empty(0) if entries is None else entries.rates
    fixed_rates = np.empty(0)
    if entries is not None:
        fixed_rates = rates[fixed[tuple(entries.cells.T)]]
    return dict(recharge=signed_rate_budget(rates=recharge * area),
        wel=signed_rate_budget(rates=rates),
        prescribed_head_wel_exchange=signed_rate_budget(rates=-fixed_rates))


def attach_accepted_well_state(*, model, head, info, dt, recharge_field=None):
    """Bind accepted WEL rates and integrated source volumes to owned outputs.

    :param model: Owning 2D model; ordinary models without WEL incur no snapshots.
    :param head: Accepted groundwater head field.
    :param info: Public solve/period metadata, modified only after convergence.
    :param dt: Actual accepted duration, or None for steady flow.
    :param recharge_field: Optional explicit pure recharge supplied to mixed execution.
    :return: None.
    """
    if not info.get('converged', False) or not getattr(model, '_track_well_sources', False):
        return
    attach_accepted_source_state(model=model, head=head, info=info, dt=dt,
        entries=model.wel_entries, budgets=source_budgets_2d(model=model, recharge_field=recharge_field),
        source_inputs=own_source_inputs(volumetric=None,
            recharge=model._recharge_input_host if recharge_field is None else recharge_field,
            volumetric_factor=1., recharge_factor=1., cell_area=model.dx * model.dx, copy_arrays=True))


def attach_accepted_source_state(*, model, head, info, dt, entries, budgets, source_inputs=None):
    """Own dimensional WEL rates and source ledgers for one accepted equation.

    :param model: Unique owning groundwater model.
    :param head: Accepted public heads; an existing resident snapshot is reused.
    :param info: Accepted metadata, modified to expose independently owned outputs.
    :param dt: Actual accepted duration or None.
    :param entries: Actual period-specific volumetric well entries, or None.
    :param budgets: Actual period-specific gross physical source budgets.
    :param source_inputs: Optional independently owned physical source decomposition.
    :return: None; rejected states receive no accepted outputs.
    """
    from uuid import uuid4
    if not info.get('converged', False):
        return
    rates = np.empty(0) if entries is None else np.asarray(entries.rates, dtype=np.float64)
    cells = np.empty((0, np.ndim(head)), dtype=np.int64) if entries is None else np.asarray(entries.cells, dtype=np.int64)
    session = getattr(model, '_session', None)
    token = info.get('accepted_state', {})
    equation_identifier = token.get('identifier') if isinstance(token, dict) else None
    equation = getattr(session, '_accepted_equations', {}).get(equation_identifier)
    if equation is None:
        accepted_head = np.asarray(head, dtype=np.float64)
        head_bytes, head_owner, shape = accepted_head.tobytes(), None, accepted_head.shape
        equation_identifier = None
    else:
        head_bytes, head_owner, shape = None, equation.head, equation.shape
    owned_budgets = tuple((name, float(budget['inflow']), float(budget['outflow']), float(budget['net']))
        for name, budget in budgets.items())
    if source_inputs is None:
        raw = getattr(model, '_volumetric_source', None)
        recharge = getattr(model, '_recharge_rate', None)
        source_inputs = own_source_inputs(volumetric=raw, recharge=recharge,
            volumetric_factor=1., recharge_factor=1., cell_area=model.dx * getattr(model, 'dy', model.dx),
            copy_arrays=raw is not None and np.shares_memory(raw, model._rhs))
    state = AcceptedWellState(head_bytes=head_bytes, shape=tuple(shape), rate_bytes=rates.tobytes(),
        budgets=owned_budgets, dt=None if dt is None else float(dt), cell_bytes=cells.tobytes(),
        head_owner=head_owner, equation_identifier=equation_identifier, source_inputs=source_inputs)
    if not hasattr(model, '_accepted_well_states'):
        model._accepted_well_states = {}
    identifier = uuid4().hex
    model._accepted_well_states[identifier] = state
    info['accepted_source_state'] = identifier
    info['accepted_source_dt'] = state.dt
    info['source_budgets'] = {name: dict(inflow=inflow, outflow=outflow, net=net)
        for name, inflow, outflow, net in owned_budgets}
    info['wel_entry_rates'] = rates.copy()
    info['wel_entry_cells'] = cells.copy()
    if state.dt is not None:
        info['source_volume_budgets'] = {name: {key: value * state.dt for key, value in budget.items()}
            for name, budget in info['source_budgets'].items()}


def accepted_well_rates(*, model, head, info):
    """Verify historical WEL outputs without consulting live forcing.

    :param model: Unique model owner of the accepted state.
    :param head: Supplied accepted head field.
    :param info: Metadata from the same accepted solve or period.
    :return: Owned volumetric well entry rates.
    :raises RuntimeError: If the state is absent, rejected or owned by another model.
    :raises ValueError: If supplied heads, entry rates, budgets or duration changed.
    """
    if not isinstance(info, dict) or not info.get('converged', False):
        raise RuntimeError('Accepted WEL rates require a converged solve')
    state = getattr(model, '_accepted_well_states', {}).get(info.get('accepted_source_state'))
    if state is None:
        raise RuntimeError('Accepted WEL state does not belong to this model')
    accepted_head = (np.frombuffer(state.head_bytes, dtype=np.float64).reshape(state.shape)
        if state.head_owner is None else state.head_owner.numpy().reshape(state.shape))
    if state.equation_identifier is not None:
        token = info.get('accepted_state', {})
        if not isinstance(token, dict) or token.get('identifier') != state.equation_identifier or token.get('dt') != state.dt:
            raise ValueError('Source metadata is associated with a different accepted equation')
        if 'dt' in info and info['dt'] != state.dt:
            raise ValueError('Duration differs from the accepted physical equation')
    cells = np.frombuffer(state.cell_bytes, dtype=np.int64).reshape((-1, len(state.shape)))
    if not np.array_equal(np.asarray(info.get('wel_entry_cells')), cells):
        raise ValueError('WEL cells do not match the accepted source state')
    rates = np.frombuffer(state.rate_bytes, dtype=np.float64)
    if not np.array_equal(np.asarray(head), accepted_head):
        raise ValueError('head does not match the accepted source state')
    if not np.array_equal(np.asarray(info.get('wel_entry_rates')), rates):
        raise ValueError('WEL rates do not match the accepted source state')
    if info.get('accepted_source_dt') != state.dt:
        raise ValueError('Duration does not match the accepted source state')
    expected = {name: dict(inflow=inflow, outflow=outflow, net=net)
                for name, inflow, outflow, net in state.budgets}
    if info.get('source_budgets') != expected:
        raise ValueError('Source budgets do not match the accepted source state')
    if state.dt is not None:
        volumes = {name: {key: value * state.dt for key, value in budget.items()}
                   for name, budget in expected.items()}
        if info.get('source_volume_budgets') != volumes:
            raise ValueError('Source volumes do not match the accepted source state')
    return rates.copy()


def combine_well_depth_source(*, model, recharge):
    """Add owned WEL to an explicit FP64 recharge field for mixed execution.

    :param model: Model owning sparse volumetric WEL entries and cell width.
    :param recharge: Explicit pure recharge depth field, before WEL assembly.
    :return: Combined FP64 depth field; no combined allocation without WEL.
    :raises ValueError: If forcing cannot be represented finitely.
    """
    rate = np.asarray(recharge, dtype=np.float64)
    if not np.all(np.isfinite(rate)):
        raise ValueError('Mixed recharge must be finite')
    entries = model.wel_entries
    if entries is None:
        return rate
    area = model.dx * model.dx
    if not np.isfinite(area) or area <= 0.:
        raise ValueError('WEL requires a positive representable cell area')
    wells = well_cell_volume_rates(entries=entries, shape=(model.ny, model.nx), dtype=np.float64)
    with np.errstate(over='ignore', invalid='ignore'):
        combined = rate + wells / (model.dx * model.dx)
    if not np.all(np.isfinite(combined)):
        raise ValueError('Combined mixed recharge and WEL rates are not representable')
    return combined


def sparse_well_cell_rates(*, entries, shape, dtype=np.float64):
    """Accumulate sparse WEL with stable sums over each cell's original entries.

    :param entries: Validated owned WEL entries, or None.
    :param shape: Grid shape.
    :param dtype: Destination precision for signed numerical cell rates.
    :return: Unique flat cell indices, signed rates and FP64 gross rates.
    :raises ValueError: If signed or gross accumulation is not representable.
    """
    from math import fsum
    if entries is None or not len(entries):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=dtype), np.empty(0, dtype=np.float64)
    flat = np.ravel_multi_index(tuple(entries.cells.T), shape)
    order = np.argsort(flat, kind='stable')
    cells = flat[order]
    rates = np.asarray(entries.rates, dtype=np.float64)[order]
    starts = np.concatenate(([0], np.flatnonzero(np.diff(cells)) + 1))
    stops = np.concatenate((starts[1:], [rates.size]))
    signed = rates[starts].copy()
    gross = np.abs(signed)
    try:
        for group in np.flatnonzero(stops - starts > 1):
            values = rates[starts[group]:stops[group]]
            signed[group] = fsum(values)
            gross[group] = fsum(np.abs(values))
    except OverflowError as error:
        raise ValueError('Accumulated well rates are not representable') from error
    with np.errstate(over='ignore', invalid='ignore'):
        signed = signed.astype(dtype)
    if not np.all(np.isfinite(signed)) or not np.all(np.isfinite(gross)):
        raise ValueError('Accumulated well rates are not representable')
    return cells[starts], signed, gross


def own_source_inputs(*, volumetric, recharge, volumetric_factor, recharge_factor, cell_area, copy_arrays):
    """Retain the physical source decomposition independently of live model fields.

    :param volumetric: Raw volumetric cell inputs, or None.
    :param recharge: Actual period-specific depth rates, or None.
    :param volumetric_factor: Multiplier applied to the owned raw inputs.
    :param recharge_factor: Multiplier applied to the owned recharge inputs.
    :param cell_area: Physical receiving-cell area.
    :param copy_arrays: Copy mutable inputs; private immutable history inputs may share.
    :return: Immutable tuple of owned source inputs and their physical multipliers.
    """
    fields = []
    for name, value in (('volumetric', volumetric), ('recharge', recharge)):
        owned = None if value is None else (np.array(value, copy=True) if copy_arrays else np.asarray(value))
        if owned is not None:
            owned.setflags(write=False)
        fields.append((name, owned))
    return (*fields, ('volumetric_factor', float(volumetric_factor)),
        ('recharge_factor', float(recharge_factor)), ('cell_area', float(cell_area)))

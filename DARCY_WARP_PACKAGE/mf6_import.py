# SPDX-License-Identifier: AGPL-3.0-or-later
"""Import a supported MF6 GWF simulation and run it using DarcyWarp NPF/STO.

FloPy is the input reader, not the solver. All quantities remain in the
original model's consistent length and time units. This initial adapter is
deliberately strict about unsupported physical options.
"""

from dataclasses import dataclass
from pathlib import Path
import json
import math
import struct
import warnings
from time import perf_counter

import numpy as np


class UnsupportedMF6Model(ValueError):
    """An MF6 input requires physics outside the supported import contract."""


def _data(package, name, default=None):
    """Read optional FloPy scalar/record data without inventing a default.

    :param package: FloPy package.
    :param name: MFData attribute name.
    :param default: Value for an absent attribute or absent data.
    """
    field = getattr(package, name, None)
    if field is None:
        return default
    value = field.get_data()
    return default if value is None else value


def _present(value):
    """Determine whether an optional keyword or record was provided.

    :param value: FloPy scalar or record value.
    """
    if value is None:
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return np.size(value) > 0


def _reject_options(package, *names):
    """Reject populated unsupported package options.

    :param package: FloPy package.
    :param names: MFData attributes corresponding to unsupported options.
    """
    for name in names:
        if _present(_data(package, name)):
            raise UnsupportedMF6Model(
                f"{package.package_type.upper()}: unsupported option {name}"
            )


def _array(package, name, shape, default=None, dtype=float):
    """Expand a FloPy array, including external files and array multipliers.

    :param package: FloPy package.
    :param name: Array attribute.
    :param shape: Required shape.
    :param default: Default when the array was not supplied.
    :param dtype: Output NumPy dtype.
    """
    field = getattr(package, name, None)
    value = None if field is None else field.get_data(apply_mult=True)
    if value is None:
        if default is None:
            raise UnsupportedMF6Model(f"{package.package_type}: missing {name}")
        value = default
    return np.broadcast_to(np.asarray(value, dtype=dtype), shape).copy()


def _spacing(package, name, count):
    """Read a uniform positive structured-grid spacing.

    :param package: FloPy DIS package.
    :param name: DELR or DELC.
    :param count: Number of cells on the axis.
    """
    values = _array(package, name, (count,))
    if not np.isfinite(values).all() or values[0] <= 0 or not np.all(values == values[0]):
        raise UnsupportedMF6Model(f"DIS: {name.upper()} must be uniform and positive")
    return float(values[0])


def _prepare_interfaces(interfaces, active):
    """Provide valid dummy geometry for excluded DIS cells only.

    The native layered solver currently requires every cell, including
    IDOMAIN=0 placeholders, to have positive thickness. Preserve both
    interfaces of every active cell exactly, and modify only unowned
    interfaces in columns with invalid excluded-cell geometry. Never
    alter hydraulic connectivity or manufacture thickness in active cells.

    :param interfaces: Original MF6 interfaces (nlay+1, nrow, ncol).
    :param active: Boolean active-cell mask (nlay, nrow, ncol).
    :return: Validated solver geometry; original MF6 arrays are untouched.
    """
    z = np.asarray(interfaces, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        thickness = z[:-1] - z[1:]
    bad = ~(np.isfinite(thickness) & (thickness > 0))
    invalid_active = bad & active
    if np.any(invalid_active):
        k, j, i = np.argwhere(invalid_active)[0]
        raise UnsupportedMF6Model(
            f"DIS: {np.count_nonzero(invalid_active)} active cells have invalid "
            f"thickness; first at zero-based ({k}, {j}, {i}), "
            f"top={z[k, j, i]}, bottom={z[k + 1, j, i]}"
        )
    affected = np.any(bad, axis=0)
    if not np.any(affected):
        return z
    z = z.copy()
    # Entirely inactive columns are common in regional models. Vectorise
    # them rather than walking millions of excluded cells in Python.
    empty = affected & ~np.any(active, axis=0)
    if np.any(empty):
        z[:, empty] = -np.arange(z.shape[0], dtype=float)[:, None]
    for j, i in np.argwhere(affected & ~empty):
        column = z[:, j, i]
        live = active[:, j, i]
        # Interface k is immutable when either adjacent cell is active.
        anchored = np.zeros(column.size, dtype=bool)
        anchored[:-1] |= live
        anchored[1:] |= live
        anchors = np.flatnonzero(anchored)
        first, last = int(anchors[0]), int(anchors[-1])
        # Dummy top and bottom extensions, chosen large enough to remain
        # representable even at unusually large finite vertical datums.
        upper_step = max(1.0, abs(column[first]) * 1e-12)
        lower_step = max(1.0, abs(column[last]) * 1e-12)
        column[:first] = column[first] + upper_step * np.arange(first, 0, -1)
        column[last + 1:] = column[last] - lower_step * np.arange(
            1, column.size - last
        )
        for left, right in zip(anchors[:-1], anchors[1:]):
            # An internal inactive gap must have sufficient space between
            # its immutable active interfaces. A zero-width gap cannot be
            # represented by positive-thickness dummy cells without moving
            # an active interface, so reject rather than change the model.
            if column[left] <= column[right]:
                raise UnsupportedMF6Model(
                    f"DIS: inactive internal gap at row {j}, column {i} "
                    f"between interfaces {left} and {right} cannot be "
                    "represented without moving active-cell elevations"
                )
            if right - left <= 1:
                continue
            column[left + 1:right] = np.linspace(
                column[left], column[right], int(right - left + 1)
            )[1:-1]
    with np.errstate(over="ignore", invalid="ignore"):
        thickness = z[:-1] - z[1:]
    if not np.all(np.isfinite(thickness) & (thickness > 0)):
        raise UnsupportedMF6Model("DIS: excluded-cell dummy geometry is not representable")
    # This invariant is checked explicitly because active elevations define
    # NPF saturation, face conductances and STO volumes.
    if (not np.array_equal(z[:-1][active], interfaces[:-1][active]) or
            not np.array_equal(z[1:][active], interfaces[1:][active])):
        raise RuntimeError("DIS: active-cell geometry changed during import")
    return z


def timestep_lengths(perlen, nstp, tsmult):
    """Reproduce the geometric MF6 TDIS timestep sequence.

    :param perlen: Stress period duration in model time units.
    :param nstp: Number of time steps.
    :param tsmult: Ratio of successive step lengths.
    :return: Positive step lengths summing to the period duration.
    """
    perlen, tsmult = float(perlen), float(tsmult)
    if (not np.isfinite(perlen) or perlen <= 0 or
            not np.isfinite(tsmult) or tsmult <= 0 or int(nstp) != nstp or nstp < 1):
        raise UnsupportedMF6Model("TDIS requires positive PERLEN, NSTP and TSMULT")
    nstp = int(nstp)
    exponent = np.arange(nstp, dtype=float) * math.log(tsmult)
    weights = np.exp(exponent - np.max(exponent))
    lengths = (weights / weights.sum()) * perlen
    lengths[-1] += perlen - math.fsum(lengths)
    if not np.all(np.isfinite(lengths) & (lengths > 0)):
        raise UnsupportedMF6Model("TDIS timestep sequence is not representable")
    return lengths


@dataclass
class ImportedModel:
    """Static input fields and FloPy stress-period packages for one GWF model."""

    name: str
    shape: tuple
    dx: float
    dy: float
    interfaces: np.ndarray
    active: np.ndarray
    initial: np.ndarray
    kx: np.ndarray
    ky: np.ndarray
    kz: np.ndarray
    cell_type: np.ndarray
    ss: np.ndarray
    sy: np.ndarray
    periods: tuple
    packages: tuple
    length_units: str
    time_units: str


def import_simulation(simulation):
    """Validate and translate a loaded single-model MF6 simulation.

    :param simulation: FloPy MFSimulation.
    :return: ImportedModel with MF6 layer/row/column array ordering unchanged.
    """
    if len(simulation.model_names) != 1:
        raise UnsupportedMF6Model("Exactly one GWF model is supported")
    gwf = simulation.get_model(simulation.model_names[0])
    if not gwf.model_type.lower().startswith("gwf"):
        raise UnsupportedMF6Model("Only groundwater flow (GWF) is supported")
    supported = {"nam", "dis", "ic", "npf", "sto", "chd", "ghb", "wel",
                 "rch", "rcha", "drn", "riv", "oc", "obs"}
    packages = tuple(gwf.packagelist)
    unsupported = sorted({p.package_type.lower() for p in packages} - supported)
    if unsupported:
        raise UnsupportedMF6Model(f"Unsupported GWF packages: {', '.join(unsupported)}")
    dis, ic, npf = gwf.get_package("dis"), gwf.get_package("ic"), gwf.get_package("npf")
    if dis is None or ic is None or npf is None:
        raise UnsupportedMF6Model("DIS, IC and NPF are required")
    tdis = simulation.get_package("tdis")
    _reject_options(tdis, "_ats_filerecord")
    _reject_options(npf, "alternative_cell_averaging", "thickstrt", "cvoptions",
                    "perched", "rewet_record", "xt3doptions", "highest_cell_saturation",
                    "dev_no_newton", "dev_omega", "_tvk_filerecord")
    shape = tuple(int(_data(dis, field)) for field in ("nlay", "nrow", "ncol"))
    nz, ny, nx = shape
    dx, dy = _spacing(dis, "delr", nx), _spacing(dis, "delc", ny)
    domain = _array(dis, "idomain", shape, default=1, dtype=int)
    if np.any(domain < 0):
        raise UnsupportedMF6Model("DIS: IDOMAIN=-1 vertical pass-through is unsupported")
    active = domain > 0
    if not active.any():
        raise UnsupportedMF6Model("DIS has no active cells")
    top = _array(dis, "top", (ny, nx))
    bottom = _array(dis, "botm", shape)
    interfaces = np.concatenate((top[None], bottom), axis=0)
    interfaces = _prepare_interfaces(interfaces, active)

    initial = _array(ic, "strt", shape)
    raw_types = _array(npf, "icelltype", shape, dtype=int)
    if np.any(raw_types[active] < 0):
        raise UnsupportedMF6Model("NPF: negative ICELLTYPE is unsupported")
    cell_type = np.where(active & (raw_types > 0), 1, 0)
    if cell_type.any() and not _present(_data(gwf.name_file, "newtonoptions")):
        raise UnsupportedMF6Model("Convertible NPF cells require the MF6 NEWTON option")
    for name in ("angle1", "angle2", "angle3"):
        if np.any(_array(npf, name, shape, default=0)[active] != 0):
            raise UnsupportedMF6Model(f"NPF: rotated anisotropy ({name}) is unsupported")
    kx = _array(npf, "k", shape)
    ky_ratio = _present(_data(npf, "k22overk"))
    kz_ratio = _present(_data(npf, "k33overk"))
    ky = _array(npf, "k22", shape, default=None if ky_ratio else kx)
    kz = _array(npf, "k33", shape, default=None if kz_ratio else kx)
    if ky_ratio:
        ky *= kx
    if kz_ratio:
        kz *= kx
    for field in (kx, ky, kz):
        if not np.all(np.isfinite(field[active]) & (field[active] > 0)):
            raise UnsupportedMF6Model("NPF requires positive finite active conductivity")
        field[~active] = 1.0
    if not np.all(np.isfinite(initial[active]) & (np.abs(initial[active]) < 1e20)):
        raise UnsupportedMF6Model("IC: active initial heads must be finite physical values")
    initial[~active] = 0.0

    sto = gwf.get_package("sto")
    ss = np.zeros(shape)
    sy = np.zeros(shape)
    if sto is not None:
        _reject_options(sto, "storagecoefficient", "ss_confined_only",
                        "dev_original_specific_storage", "dev_oldstorageformulation",
                        "_tvs_filerecord")
        iconvert = _array(sto, "iconvert", shape, dtype=int)
        if np.any(iconvert[active] < 0) or np.any((iconvert[active] > 0) != cell_type[active]):
            raise UnsupportedMF6Model("STO ICONVERT must match NPF ICELLTYPE")
        ss = _array(sto, "ss", shape)
        sy = _array(sto, "sy", shape, default=0)
        for field in (ss, sy):
            if not np.all(np.isfinite(field[active]) & (field[active] >= 0)):
                raise UnsupportedMF6Model("STO requires finite nonnegative Ss and Sy")
            field[~active] = 0.0

    periods = []
    transient = sto is not None  # MF6 STO defaults to transient.
    for kper, record in enumerate(tdis.perioddata.get_data()):
        if sto is not None:
            steady_flag = bool(sto.steady_state.get_data(key=kper))
            transient_flag = bool(sto.transient.get_data(key=kper))
            if steady_flag and transient_flag:
                raise UnsupportedMF6Model(f"STO: conflicting flags in period {kper + 1}")
            if steady_flag:
                transient = False
            if transient_flag:
                transient = True
        periods.append((transient, timestep_lengths(record[0], record[1], record[2])))
    if not periods:
        raise UnsupportedMF6Model("TDIS contains no stress periods")

    boundaries = tuple(p for p in packages if p.package_type.lower() in
                       {"chd", "ghb", "wel", "rch", "rcha", "drn", "riv"})
    for package in boundaries:
        _reject_options(package, "auxmultname", "auxdepthname", "mover",
                        "auto_flow_reduce", "auto_flow_reduce_auxname",
                        "flow_reduction_length", "dev_cubic_scaling",
                        "_ts_filerecord", "_tas_filerecord")
    return ImportedModel(gwf.name, shape, dx, dy, interfaces, active, initial,
                         kx, ky, kz, cell_type, ss, sy, tuple(periods), boundaries,
                         str(_data(dis, "length_units", "UNSPECIFIED")),
                         str(_data(tdis, "time_units", "UNSPECIFIED")))


def _list_schedule(package, nper):
    """Resolve omitted list periods and explicit empty blocks independently.

    :param package: FloPy list stress package.
    :param nper: Total number of stress periods.
    :return: Per-period lists, preserving FloPy's reuse/empty semantics.
    """
    data = package.stress_period_data
    explicit = data.get_data(apply_mult=True) or {}
    empty = getattr(data, "empty_keys", {})
    # FloPy's array=True accessor recognises empty PERIOD block headers in
    # loaded files, including a header without a stored recarray.
    expanded = data.get_data(array=True, apply_mult=True)
    current = ()
    result = []
    for period in range(nper):
        if period in explicit:
            current = explicit[period]
        elif empty.get(period, False):
            current = ()
        elif expanded is not None and expanded[period] is None:
            current = ()
        if current is None:
            current = ()
        result.append(current)
    return result


def _array_schedule(field, nper, default):
    """Resolve inherited FloPy transient arrays.

    :param field: FloPy transient array.
    :param nper: Number of stress periods.
    :param default: Array value before the first explicit input.
    """
    explicit = field.get_data(apply_mult=True) or {}
    current = default
    result = []
    for period in range(nper):
        if explicit.get(period) is not None:
            current = explicit[period]
        result.append(current)
    return result


def _cell(value, shape):
    """Validate a zero-based MF6 DIS cell identifier.

    :param value: FloPy cellid.
    :param shape: Structured grid shape.
    """
    cell = tuple(value)
    if len(cell) != 3 or any(not isinstance(i, (int, np.integer)) or i < 0 or i >= n
                             for i, n in zip(cell, shape)):
        raise UnsupportedMF6Model(f"Invalid DIS cellid {value!r}")
    return tuple(int(i) for i in cell)


def _number(value, label):
    """Reject time-series names, nonfinite values and nonnumeric input.

    :param value: FloPy stress-package value.
    :param label: Diagnostic label.
    """
    try:
        number = float(value)
    except (ValueError, TypeError) as error:
        raise UnsupportedMF6Model(f"{label}: expected a numeric value, got {value!r}") from error
    if not np.isfinite(number):
        raise UnsupportedMF6Model(f"{label}: nonfinite value")
    return number


def _add_recharge(rhs, model, cell, rate, fixed):
    """Convert list recharge on an existing MF6 node to volumetric forcing.

    :param rhs: Dense volumetric source, updated in place.
    :param model: Imported static geometry.
    :param cell: Requested recharge cell.
    :param rate: Signed recharge depth per model time.
    :param fixed: Current CHD mask.
    """
    if not model.active[cell]:
        raise UnsupportedMF6Model(f"RCH: list entry on an excluded IDOMAIN cell {cell}")
    if not fixed[cell]:
        rhs[cell] += rate * model.dx * model.dy


def _add_array_recharge(rhs, model, rate, layers, fixed):
    """Apply RCHA directly to the specified reduced-grid nodes.

    MF6 6.6.2 maps IRCH to reduced node numbers before formulating
    recharge. IDOMAIN-excluded nodes have no equation and receive no
    recharge, even without FIXED_CELL. Dynamic dry-cell reassignment is
    outside this adapter's native NEWTON-only convertible contract.

    :param rhs: Dense volumetric source, updated in place.
    :param model: Imported static geometry.
    :param rate: Horizontal recharge field in length per time.
    :param layers: Zero-based FloPy IRCH values (default zero).
    :param fixed: Current CHD mask.
    """
    horizontal = model.shape[1:]
    rate = np.broadcast_to(np.asarray(rate, dtype=float), horizontal)
    irch = np.broadcast_to(np.asarray(layers, dtype=int), horizontal)
    if np.any((irch < 0) | (irch >= model.shape[0])):
        raise UnsupportedMF6Model("RCHA: IRCH outside the structured grid")
    j, i = np.indices(horizontal)
    receiving = model.active[irch, j, i] & ~fixed[irch, j, i]
    if not np.isfinite(rate[receiving]).all():
        raise UnsupportedMF6Model("RCHA: nonfinite recharge on receiving cells")
    k, j, i = irch[receiving], j[receiving], i[receiving]
    rhs[k, j, i] += rate[receiving] * model.dx * model.dy


def _record_ignored_boundary(diagnostics, kind, reason, cell):
    """Count excluded stress entries and retain one example cell.

    :param diagnostics: Mutable dictionary for the current stress period.
    :param kind: MF6 package type.
    :param reason: Either prescribed-head suppression or inactive exclusion.
    :param cell: Zero-based DIS cell identifier.
    """
    package = diagnostics.setdefault(kind.upper(), {})
    entry = package.setdefault(reason, {"count": 0, "first_cell": list(cell)})
    entry["count"] += 1


def _stress_period(model, schedules, kper, *, ignore_inactive_boundaries=False,
                   diagnostics=None):
    """Translate all packages for one stress period.

    :param model: Imported static geometry.
    :param schedules: List of package and resolved data tuples.
    :param kper: Zero-based stress period.
    :param ignore_inactive_boundaries: Explicitly allow list stresses on
        IDOMAIN=0 cells to be omitted. Native MF6 rejects such list input.
    :param diagnostics: Optional dictionary populated with excluded-entry counts.
    :return: Keyword arguments for the native layered model builder.
    """
    ignored = {} if diagnostics is None else diagnostics
    fixed = np.zeros(model.shape, dtype=bool)
    values = np.zeros(model.shape)
    for package, rows, unused in schedules:
        if package.package_type != "chd":
            continue
        for record in rows[kper]:
            cell = _cell(record["cellid"], model.shape)
            if not model.active[cell] or fixed[cell]:
                raise UnsupportedMF6Model(f"CHD: inactive or duplicate cell {cell}")
            fixed[cell] = True
            values[cell] = _number(record["head"], "CHD head")

    rhs = np.zeros(model.shape)
    ghb_cond = np.zeros(model.shape)
    ghb_weighted = np.zeros(model.shape)
    wel, drn, riv = [], [], []
    for package, rows, layers in schedules:
        kind = package.package_type
        if kind == "chd":
            continue
        if kind == "rcha":
            _add_array_recharge(rhs, model, rows[kper], layers[kper], fixed)
            continue
        for record in rows[kper]:
            cell = _cell(record["cellid"], model.shape)
            if not model.active[cell]:
                if not ignore_inactive_boundaries:
                    raise UnsupportedMF6Model(
                        f"{kind.upper()}: list entry on IDOMAIN=0 cell {cell} "
                        f"(stress period {kper + 1}; zero-based indices). "
                        "MF6 rejects excluded list cells. To explicitly omit "
                        "these records, pass ignore_inactive_boundaries=True."
                    )
                _record_ignored_boundary(ignored, kind, "inactive", cell)
                continue
            # Native MF6 formulations set HCOF/RHS (and reported package
            # exchange) to zero on prescribed-head cells, irrespective of
            # package ordering. Preserve the CHD equation and suppress only
            # the other stress entry, not the cell itself.
            if fixed[cell]:
                _record_ignored_boundary(ignored, kind, "chd", cell)
                continue
            if kind == "rch":
                _add_recharge(rhs, model, cell, _number(record["recharge"], "RCH"), fixed)
                continue
            if kind == "wel":
                wel.append((cell, _number(record["q"], "WEL rate")))
                continue
            cond = _number(record["cond"], f"{kind.upper()} conductance")
            if cond < 0:
                raise UnsupportedMF6Model(f"{kind.upper()}: negative conductance")
            if kind == "ghb":
                ghb_cond[cell] += cond
                ghb_weighted[cell] += cond * _number(record["bhead"], "GHB head")
            elif kind == "drn":
                drn.append((cell, cond, _number(record["elev"], "DRN elevation")))
            elif kind == "riv":
                riv.append((cell, cond, _number(record["stage"], "RIV stage"),
                            _number(record["rbot"], "RIV bottom")))
    inactive_counts = {kind: reasons["inactive"]["count"]
                       for kind, reasons in ignored.items() if "inactive" in reasons}
    if inactive_counts:
        warnings.warn(
            f"MF6 import period {kper + 1}: explicitly omitting list entries "
            f"on excluded IDOMAIN=0 cells: {inactive_counts}. "
            "This is a repair of invalid native MF6 list input, not exact input parity.",
            UserWarning, stacklevel=2,
        )
    # Parallel GHB entries are algebraically equivalent to one conductance-
    # weighted boundary per cell. Individual GHB fluxes are not exported here.
    ghb_head = np.divide(ghb_weighted, ghb_cond, out=np.zeros(model.shape),
                         where=ghb_cond > 0)
    has_ghb = np.any(ghb_cond > 0)
    return dict(bc_mask=fixed, bc_values=values, rhs=rhs,
                ghb_conductance=ghb_cond if has_ghb else None,
                ghb_head=ghb_head if has_ghb else None,
                wel_entries=wel or None, drn_entries=drn or None,
                riv_entries=riv or None)


def _schedules(model):
    """Extract numeric stress period schedules from all supported packages.

    :param model: ImportedModel.
    """
    result = []
    nper = len(model.periods)
    for package in model.packages:
        if package.package_type == "rcha":
            recharge = _array_schedule(package.recharge, nper, 0.0)
            layers = _array_schedule(package.irch, nper, 0)
            result.append((package, recharge, layers))
        else:
            result.append((package, _list_schedule(package, nper), None))
    return result


def _write_heads(stream, head, active, kstp, kper, pertim, totim):
    """Write a double-precision MF6 binary HEAD record for each layer.

    :param stream: Open binary output stream.
    :param head: Accepted DarcyWarp head array in MF6 ordering.
    :param active: Active-cell mask.
    :param kstp: One-based time step number.
    :param kper: One-based stress period number.
    :param pertim: Time elapsed in the period.
    :param totim: Cumulative simulation time.
    """
    nz, ny, nx = head.shape
    for k in range(nz):
        stream.write(struct.pack("<iidd16siii", kstp, kper, pertim, totim,
                                 b"HEAD            ", nx, ny, k + 1))
        values = np.where(active[k], head[k], 1e30)
        stream.write(np.asarray(values, dtype="<f8", order="C").tobytes())


def _numeric_summary(values):
    """Retain scalar numerical diagnostics without copying full-grid histories.

    :param values: Solver diagnostic dictionary.
    """
    result = {}
    if isinstance(values, dict):
        for key, value in values.items():
            if isinstance(value, (bool, np.bool_)):
                result[key] = bool(value)
            elif isinstance(value, (int, float, np.integer, np.floating)):
                result[key] = float(value) if np.isfinite(value) else None
            elif isinstance(value, dict):
                result[key] = _numeric_summary(value)
    return result


def _import_solver_options(*, model, device, precision, solver_options):
    """Build the fast NPF profile without changing physical acceptance gates.

    :param model: Imported model, including its cell types and packages.
    :param device: Requested Warp device.
    :param precision: Mixed-fast or full FP64 execution profile.
    :param solver_options: Explicit overrides for the native NPF solve API.
    :return: Solver options, with explicit overrides taking precedence.
    """
    from DARCY_WARP_PACKAGE.solvers.npf662_kcycle import NPF662KCycleControls
    from DARCY_WARP_PACKAGE.solvers.npf662_newton import NPF662NewtonControls

    if precision not in ("mixed_fast", "float64"):
        raise ValueError("precision must be 'mixed_fast' or 'float64'")
    options = dict(solver_options or {})
    if options.get("npf_preconditioner", "kcycle") in (None, "kcycle"):
        if options.get("npf_kcycle_controls") is None:
            convertible = bool(model.cell_type.any())
            nonlinear_boundaries = any(
                package.package_type in ("drn", "riv") for package in model.packages
            )
            options["npf_kcycle_controls"] = NPF662KCycleControls(
                levels=3 if convertible else 5,
                coarse_sweeps=2,
                cached_line_factors=True,
                preconditioner_precision="float32" if precision == "mixed_fast" else "float64",
                coefficient_state="current" if convertible or nonlinear_boundaries else "saturated",
                device_hierarchy_build=True,
                use_cuda_graph=True,
            )
    if options.get("npf_newton_controls") is None:
        options["npf_newton_controls"] = NPF662NewtonControls(
            reuse_trial_state=True, tiled_reductions=True,
        )
    options.setdefault("npf_device_native", str(device).startswith("cuda"))
    return options


def _prepare_run_outputs(output, *, overwrite, reference_head_file=None):
    """Prepare only the three files owned by the import driver.

    :param output: Resolved output directory.
    :param overwrite: Replace existing run files when explicitly enabled.
    :param reference_head_file: Optional external MF6 reference, never replaced.
    :return: Final head path, partial head path and report path.
    """
    paths = tuple(output / name for name in
                  ("darcywarp.hds", "darcywarp.hds.partial", "darcywarp_report.json"))
    if reference_head_file is not None:
        reference = Path(reference_head_file).expanduser().resolve()
        if any(path.resolve() == reference for path in paths):
            raise ValueError("Reference head file must not be a DarcyWarp output being replaced")
    output.mkdir(parents=True, exist_ok=True)
    existing = [path for path in paths if path.exists() or path.is_symlink()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Run files already exist in {output}; pass overwrite=True to replace them"
        )
    for path in existing:
        if path.is_dir() and not path.is_symlink():
            raise IsADirectoryError(path)
    for path in existing:
        path.unlink()
    return paths


def run_mf6_model(model_directory, output_directory, *, device="cuda:0",
                  precision="mixed_fast", solver_options=None, reference_head_file=None,
                  overwrite=False, ignore_inactive_boundaries=False):
    """Run an existing MF6 workspace without invoking or modifying MF6.

    :param model_directory: Directory containing mfsim.nam.
    :param output_directory: Separate directory for DarcyWarp outputs.
    :param device: Warp CPU or CUDA device.
    :param precision: 'mixed_fast' (FP32 K-cycle, FP64 physical state) or
        'float64' (FP64 K-cycle and physical state). Explicit solver options
        override the corresponding profile controls.
    :param solver_options: Optional native NPF solver controls, not MF6 IMS controls.
    :param reference_head_file: Optional existing MF6 binary head file for
        comparison at common (kstp, kper) records.
    :param overwrite: Replace previous DarcyWarp head, partial head and report
        files, leaving every unrelated file in the directory untouched.
    :param ignore_inactive_boundaries: Explicitly omit non-CHD list stress
        entries on excluded IDOMAIN cells; default is strict MF6 validation.
        Counts are reported by period and package.
    :return: Report containing run metadata and accepted-step diagnostics.
    """
    import flopy
    from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D

    started = perf_counter()
    source = Path(model_directory).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()
    if source == output:
        raise ValueError("Use a separate output directory")
    if not (source / "mfsim.nam").is_file():
        raise FileNotFoundError(source / "mfsim.nam")
    options = dict(solver_options or {})
    reserved = {"unconfined", "npf_cell_type", "transient", "dt", "head_prev",
                "ss", "sy", "return_info", "top_elevation"}
    if reserved.intersection(options):
        raise ValueError("Physical solver controls belong to the imported MF6 model")
    simulation = flopy.mf6.MFSimulation.load(sim_ws=str(source), strict=True,
                                              verbosity_level=0)
    model = import_simulation(simulation)
    options = _import_solver_options(model=model, device=device,
                                     precision=precision, solver_options=options)
    schedules = _schedules(model)
    head_path, partial_path, report_path = _prepare_run_outputs(
        output, overwrite=overwrite, reference_head_file=reference_head_file
    )

    nz, ny, nx = model.shape
    previous = model.initial.copy()
    current_mask = None
    previous_ghb = False
    elapsed = 0.0
    steps = []
    excluded_boundaries = []
    with partial_path.open("xb") as stream:
        with WarpDarcySolver3D(nx=nx, ny=ny, nz=nz, dx=model.dx, dy=model.dy,
                               device=device, formulation="mf6_6_6_2_npf",
                               implementation="fast", solver="kcycle") as solver:
            for kper, (transient, lengths) in enumerate(model.periods):
                diagnostics = {}
                forcing = _stress_period(
                    model, schedules, kper,
                    ignore_inactive_boundaries=ignore_inactive_boundaries,
                    diagnostics=diagnostics,
                )
                if diagnostics:
                    excluded_boundaries.append(dict(kper=kper + 1, packages=diagnostics))
                has_ghb = forcing["ghb_conductance"] is not None
                if current_mask is None or not np.array_equal(current_mask, forcing["bc_mask"]):
                    solver.build_from_layers(z_interfaces=model.interfaces,
                        kx_field=model.kx, ky_field=model.ky, kz_field=model.kz,
                        active=model.active, initial_head=previous, **forcing)
                    current_mask = forcing["bc_mask"].copy()
                else:
                    solver.update_boundary_values(forcing["bc_values"])
                    solver.update_rhs(forcing["rhs"])
                    solver.update_wel(forcing["wel_entries"])
                    if has_ghb:
                        solver.update_ghb(ghb_conductance=forcing["ghb_conductance"],
                                          ghb_head=forcing["ghb_head"])
                    elif previous_ghb:
                        # The native GHB update API disables existing entries
                        # with zero conductance, not None.
                        solver.update_ghb(ghb_conductance=0.0, ghb_head=0.0)
                    solver.update_drn_entries(forcing["drn_entries"])
                    solver.update_riv_entries(forcing["riv_entries"])
                previous_ghb = has_ghb
                pertim = 0.0
                for kstp, dt in enumerate(lengths):
                    solver.update_initial_head(previous)
                    controls = dict(options, unconfined=bool(model.cell_type.any()),
                                    npf_cell_type=model.cell_type, return_info=True)
                    if transient:
                        controls.update(transient=True, dt=float(dt),
                                        head_prev=previous, ss=model.ss, sy=model.sy)
                    else:
                        controls.update(transient=False)
                        controls.pop("npf_device_native", None)
                    head, info = solver.solve(**controls)
                    if not info["converged"]:
                        raise RuntimeError(f"DarcyWarp failed at period {kper + 1}, "
                                           f"step {kstp + 1}: {info.get('stop_reason')}")
                    head = np.asarray(head, dtype=float)
                    if not np.all(np.isfinite(head[model.active]) &
                                  (np.abs(head[model.active]) < 1e20)):
                        raise RuntimeError("Solver returned invalid active heads")
                    pertim += float(dt)
                    elapsed += float(dt)
                    _write_heads(stream, head, model.active, kstp + 1, kper + 1, pertim, elapsed)
                    steps.append(dict(kstp=kstp + 1, kper=kper + 1, dt=float(dt),
                                      totim=elapsed, transient=transient,
                                      converged=True, seconds=info.get("public_total_seconds"),
                                      head_precision=info.get("precision"),
                                      preconditioner_precision=info.get("multigrid", {}).get("precision"),
                                      budget=_numeric_summary(info.get("budget")),
                                      sources=_numeric_summary(info.get("source_budgets")),
                                      drn=_numeric_summary(info.get("drn_budget")),
                                      riv=_numeric_summary(info.get("riv_budget"))))
                    previous = head.copy()
    partial_path.rename(head_path)
    report = dict(model=model.name, shape=list(model.shape), device=device,
                  precision_profile=precision,
                  ignore_inactive_boundaries=bool(ignore_inactive_boundaries),
                  excluded_boundary_entries=excluded_boundaries,
                  length_units=model.length_units, time_units=model.time_units,
                  head_file=str(head_path), accepted_steps=len(steps), steps=steps,
                  run_seconds=perf_counter() - started,
                  output_scope="all accepted heads; scalar budget diagnostics, not MF6 CBC")
    if reference_head_file is not None:
        reference = flopy.utils.HeadFile(str(reference_head_file))
        actual = flopy.utils.HeadFile(str(head_path), precision="double")
        try:
            reference_times = dict(zip(reference.get_kstpkper(), reference.get_times()))
            actual_times = dict(zip(actual.get_kstpkper(), actual.get_times()))
            common = set(reference_times).intersection(actual_times)
            comparisons = []
            for kstp, kper in sorted(common, key=_step_order):
                step = (kstp, kper)
                if not np.isclose(reference_times[step], actual_times[step],
                                  rtol=1e-7, atol=1e-12):
                    raise ValueError(f"Reference elapsed time differs at {step}")
                expected = reference.get_data(kstpkper=(kstp, kper))
                calculated = actual.get_data(kstpkper=(kstp, kper))
                if expected.shape != model.shape:
                    raise ValueError("Reference head grid does not match the imported model")
                valid = model.active & np.isfinite(expected) & (np.abs(expected) < 1e20)
                if not np.array_equal(valid, model.active):
                    raise ValueError("Reference has dry or invalid active head markers")
                delta = calculated[valid] - expected[valid]
                comparisons.append(dict(kstp=int(kstp) + 1, kper=int(kper) + 1,
                                        rmse=float(np.sqrt(np.mean(delta ** 2))),
                                        max_abs=float(np.max(np.abs(delta)))))
            if not comparisons:
                raise ValueError("No common time-step records in reference head file")
            report["head_comparison"] = comparisons
        finally:
            reference.close()
            actual.close()
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def _step_order(step):
    """Sort a zero-based (kstp, kper) pair chronologically.

    :param step: FloPy timestep/period pair.
    """
    return step[1], step[0]

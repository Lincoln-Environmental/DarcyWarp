"""Compare production 3D operators with native MF6 budgets at identical heads.

This is an evaluation audit, not a solve or a timing benchmark. MF6 references
are read-only. Solution agreement and operand-scaled arithmetic agreement are
different measurements and are deliberately reported separately.
"""
from pathlib import Path
from dataclasses import replace
import hashlib
import json

import numpy as np
import warp as wp

from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D
from DARCY_WARP_PACKAGE.solvers.npf662_kernels import ghb_flow_at
from DARCY_WARP_PACKAGE.solvers.npf662_newton import NPF662NewtonControls
from DARCY_WARP_PACKAGE.validation.npf662_reference import load_npf662_case, quadratic_saturation_derivative


@wp.kernel
def sample_ghb(head: wp.array(dtype=wp.float64), conductance: wp.array(dtype=wp.float64),
               boundary: wp.array(dtype=wp.float64), inflow: wp.array(dtype=wp.float64)):
    i = wp.tid()
    inflow[i] = -ghb_flow_at(i, head, conductance, boundary)


def file_digest(*, path: Path):
    with path.open(mode="rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()  # Positional-only stdlib API.


def load_reference(*, directory: Path):
    """Verify both stored arrays and the complete original MF6 input identity."""
    manifest = json.loads(s=(directory / "manifest.json").read_text())
    fingerprint = hashlib.sha256(json.dumps(obj=manifest["inputs"], sort_keys=True).encode()).hexdigest()
    if fingerprint != manifest["fingerprint"] or directory.name != fingerprint:
        raise ValueError("Reference input fingerprint mismatch")
    for name in ("inputs.npz", "reference.npz"):
        if file_digest(path=directory / name) != manifest["sha256"][name]:
            raise ValueError(f"Corrupt reference: {name}")
    case = load_npf662_case(path=directory / "inputs.npz")
    if case.digest() != manifest["inputs"]["case_sha256"]:
        raise ValueError("Reference case digest mismatch")
    with np.load(file=directory / "reference.npz", allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    inputs = manifest["inputs"]
    transient = inputs["mode"] == "transient"
    count = len(inputs["factors"]) if transient else 1
    if arrays["heads"].shape != (count, *case.active.shape):
        raise ValueError("Reference head history shape mismatch")
    if transient and not np.array_equal(arrays["times"], inputs["dt"] * np.arange(1, count + 1)):
        raise ValueError("Reference time grid mismatch")
    if not manifest["accepted"]:
        raise ValueError("Reference did not pass its original sanity checks")
    return case, arrays, manifest


def edge_indices(*, case):
    """Native adapter order: x faces, y faces, then z faces; positive outward."""
    ids = np.arange(case.active.size).reshape(case.active.shape)
    left, right, axes = [], [], []
    for axis in (2, 1, 0):
        lower, upper = [slice(None)] * 3, [slice(None)] * 3
        lower[axis], upper[axis] = slice(None, -1), slice(1, None)
        lower, upper = tuple(lower), tuple(upper)
        connected = (case.active[lower] != 0) & (case.active[upper] != 0)
        left.append(ids[lower][connected])
        right.append(ids[upper][connected])
        axes.append(np.full(shape=int(connected.sum()), fill_value=2 - axis, dtype=int))
    return np.concatenate(left), np.concatenate(right), np.concatenate(axes)


def face_operands(*, case, head, edges, c0):
    """Operand magnitudes include the convertible saturation's layer datum."""
    left, right, axes = edges
    h = head.ravel()
    donor = np.where((h[left] > h[right]) | ((h[left] == h[right]) & (axes == 1)), left, right)
    variable = (axes != 2) & ((case.cell_types.ravel()[left] != 0) | (case.cell_types.ravel()[right] != 0))
    datum = np.abs(case.bottoms.ravel()[donor]) / case.thicknesses.ravel()[donor] + 1.
    scale = c0 * (np.abs(h[left]) + np.abs(h[right]) +
        np.where(variable, np.abs(h[left] - h[right]) * datum, 0.))
    return scale, donor, variable


def arithmetic_comparison(*, actual, native, scale, roundoff_factor):
    """An explicit FP64 backward-error allowance, with no absolute error floor."""
    actual, native, scale = np.broadcast_arrays(actual, native, scale)
    if not np.isfinite(roundoff_factor) or roundoff_factor <= 0:
        raise ValueError("roundoff_factor must be finite and positive")
    if not all(np.isfinite(value).all() for value in (actual, native, scale)) or np.any(scale < 0):
        raise ValueError("Nonfinite values or negative arithmetic scale")
    difference = np.abs(actual - native)
    allowance = roundoff_factor * np.finfo(np.float64).eps * scale
    ratios = np.divide(difference, allowance, out=np.zeros_like(difference), where=allowance > 0)
    ratios[(allowance == 0) & (difference != 0)] = np.inf
    return dict(passed=bool(np.all(difference <= allowance)), entries=int(difference.size),
                failures=int(np.count_nonzero(difference > allowance)),
                max_absolute=float(np.max(difference, initial=0.)),
                rms=float(np.sqrt(np.mean(difference ** 2))) if difference.size else 0.,
                max_allowance_ratio=float(np.max(ratios, initial=0.)))


def build_model(*, case, family, device):
    nz, ny, nx = case.active.shape
    if family not in ("cartesian", "npf"):
        raise ValueError("Unknown solver family")
    if family == "cartesian" and (np.any(case.cell_types) or not np.all(case.thicknesses == case.thicknesses.flat[0])):
        raise ValueError("Cartesian MF6 parity requires confined uniform-thickness cells")
    model = WarpDarcySolver3D(nx=nx, ny=ny, nz=nz, dx=case.dx, dy=case.dy,
        dz=float(case.thicknesses.flat[0]), device=device, implementation="fast",
        formulation="legacy" if family == "cartesian" else "mf6_6_6_2_npf")
    fields = dict(kx_field=case.kx, ky_field=case.ky, kz_field=case.kz,
        active=case.active, bc_mask=case.fixed, bc_values=case.values,
        rhs=case.source, initial_head=case.initial,
        ghb_conductance=case.ghb_conductance, ghb_head=case.ghb_head)
    if family == "cartesian":
        model.build_from_K_fields(**fields)
    elif hasattr(case, "z_interfaces"):
        model.build_from_layers(z_interfaces=case.z_interfaces, **fields)
    else:
        model.build_from_K_fields(**fields)
    return model


def evaluate_device(*, model, case, family, head, previous, source_factor, dt, ss, sy, edges):
    """Call the public solver with zero work budgets, then inspect its operator."""
    model.update_rhs(rhs=case.source * source_factor)
    model.update_initial_head(initial_head=head)
    options = dict(max_cycles=0)
    if family == "npf":
        options.update(max_outer=0, krylov_restart=2, npf_preconditioner="diagonal",
            npf_cell_type=case.cell_types, unconfined=bool(np.any(case.cell_types)),
            npf_newton_controls=NPF662NewtonControls(tiled_reductions=True))
        if not hasattr(case, "z_interfaces"):
            options["top_elevation"] = case.top
    else:
        options.update(max_levels=1, dh_rms_tol=None, dh_max_tol=None)
    if previous is not None:
        options.update(transient=True, dt=dt, head_prev=previous)
        options.update(dict(ss=ss, sy=sy, npf_device_native=True) if family == "npf" else dict(storage_coeff=ss))
    returned, unused = model.solve(**options)
    active = case.active != 0
    if not np.array_equal(returned[active], head[active]):
        raise ValueError("Evaluation-only call changed the MF6 heads")
    left, right, axes = edges
    if family == "npf":
        workspace = model._session._npf662_workspace_holder["workspace"]
        sampled = wp.zeros(shape=case.active.size, dtype=wp.float64, device=model.device)
        if workspace.ghb_conductance is not None:
            wp.launch(kernel=sample_ghb, dim=case.active.size, inputs=[workspace.head,
                workspace.ghb_conductance, workspace.ghb_head, sampled], device=model.device)
        result = dict(flux=workspace.q.numpy()[left, axes],
            c0=workspace.c0.numpy()[left, axes], ghb=sampled.numpy().reshape(head.shape),
            storage=workspace.storage_rate.numpy().reshape(head.shape),
            residual=workspace.residual.numpy().reshape(head.shape))
    else:
        fine = model._session._fast_workspace_holder["workspace"]["levels"][0]
        c0 = np.stack([fine["faces"][axis].numpy().ravel() for axis in (0, 2, 4)], axis=1)[left, axes]
        conductance = np.zeros_like(head) if case.ghb_conductance is None else np.broadcast_to(case.ghb_conductance, head.shape)
        boundary = np.zeros_like(head) if case.ghb_head is None else np.broadcast_to(case.ghb_head, head.shape)
        storage = np.zeros_like(head) if previous is None else ss * case.thicknesses * case.dx * case.dy / dt * (head - previous)
        result = dict(flux=c0 * (head.ravel()[left] - head.ravel()[right]), c0=c0,
            ghb=conductance * (boundary - head), storage=np.where((case.active != 0) & (case.fixed == 0), storage, 0.),
            residual=fine["residual"].numpy())
    return result


def compare_endpoint(*, case, head, previous, source_factor, dt, ss, sy, edges,
                     actual, native_flux, native_ghb, native_storage, roundoff_factor,
                     native_saturation=None):
    """Compare CBC face/GHB/STO entries and every free-cell conservation row."""
    left, right, axes = edges
    free = (case.active != 0) & (case.fixed == 0)
    if native_flux.shape != left.shape or any(value.shape != head.shape for value in (native_ghb, native_storage)):
        raise ValueError("Native budget shape mismatch")
    h = head.ravel()
    face_scale, donor, variable = face_operands(case=case, head=head, edges=edges, c0=actual["c0"])
    conductance = np.zeros_like(head) if case.ghb_conductance is None else np.broadcast_to(case.ghb_conductance, head.shape)
    boundary = np.zeros_like(head) if case.ghb_head is None else np.broadcast_to(case.ghb_head, head.shape)
    ghb_scale = conductance * (np.abs(head) + np.abs(boundary))
    storage_scale = np.zeros_like(head)
    if previous is not None:
        magnitude = np.abs(head) + np.abs(previous)
        convertible = case.cell_types != 0
        datum_magnitude = magnitude + 2. * (np.abs(case.bottoms) + case.thicknesses)
        storage_scale = case.dx * case.dy / dt * (
            ss * case.thicknesses * np.where(convertible, datum_magnitude, magnitude) +
            np.where(convertible, sy * datum_magnitude, 0.))
        storage_scale = np.where(free, storage_scale, 0.)
    n = head.size
    divergence = np.bincount(left, weights=native_flux, minlength=n) - np.bincount(right, weights=native_flux, minlength=n)
    native_residual = case.source * source_factor + native_ghb - native_storage - divergence.reshape(head.shape)
    row_scale = (np.bincount(left, weights=face_scale, minlength=n) +
        np.bincount(right, weights=face_scale, minlength=n)).reshape(head.shape)
    row_scale += np.abs(case.source * source_factor) + ghb_scale + storage_scale
    comparisons = dict(
        face_flux=arithmetic_comparison(actual=actual["flux"], native=native_flux, scale=face_scale, roundoff_factor=roundoff_factor),
        ghb_inflow=arithmetic_comparison(actual=actual["ghb"], native=native_ghb, scale=ghb_scale, roundoff_factor=roundoff_factor),
        storage_rate=arithmetic_comparison(actual=actual["storage"], native=native_storage, scale=storage_scale, roundoff_factor=roundoff_factor),
        cell_residual=arithmetic_comparison(actual=actual["residual"][free], native=native_residual[free], scale=row_scale[free], roundoff_factor=roundoff_factor))
    result = dict(operator_parity=all(value["passed"] for value in comparisons.values()), comparisons=comparisons,
        native_residual_rms=float(np.sqrt(np.mean(native_residual[free] ** 2))),
        device_residual_at_native_head_rms=float(np.sqrt(np.mean(actual["residual"][free] ** 2))))
    if native_saturation is not None:
        if native_saturation.shape != head.shape or not np.isfinite(native_saturation[case.active != 0]).all():
            raise ValueError("Invalid native DATA-SAT field")
        current, unused = quadratic_saturation_derivative(head=head, bottom=case.bottoms,
            thickness=case.thicknesses, cell_type=case.cell_types)
        reported_flux = actual["c0"] * np.where(variable, native_saturation.ravel()[donor], 1.) * (h[left] - h[right])
        result["reported_saturation_diagnostic"] = dict(
            max_saturation_difference=float(np.max(np.abs(native_saturation[free] - current[free]), initial=0.)),
            reconstructed_face_flux=arithmetic_comparison(actual=reported_flux, native=native_flux,
                scale=face_scale, roundoff_factor=roundoff_factor),
            note="Host reconstruction with production C0 and MF6 DATA-SAT; does not replace the production final-head operator check.")
    return result


def prescribed_flow_probe(*, case, head, actual, edges, directory, executable,
                          native_options, roundoff_factor):
    """Independent all-CHD flow-law probe, separate from the original solve.

    MF6 does not apply GHB or STO to CHD cells. They are not checked by this
    probe; their parity is checked against the original free-cell CBC data.
    Only saved MF6 heads are prescribed, never DarcyWarp solution heads.
    """
    from .npf662_mf6 import run_reference_mf6
    probe_case = replace(case, ghb_conductance=None, ghb_head=None)
    returned, info, fields = run_reference_mf6(case=probe_case, directory=directory,
        executable=executable, prescribed_state=head, native_solver_options=native_options,
        replace_native_solver_options=True, head_tolerance_m=native_options["outer_dvclose"])
    if not info["native_success"] or not np.array_equal(returned[case.active != 0], head[case.active != 0]):
        raise ValueError("Native prescribed-state probe failed or advanced heads")
    scale, unused_donor, unused_variable = face_operands(case=case, head=head, edges=edges, c0=actual["c0"])
    return dict(face_flux=arithmetic_comparison(actual=actual["flux"], native=fields["flux_native"],
            scale=scale, roundoff_factor=roundoff_factor),
        settings=native_options, settings_changed=False, prescribed_heads="saved MF6 endpoint",
        native_saturation_max_error=info["saturation_max_error"],
        head_sha256=hashlib.sha256(np.ascontiguousarray(head).tobytes()).hexdigest(),
        native_result_sha256=file_digest(path=directory / "result.npz"),
        scope="Fixed-head NPF face constitutive law only; GHB and STO checked on original free cells")


def read_native_saturations(*, directory, case, count, transient):
    """Read extra diagnostic CBC data; record its hash separately from the cache."""
    import flopy
    path = directory / "native" / ("transient.cbc" if transient else "npf662.cbc")
    mapping = np.arange(case.active.size).reshape(case.active.shape)[:, ::-1].ravel()
    budget = flopy.utils.CellBudgetFile(filename=str(path), precision="double")
    arrays = []
    try:
        if len(budget.get_times()) != count:
            raise ValueError("Native CBC timestep count mismatch")
        for period in range(count):
            records = budget.get_data(kstpkper=(0, period), text="DATA-SAT")
            if len(records) != 1:
                raise ValueError("Expected one DATA-SAT record per endpoint")
            record = records[0]
            nodes = record["node"] - 1
            if np.any(nodes < 0) or np.any(nodes >= case.active.size) or np.unique(nodes).size != nodes.size:
                raise ValueError("Invalid DATA-SAT node mapping")
            values = np.full(shape=case.active.size, fill_value=np.nan)
            values[mapping[nodes]] = record["sat"]
            arrays.append(values.reshape(case.active.shape))
    finally:
        budget.close()
    return arrays, file_digest(path=path)


def compare_heads(*, actual, native, active, head_tolerance):
    if actual.shape != native.shape or not np.isfinite(actual[:, active]).all():
        raise ValueError("Invalid solution head history")
    delta = actual[:, active] - native[:, active]
    maximum = float(np.max(np.abs(delta)))
    return dict(max_absolute_m=maximum, rms_m=float(np.sqrt(np.mean(delta ** 2))),
        tolerance_m=head_tolerance, passed=bool(maximum <= head_tolerance))

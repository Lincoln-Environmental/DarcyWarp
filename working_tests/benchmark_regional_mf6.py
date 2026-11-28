"""Audit steady layered DIS workspaces and benchmark the CUDA importer.

Run from the repository root with ``python -m working_tests.benchmark_regional_mf6``.
Configure stages and paths in the execution guard. Native runs use fresh copies.
"""
import csv
import json
from pathlib import Path
import re
import shutil
import subprocess
from time import perf_counter

import flopy
import numpy as np
from flopy.mf6.utils import MfGrdFile


def describe(*, values):
    """Return finite-array statistics used by the audit.

    :param values: Array restricted to the cells or connections being compared.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    if not values.size:
        return dict(count=0)
    return dict(count=values.size, min=float(values.min()), max=float(values.max()),
                mean=float(values.mean()), mae=float(np.abs(values).mean()),
                rms=float(np.sqrt(np.mean(values**2))),
                percentiles=dict(zip(("p01", "p50", "p95", "p99"),
                                     np.percentile(values, [1, 50, 95, 99]).tolist())),
                abs_p95=float(np.percentile(np.abs(values), 95)),
                abs_p99=float(np.percentile(np.abs(values), 99)),
                max_abs=float(np.abs(values).max()))


def comparison_gate(*, actual, expected, atol, rtol=0., roundoff=0.):
    """Check every value against an explicit absolute and relative tolerance.

    :param actual: Evaluated physical quantity.
    :param expected: Independent reference quantity.
    :param atol: Absolute tolerance in the quantity's units.
    :param rtol: Relative tolerance at each reference value.
    :param roundoff: Optional entrywise arithmetic error bound. The existing
        absolute and relative tolerance still applies wherever it is larger.
    :return: Pass/fail and maximum error, retaining diagnostics on failure.
    """
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape:
        raise ValueError("Parity gate arrays have different shapes")
    difference = np.abs(actual - expected)
    allowed = np.maximum(atol + rtol * np.abs(expected), roundoff)
    passed = bool(np.isfinite(actual).all() and np.isfinite(expected).all()
                  and np.all(difference <= allowed))
    return dict(passed=passed, max_abs=float(difference.max(initial=0.)),
                atol=float(atol), rtol=float(rtol))


def finish_audit(*, output, label, report, maps, active, gates):
    """Persist the receipt before rejecting a failed physical comparison.

    :param output: Audit artifact directory.
    :param label: Audit stage name.
    :param report: Complete diagnostic record.
    :param maps: Optional spatial diagnostic arrays.
    :param active: Active grid mask.
    :param gates: Named parity checks.
    """
    failed = [name for name, check in gates.items() if not check["passed"]]
    report.update(gates=gates, gates_passed=not failed)
    write_audit(output=output, label=label, report=report, maps=maps, active=active)
    if failed:
        raise RuntimeError(f"{label} parity failed: {', '.join(failed)}; "
                           f"see {output / (label + '.json')}")


def native_run(*, archive, directory, executable, save_saturation,
               ims_dvclose=None):
    """Run native MF6 in a new workspace and retain the listing and timing.

    :param archive: Untouched MF6 input directory.
    :param directory: New native working directory.
    :param executable: Absolute path to the native executable.
    :param save_saturation: Insert the NPF output-only SAVE_SATURATION option.
    :param ims_dvclose: Optional tighter outer/inner head closure for a separate
        diagnostic reference run; None preserves the archived IMS settings.
    """
    shutil.copytree(archive, directory)
    if save_saturation:
        npf = next(directory.glob("*.npf"))
        text = npf.read_text()
        npf.write_text(text.replace("BEGIN OPTIONS", "BEGIN OPTIONS\n  SAVE_SATURATION", 1))
    if ims_dvclose is not None:
        for ims in directory.glob("*.ims"):
            text = ims.read_text()
            text = re.sub(r"(?im)^\s*outer_[dh]vclose\s+[^\n]*\n", "", text)
            text = re.sub(r"(?im)^\s*inner_[dh]vclose\s+[^\n]*\n", "", text)
            text = text.replace("BEGIN NONLINEAR", f"BEGIN NONLINEAR\n  OUTER_DVCLOSE {ims_dvclose}", 1)
            text = text.replace("BEGIN LINEAR", f"BEGIN LINEAR\n  INNER_DVCLOSE {ims_dvclose}", 1)
            ims.write_text(text)
    started = perf_counter()
    with (directory / "native.log").open(mode="w") as stream:
        process = subprocess.run(args=[str(executable)], cwd=directory, stdout=stream,
                                 stderr=subprocess.STDOUT, check=False)
    elapsed = perf_counter() - started
    log = (directory / "native.log").read_text()
    if process.returncode or "Normal termination of simulation." not in log:
        raise RuntimeError(f"Native MF6 failed. See {directory / 'native.log'}")
    result = dict(seconds=elapsed, version=subprocess.check_output(
        args=[str(executable), "-v"], text=True).strip(),
        ims_dvclose=ims_dvclose)
    (directory / "native_run.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def independent_inputs(*, gwf):
    """Read static arrays and stress entries from FloPy's package data.

    :param gwf: FloPy structured groundwater model for a steady one-step case
        with CHD, GHB, DRN and RCHA.
    """
    dis, npf = gwf.dis, gwf.npf
    active = dis.idomain.array > 0
    kx = npf.k.array.copy()
    ky = kx if npf.k22.array is None else npf.k22.array.copy()
    kz = kx if npf.k33.array is None else npf.k33.array.copy()
    if npf.k22overk.get_data():
        ky = ky * kx
    if npf.k33overk.get_data():
        kz = kz * kx
    bottom = dis.botm.array
    top = np.concatenate((dis.top.array[None], bottom[:-1]), axis=0)
    initial = gwf.ic.strt.array
    fixed = np.zeros_like(active)
    fixed_values = np.zeros_like(initial)
    entries = {}
    for kind in ("CHD", "DRN", "GHB"):
        package = gwf.get_package(kind)
        if package is None:
            continue
        package_list = package if isinstance(package, list) else [package]
        if not package_list:
            continue
        package_rows = [item.stress_period_data.get_data(key=0)
                        for item in package_list]
        rows = np.concatenate(package_rows) if len(package_rows) > 1 else package_rows[0]
        ids = np.ravel_multi_index(np.asarray(rows["cellid"].tolist()).T, active.shape)
        entries[kind] = (ids, rows)
        if kind == "CHD":
            fixed.ravel()[ids] = True
            fixed_values.ravel()[ids] = rows["head"]
    dx, dy = dis.delr.array, dis.delc.array
    rcha = gwf.get_package("RCHA")
    horizontal = active.shape[1:]
    recharge = np.asarray(rcha.recharge.get_data(key=0))
    recharge = np.broadcast_to(recharge, horizontal) if recharge.ndim == 0 else recharge.reshape(horizontal)
    layer_data = rcha.irch.get_data(key=0)
    if layer_data is None:
        layers = np.zeros(horizontal, dtype=int)
    else:
        layers = np.asarray(layer_data, dtype=int).reshape(horizontal)
    j, i = np.indices(horizontal)
    eligible = active[layers, j, i] & ~fixed[layers, j, i]
    source = np.zeros(active.shape)
    source[layers, j, i] = np.where(eligible, recharge * dx[None, :] * dy[:, None], 0.)
    return dict(active=active, kx=kx, ky=ky, kz=kz, tops=top, bottoms=bottom,
                thicknesses=top-bottom, initial=initial, values=fixed_values, fixed=fixed,
                cell_types=(npf.icelltype.array > 0), source=source), entries


def translation_metrics(*, raw, entries, case):
    """Compare independent FloPy data to the effective DarcyWarp operator.

    :param raw: Static independent field mapping.
    :param entries: Original list records and flat user-node indices.
    :param case: DarcyWarp case built from the production layered operator.
    """
    active = raw["active"]
    report = {}
    for name, expected in raw.items():
        mask = np.ones_like(active) if name in ("active", "fixed", "source") else active
        actual = np.asarray(getattr(case, name))
        report[name] = dict(native=describe(values=expected[mask]),
                            difference=describe(values=actual[mask]-expected[mask].astype(float)),
                            gate=comparison_gate(actual=actual[mask], expected=expected[mask],
                                atol=1e-12 if name == "source" else 0.,
                                rtol=1e-14 if name == "source" else 0.))
    ids, ghb = entries["GHB"]
    keep = ~raw["fixed"].ravel()[ids]
    count = active.size
    cond = np.bincount(ids[keep], weights=ghb["cond"][keep], minlength=count)
    moment = np.bincount(ids[keep], weights=(ghb["cond"]*ghb["bhead"])[keep], minlength=count)
    report["ghb_conductance"] = dict(native=describe(values=ghb["cond"]),
        difference=describe(values=case.ghb_conductance.ravel()-cond),
        gate=comparison_gate(actual=case.ghb_conductance.ravel(), expected=cond, atol=1e-12, rtol=1e-14))
    occupied = cond > 0
    expected_head = moment[occupied] / cond[occupied]
    report["ghb_head"] = dict(native=describe(values=ghb["bhead"]),
        difference=describe(values=case.ghb_head.ravel()[occupied]-expected_head),
        gate=comparison_gate(actual=case.ghb_head.ravel()[occupied], expected=expected_head, atol=1e-12, rtol=1e-14))
    ids, drn = entries["DRN"]
    keep = np.flatnonzero(~raw["fixed"].ravel()[ids])
    order = keep[np.argsort(ids[keep], kind="stable")]
    actual_cells = np.ravel_multi_index(np.asarray(case.drn_entries.cells).T, active.shape)
    np.testing.assert_array_equal(actual_cells, ids[order])
    for name, column, values in (("drn_conductance", "cond", case.drn_entries.conductance),
                                 ("drn_elevation", "elev", case.drn_entries.first_level)):
        report[name] = dict(native=describe(values=drn[column]),
                            difference=describe(values=values-drn[column][order]),
                            gate=comparison_gate(actual=values, expected=drn[column][order], atol=0.))
    return report


def read_native(*, directory, active):
    """Decode native heads, signed budget records and both CSR directions.

    :param directory: Completed native workspace with DIS GRB, HDS and CBC files.
    :param active: Independently read IDOMAIN mask in user-grid ordering.
    """
    grid = MfGrdFile(next(directory.glob("*.grb")))
    with flopy.utils.HeadFile(next(directory.glob("*.hds"))) as reader:
        head = reader.get_data()
    with flopy.utils.CellBudgetFile(next(directory.glob("*.cbc")), precision="double") as reader:
        budget = {}
        for name in reader.get_unique_record_names():
            text = name.decode().strip()
            records = reader.get_data(text=text)
            if (len(records) > 1 and records[0].dtype.names is not None
                    and "node2" in records[0].dtype.names):
                adjusted, offset = [], 0
                for record in records:
                    item = record.copy()
                    local_node = item["node2"].copy()
                    item["node2"] = local_node + offset
                    offset += int(local_node.max(initial=0))
                    adjusted.append(item)
                budget[text] = np.concatenate(adjusted)
            else:
                budget[text] = records[-1]
    mapping = (np.arange(active.size) if grid.ia.size == active.size+1
               else np.flatnonzero(active))
    if grid.ia.size != mapping.size+1:
        raise ValueError("DIS CSR node count matches neither user nor reduced nodes")
    rows = np.repeat(mapping, np.diff(grid.ia))
    columns = mapping[grid.ja]
    flows = budget.pop("FLOW-JA-FACE").ravel()
    selection = rows != columns
    keys = rows[selection]*active.size + columns[selection]
    order = np.argsort(keys)
    keys, values = keys[order], flows[selection][order]
    residual = np.zeros(active.size)
    residual[rows[~selection]] = flows[~selection]
    return head, keys, values, residual, budget


def package_metrics(*, records, entries, case, head, fields,
                    flow_atol, flow_rtol, budget_rtol, native_divergence=None):
    """Verify CBC user-node IDs, CHD suppression and per-entry exchanges.

    :param records: Native final package budget arrays, with inflow positive.
    :param entries: Independent FloPy list records.
    :param case: Effective DarcyWarp operator.
    :param head: Evaluation head field.
    :param fields: CPU reference physical fields.
    :param flow_atol: Absolute entry exchange tolerance.
    :param flow_rtol: Relative entry exchange tolerance.
    :param budget_rtol: Relative gross package-budget tolerance.
    :param native_divergence: Native face divergence for CHD consistency at
        native heads; None compares actual DarcyWarp CHD exchange.
    """
    cells, summaries = {}, {}
    for kind in ("CHD", "DRN", "GHB", "RCHA"):
        if kind not in records:
            continue
        data = records[kind]
        ids = data["node"].astype(np.int64) - 1
        q = data["q"]
        cells[kind] = np.bincount(ids, weights=q, minlength=head.size).reshape(head.shape)
        summary = dict(inflow=float(q[q > 0].sum()), outflow=float(-q[q < 0].sum()),
                       duplicate_entries=int(ids.size - np.unique(ids).size))
        if kind in entries:
            source_ids, rows = entries[kind]
            entry = data["node2"] - 1
            np.testing.assert_array_equal(ids, source_ids[entry])
        roundoff = 0.
        if kind == "GHB":
            expected = rows["cond"][entry] * (rows["bhead"][entry]-head.ravel()[ids])
        elif kind == "DRN":
            conductance = rows["cond"][entry]
            elevation = rows["elev"][entry]
            saved_head = head.ravel()[ids]
            drawdown = np.maximum(saved_head - elevation, 0.)
            expected = -conductance * drawdown
            if native_divergence is not None:
                # MF6 saves C*z - C*h; the independent law evaluates -C*(h-z).
                # Each rounded product C*h or C*z can contribute half a local
                # ULP even when the head difference is small. Include rounding
                # of the two subtractions and the factored product as well.
                rounded_head = conductance * saved_head
                rounded_elevation = conductance * elevation
                expanded = rounded_elevation - rounded_head
                bound = 0.5 * (
                    np.spacing(np.abs(rounded_head)) + np.spacing(np.abs(rounded_elevation))
                    + np.spacing(np.abs(expanded)) + np.spacing(np.abs(expected))
                    + conductance * np.spacing(drawdown)
                )
                flowing = (drawdown > 0.) & ~case.fixed.ravel()[ids]
                roundoff = np.where(flowing, np.nextafter(bound, np.inf), 0.)
        elif kind == "CHD":
            expected = (fields["chd_cell_rate"] if native_divergence is None else
                        native_divergence - case.source).ravel()[ids]
        else:
            expected = case.source.ravel()[ids]
        if kind != "CHD":
            overlap = case.fixed.ravel()[ids] != 0
            expected = np.where(overlap, 0., expected)
            summary.update(chd_overlap=int(overlap.sum()), flowing_overlap=int(np.count_nonzero(q[overlap])))
        summary["entry_difference"] = describe(values=expected-q)
        summary["entry_gate"] = comparison_gate(actual=expected, expected=q,
            atol=flow_atol, rtol=flow_rtol, roundoff=roundoff)
        predicted = np.array([expected[expected > 0].sum(), -expected[expected < 0].sum()])
        native = np.array([summary["inflow"], summary["outflow"]])
        summary["computed_inflow"], summary["computed_outflow"] = map(float, predicted)
        summary["gross_budget_gate"] = comparison_gate(
            actual=predicted, expected=native, atol=flow_atol + budget_rtol * float(native.sum()))
        summaries[kind] = summary
    return summaries, cells


def audit(*, archive, directory, output, device, use_darcywarp_heads=False,
          translation_only=False, head_gate=0.005, head_rmse_gate=1e-4,
          saturation_head_gate=0.1, flow_atol=1e-6, flow_rtol=1e-10,
          budget_rtol=1e-3, write_maps=True):
    """Compare translation, native states and both DarcyWarp evaluators.

    :param archive: Original MF6 workspace.
    :param directory: Native workspace containing final budgets and heads.
    :param output: Directory for JSON, raw mapped differences and figures.
    :param device: CUDA device to exercise the production physical kernels.
    :param use_darcywarp_heads: Evaluate a separately accepted importer result.
    :param translation_only: Audit operator inputs without requiring native outputs.
    :param head_gate: Maximum head difference against the chosen MF6 reference.
    :param head_rmse_gate: RMS head difference against the chosen reference.
    :param saturation_head_gate: Saturated thickness tolerance at native heads;
        allow explicitly for lagged native saturation in loosely converged runs.
    :param flow_atol: Absolute face, coefficient, residual and entry tolerance
        in the corresponding model units.
    :param flow_rtol: Relative face and package entry tolerance.
    :param budget_rtol: Gross inflow/outflow tolerance relative to package throughput.
    :param write_maps: Write full spatial diagnostics and per-layer figures.
    """
    from DARCY_WARP_PACKAGE.mf6_import import (
        _regional_anchored_cell_head_tolerance, _schedules, _stress_period,
        import_simulation, load_mf6_workspace,
    )
    from DARCY_WARP_PACKAGE.solvers.npf662_3d import case_from_model, independent_reference
    from DARCY_WARP_PACKAGE.solvers.npf662_newton import NPF662Workspace
    from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D

    started = perf_counter()
    simulation = load_mf6_workspace(directory=archive)
    model = import_simulation(simulation=simulation)
    nz, ny, nx = model.shape
    if len(model.periods) != 1 or model.periods[0][0] or len(model.periods[0][1]) != 1:
        raise ValueError("This regional audit expects one steady step")
    package_types = {p.package_type for p in model.packages}
    if not {"ghb", "drn", "rcha"}.issubset(package_types) or package_types - {
            "chd", "ghb", "drn", "rcha"}:
        raise ValueError("This audit requires GHB, DRN and RCHA, with optional CHD")
    raw, entries = independent_inputs(gwf=simulation.get_model())
    forcing = _stress_period(model=model, schedules=_schedules(model=model), kper=0)
    imported_seconds = perf_counter()-started
    with WarpDarcySolver3D(nx=nx, ny=ny, nz=nz, dx=model.dx, dy=model.dy, device=device,
                           formulation="mf6_6_6_2_npf", implementation="fast", solver="kcycle") as solver:
        solver._anchored_cell_head_tolerance_m = _regional_anchored_cell_head_tolerance(model=model)
        started = perf_counter()
        solver.build_from_layers(z_interfaces=model.interfaces, kx_field=model.kx,
                                 ky_field=model.ky, kz_field=model.kz, active=model.active,
                                 initial_head=model.initial, **forcing)
        case = case_from_model(model=solver, top=0., cell_type=model.cell_type)
        build_seconds = perf_counter()-started
        translation = translation_metrics(raw=raw, entries=entries, case=case)
        gates = {f"translation:{name}": record["gate"] for name, record in translation.items()}
        if translation_only:
            finish_audit(output=output, label="translation", report=dict(translation=translation),
                         maps={}, active=raw["active"], gates=gates)
            return
        native_head, keys, native_flows, native_residual, budget = read_native(
            directory=directory, active=raw["active"])
        head = native_head
        if use_darcywarp_heads:
            with flopy.utils.HeadFile(output / "darcywarp.hds") as reader:
                head = reader.get_data()
        reference, _ = independent_reference(model=solver, case=case)
        datum_native_head, native_datum_records = reference.apply_nullspace_datum(
            head=native_head)
        if use_darcywarp_heads:
            datum_head, darcywarp_datum_records = reference.apply_nullspace_datum(
                head=head)
        else:
            datum_head = datum_native_head
            darcywarp_datum_records = []
        # Check the accepted physical equations at the evaluated heads. The
        # native baseline can have a looser IMS accuracy than DarcyWarp.
        metrics, cpu = reference.evaluate(head=head, relative_tolerance=1e-7)
        initial = np.where(case.fixed, case.values, case.initial)
        initial_saturation = reference.face_terms(head=initial)[3]
        workspace = NPF662Workspace(model=solver, case=case, restart=2,
                                    krylov_budget=0, reference=reference)
        try:
            workspace.head.assign(np.where(case.active, head, 0.).ravel())
            workspace.relative_tolerance = 1e-7
            gpu_metrics = workspace.evaluate(trial=False, linearize=True)
            gpu_flows = workspace.q.numpy()[reference.left, 2-reference.axes]
            gpu_conductance = workspace.c.numpy()[reference.left, 2-reference.axes]
            gpu_residual = workspace.residual.numpy()
            gpu_packages = workspace.boundary_budget() | workspace.chd_budget()
            gpu_drn = workspace.drn_entry_rate.numpy()
        finally:
            workspace.close()
        left, right, free = reference.left, reference.right, reference.free
        pos = np.searchsorted(keys, left*head.size+right)
        rev = np.searchsorted(keys, right*head.size+left)
        np.testing.assert_array_equal(keys[pos], left*head.size+right)
        np.testing.assert_array_equal(keys[rev], right*head.size+left)
        if keys.size != 2*left.size:
            raise ValueError("Native and DarcyWarp face counts disagree")
        native_q = -native_flows[pos]
        native_residual = native_residual.reshape(head.shape)
        reference_divergence = (np.bincount(left, weights=native_q, minlength=head.size)
                                - np.bincount(right, weights=native_q, minlength=head.size))
        packages, package_cells = package_metrics(records=budget, entries=entries,
            case=case, head=head, fields=cpu, flow_atol=flow_atol, flow_rtol=flow_rtol,
            budget_rtol=budget_rtol, native_divergence=None if use_darcywarp_heads
            else reference_divergence.reshape(head.shape))
        reassembled_native = sum(package_cells.values()) - reference_divergence.reshape(head.shape)
        physical = {name: metrics[name] for name in ("physical_rms", "head_residual_rms",
                  "acceptance_head_residual_rms",
                  "relative_cell_imbalance", "signed_imbalance", "independent_accepted",
                  "ungrounded_current_components", "exact_dry_zero_rows")}
        active = raw["active"]
        thickness = raw["thicknesses"]
        face_delta = cpu["flux"] - native_q
        worst_face = int(np.argmax(np.abs(face_delta)))
        worst_cell = int(np.argmax(np.abs(np.where(free, cpu["residual"]-native_residual, 0.))))
        cpu_drn = reference.boundary_terms(head=head)[0]
        native_drn = np.zeros(head.size)
        drn_records = budget["DRN"]
        np.add.at(native_drn, drn_records["node"]-1, drn_records["q"])
        gpu_drn_grid = np.bincount(reference.drn_csr.flat_cells, weights=-gpu_drn,
                                   minlength=head.size)
        report = dict(shape=list(head.shape), active=int(raw["active"].sum()),
                      spacing_m=[case.dx, case.dy], translation_seconds=imported_seconds,
                      operator_seconds=build_seconds, translation=translation, packages=packages,
                      cpu=physical, gpu=gpu_metrics, gpu_package_totals=gpu_packages,
                      physical_acceptance_thresholds=metrics["final_acceptance_thresholds"],
                      native_residual=describe(values=native_residual[free]),
                      native_residual_reassembly=describe(values=(reassembled_native-native_residual)[free]),
                      head_difference=describe(values=(head-native_head)[raw["active"]]),
                      datum_aligned_head_difference=describe(
                          values=(datum_head-datum_native_head)[raw["active"]]),
                      native_datum_adjustments=native_datum_records,
                      darcywarp_datum_adjustments=darcywarp_datum_records,
                      raw_head_gates=dict(
                          maximum=comparison_gate(actual=head[raw["active"]],
                              expected=native_head[raw["active"]], atol=head_gate),
                          rms=comparison_gate(actual=(head-native_head)[raw["active"]],
                              expected=np.zeros(int(raw["active"].sum())),
                              atol=head_rmse_gate)),
                      face_antisymmetry=describe(values=native_flows[pos]+native_flows[rev]),
                      cpu_face_difference=describe(values=face_delta),
                      gpu_face_difference=describe(values=gpu_flows-native_q),
                      worst_face=dict(left=list(map(int, np.unravel_index(left[worst_face], head.shape))),
                                      right=list(map(int, np.unravel_index(right[worst_face], head.shape))),
                                      difference=float(face_delta[worst_face])),
                      worst_residual_cell=list(map(int, np.unravel_index(worst_cell, head.shape))),
                      cpu_gpu_drn_entry_difference=describe(values=cpu_drn-gpu_drn),
                      gpu_native_drn_cell_difference=describe(values=(gpu_drn_grid-native_drn)[active.ravel()]),
                      initial_head=describe(values=initial[active]),
                      final_head=describe(values=head[active]),
                      initial_saturated_thickness=describe(values=(initial_saturation*thickness)[active]),
                      final_saturated_thickness=describe(values=(cpu["saturation"]*thickness)[active]),
                      initial_dry=int(np.count_nonzero(active & (initial_saturation == 0.))),
                      final_dry=int(np.count_nonzero(active & (cpu["saturation"] == 0.))),
                      cpu_gpu_conductance=describe(values=cpu["conductance"]-gpu_conductance),
                      cpu_residual_difference=describe(values=(cpu["residual"]-native_residual)[free]),
                      gpu_residual_difference=describe(values=(gpu_residual.reshape(head.shape)-native_residual)[free]))
        maps = {"Head difference (m)": head-native_head,
                "Datum-aligned head difference (m)": datum_head-datum_native_head,
                "CPU residual difference (m³/day)": cpu["residual"]-native_residual,
                "GHB difference (m³/day)": -cpu["ghb_outflow"]-package_cells["GHB"],
                "DRN difference (m³/day)": -cpu["drn_outflow"]-package_cells["DRN"]}
        if "CHD" in package_cells:
            maps["CHD difference (m³/day)"] = (
                cpu["chd_cell_rate"]-package_cells["CHD"])
        # These compare two different implementations at exactly the same
        # heads, independent of the native solver's convergence tolerance.
        gates["CPU/GPU face flow"] = comparison_gate(actual=gpu_flows, expected=cpu["flux"],
                                                       atol=flow_atol, rtol=flow_rtol)
        gates["CPU/GPU conductance"] = comparison_gate(actual=gpu_conductance,
            expected=cpu["conductance"], atol=flow_atol, rtol=flow_rtol)
        gates["CPU/GPU residual"] = comparison_gate(actual=gpu_residual.reshape(head.shape)[free],
            expected=cpu["residual"][free], atol=flow_atol, rtol=flow_rtol)
        gates["CPU/GPU DRN entries"] = comparison_gate(actual=gpu_drn, expected=cpu_drn,
                                                       atol=flow_atol, rtol=flow_rtol)
        gates["native face reciprocity"] = comparison_gate(actual=native_flows[pos],
            expected=-native_flows[rev], atol=flow_atol, rtol=flow_rtol)
        gates["native residual reassembly"] = comparison_gate(actual=reassembled_native[free],
            expected=native_residual[free], atol=flow_atol, rtol=flow_rtol)
        for kind, summary in packages.items():
            if summary.get("flowing_overlap", 0):
                gates[f"{kind}: CHD suppression"] = dict(passed=False)
            if use_darcywarp_heads:
                gates[f"{kind}: gross budget"] = summary["gross_budget_gate"]
            else:
                gates[f"{kind}: native entry exchange"] = summary["entry_gate"]
        if "DATA-SAT" in budget:
            data = budget["DATA-SAT"]
            saved = np.zeros(head.size)
            saved[data["node"]-1] = data["sat"]
            dh = native_head.ravel()[left]-native_head.ravel()[right]
            donor = np.where((dh > 0) | ((dh == 0) & (reference.axes == 1)), left, right)
            variable = (reference.axes != 0) & (
                (case.cell_types.ravel()[left] != 0) | (case.cell_types.ravel()[right] != 0))
            # Default vertical NPF conductance is full-thickness even when
            # one cell is dry; only convertible horizontal faces use SAT.
            c_native = reference.c0 * np.where(variable, saved[donor], 1.)
            report["saved_saturation_face_check"] = describe(values=c_native*dh-native_q)
            report["cpu_native_conductance"] = describe(values=cpu["conductance"]-c_native)
            saturation_delta = cpu["saturation"]-saved.reshape(head.shape)
            report["saturation_difference"] = describe(values=saturation_delta[raw["active"]])
            maps["Saturated thickness difference (m)"] = saturation_delta*raw["thicknesses"]
            gates["native saved-saturation face law"] = comparison_gate(
                actual=c_native*dh, expected=native_q, atol=flow_atol, rtol=flow_rtol)
            if not use_darcywarp_heads:
                gates["native saturated thickness"] = comparison_gate(
                    actual=(cpu["saturation"]*thickness)[active],
                    expected=(saved.reshape(head.shape)*thickness)[active], atol=saturation_head_gate)
        elif not use_darcywarp_heads:
            # No lagged saturation data: the face law must match directly.
            gates["native face law"] = comparison_gate(actual=cpu["flux"], expected=native_q,
                                                         atol=flow_atol, rtol=flow_rtol)
        if use_darcywarp_heads:
            gates["datum-aligned head maximum"] = comparison_gate(
                                                     actual=datum_head[active],
                                                     expected=datum_native_head[active],
                                                     atol=head_gate)
            gates["datum-aligned head RMS"] = comparison_gate(
                                                actual=report["datum_aligned_head_difference"]["rms"],
                                                expected=0., atol=head_rmse_gate)
            gates["DarcyWarp independent physical acceptance"] = dict(passed=bool(metrics["independent_accepted"]))
        gates["GPU finite physical metrics"] = dict(passed=bool(gpu_metrics["finite"]))
        label = "solution" if use_darcywarp_heads else "native_heads"
        finish_audit(output=output, label=label, report=report,
                     maps=maps if write_maps else {}, active=raw["active"], gates=gates)


def write_audit(*, output, label, report, maps, active):
    """Write scalar metrics, spatial fields and signed difference maps.

    :param output: Artifact directory.
    :param label: Stem identifying the evaluated head state.
    :param report: Numeric audit results.
    :param maps: Grid-aligned named difference fields.
    :param active: Boolean grid mask.
    """
    from DARCY_WARP_PACKAGE.validation.head_checks import finite_diagnostics

    output.mkdir(parents=True, exist_ok=True)
    (output / f"{label}.json").write_text(
        json.dumps(finite_diagnostics(value=report), indent=2, allow_nan=False) + "\n")
    if not maps:
        return

    np.savez_compressed(output / f"{label}_maps.npz", **maps)
    import matplotlib.pyplot as plt

    for layer in range(active.shape[0]):
        for name, values in maps.items():
            grid = np.where(active[layer], values[layer], np.nan)
            finite = grid[np.isfinite(grid)]
            limit = float(np.max(np.abs(finite), initial=0.)) or 1.
            field_name = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
            suffix = "" if active.shape[0] == 1 else f"_layer_{layer+1:03d}"
            plt.imsave(output / f"{label}_{field_name}{suffix}.png", grid,
                       cmap="RdBu_r", vmin=-limit, vmax=limit)


def benchmark_table(*, cases, output):
    """Write comparable wall times and accuracy for a series of accepted cases.

    :param cases: Mapping from model label to (native directory, DarcyWarp directory).
    :param output: Directory for the CSV and scaling figure.
    """
    import matplotlib.pyplot as plt

    rows = []
    for label, (native_dir, run_dir) in cases.items():
        native = json.loads((native_dir / "native_run.json").read_text())
        run = json.loads((run_dir / "darcywarp_report.json").read_text())
        comparison = json.loads((run_dir / "solution.json").read_text())
        native_audit = json.loads((run_dir / "native_heads.json").read_text())
        if not comparison.get("gates_passed") or not native_audit.get("gates_passed"):
            raise RuntimeError(f"{label}: missing or failed equation/solution parity gates")
        if not run["steps"] or not run["steps"][-1].get("converged"):
            raise RuntimeError(f"{label}: no accepted DarcyWarp endpoint")
        step = run["steps"][-1]
        rows.append(dict(model=label, cells=int(np.prod(run["shape"])), active=run["active_cells"],
            native_version=native["version"], native_seconds=native["seconds"],
            import_seconds=run["import_seconds"], operator_seconds=step["operator_seconds"],
            reference_setup_seconds=step["reference_setup_seconds"],
            solver_setup_seconds=step["solver_setup_seconds"], initialization_seconds=0.,
            solve_seconds=step["solve_seconds"], end_to_end_seconds=run["run_seconds"],
            outer=step["outer_iterations"], krylov=step["krylov_iterations"],
            head_rmse_m=comparison["head_difference"]["rms"],
            head_max_abs_m=comparison["head_difference"]["max_abs"],
            face_rmse=comparison["cpu_face_difference"]["rms"],
            relative_cell_imbalance=comparison["cpu"]["relative_cell_imbalance"]))
    output.mkdir(parents=True, exist_ok=True)
    with (output / "regional_benchmarks.csv").open(mode="w", newline="") as stream:
        writer = csv.DictWriter(f=stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    figure, (runtime, accuracy) = plt.subplots(ncols=2, figsize=(11, 4), layout="constrained")
    active = [row["active"] for row in rows]
    elapsed = [row["end_to_end_seconds"] for row in rows]
    runtime.scatter(active, elapsed, label="DarcyWarp")
    runtime.scatter(active, [row["native_seconds"] for row in rows], label="MF6")
    runtime.set(xlabel="Active cells", ylabel="Wall time (s)")
    runtime.legend()
    accuracy.scatter(elapsed, [row["head_rmse_m"] for row in rows])
    accuracy.set(xlabel="DarcyWarp wall time (s)", ylabel="Head RMSE against MF6 (m)")
    figure.savefig(output / "regional_scaling.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    archive = Path("USGS_examples/models.01/0101_0102_0103_0105_MF6_SS_Unconfined_250")
    executable = Path("/bin/modflow/mf6")
    native_directory = Path("/tmp/darcywarp-zell-native")
    output_directory = Path("/tmp/darcywarp-zell-qualification")
    device = "cuda:0"
    run_native = False
    run_audit = True
    translation_only = False
    run_darcywarp = False
    compare_solution = False
    save_saturation = True
    # Set stricter values for a tightly converged native reference.
    audit_tolerances = dict(head_gate=0.005, head_rmse_gate=1e-4,
                            saturation_head_gate=0.1, flow_atol=1e-6,
                            flow_rtol=1e-10, budget_rtol=1e-3)
    solver_options = {}
    table_cases = {}

    if run_native:
        print(native_run(archive=archive, directory=native_directory, executable=executable,
                         save_saturation=save_saturation))
    if run_audit:
        audit(archive=archive, directory=native_directory, output=output_directory, device=device,
              translation_only=translation_only, **audit_tolerances)
    if run_darcywarp:
        from DARCY_WARP_PACKAGE.mf6_import import run_mf6_model

        run_mf6_model(model_directory=archive, output_directory=output_directory, device=device,
                      solver_options=solver_options, overwrite=True)
    if compare_solution:
        audit(archive=archive, directory=native_directory, output=output_directory, device=device,
              use_darcywarp_heads=True, **audit_tolerances)
    if table_cases:
        benchmark_table(cases=table_cases, output=output_directory)

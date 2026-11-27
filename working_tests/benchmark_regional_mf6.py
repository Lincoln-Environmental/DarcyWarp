"""Audit one-layer Zell–Sanford MF6 workspaces and benchmark the CUDA importer.

Run from the repository root with ``python -m working_tests.benchmark_regional_mf6``.
Configure stages and paths in the execution guard. Native runs use fresh copies.
"""
import csv
import json
from pathlib import Path
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


def native_run(*, archive, directory, executable, save_saturation):
    """Run native MF6 in a new workspace and retain the listing and timing.

    :param archive: Untouched MF6 input directory.
    :param directory: New native working directory.
    :param executable: Absolute path to the native executable.
    :param save_saturation: Insert the NPF output-only SAVE_SATURATION option.
    """
    shutil.copytree(archive, directory)
    if save_saturation:
        npf = next(directory.glob("*.npf"))
        text = npf.read_text()
        npf.write_text(text.replace("BEGIN OPTIONS", "BEGIN OPTIONS\n  SAVE_SATURATION", 1))
    started = perf_counter()
    with (directory / "native.log").open(mode="w") as stream:
        process = subprocess.run(args=[str(executable)], cwd=directory, stdout=stream,
                                 stderr=subprocess.STDOUT, check=False)
    elapsed = perf_counter() - started
    log = (directory / "native.log").read_text()
    if process.returncode or "Normal termination of simulation." not in log:
        raise RuntimeError(f"Native MF6 failed. See {directory / 'native.log'}")
    result = dict(seconds=elapsed, version=subprocess.check_output(
        args=[str(executable), "-v"], text=True).strip())
    (directory / "native_run.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def independent_inputs(*, gwf):
    """Read static arrays and stress entries from FloPy's package data.

    :param gwf: FloPy groundwater model for a one-layer, steady one-step case.
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
    top, bottom = dis.top.array[None], dis.botm.array
    initial = gwf.ic.strt.array
    fixed = np.zeros_like(active)
    fixed_values = np.zeros_like(initial)
    entries = {}
    for kind in ("CHD", "DRN", "GHB"):
        rows = gwf.get_package(kind).stress_period_data.get_data(key=0)
        ids = np.ravel_multi_index(np.asarray(rows["cellid"].tolist()).T, active.shape)
        entries[kind] = (ids, rows)
        if kind == "CHD":
            fixed.ravel()[ids] = True
            fixed_values.ravel()[ids] = rows["head"]
    dx, dy = dis.delr.array, dis.delc.array
    recharge = gwf.get_package("RCHA").recharge.get_data(key=0)
    source = np.where(active & ~fixed, recharge * dx[None, :] * dy[:, None], 0.)
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
                            difference=describe(values=actual[mask]-expected[mask].astype(float)))
    ids, ghb = entries["GHB"]
    keep = ~raw["fixed"].ravel()[ids]
    count = active.size
    cond = np.bincount(ids[keep], weights=ghb["cond"][keep], minlength=count)
    moment = np.bincount(ids[keep], weights=(ghb["cond"]*ghb["bhead"])[keep], minlength=count)
    report["ghb_conductance"] = dict(native=describe(values=ghb["cond"]),
        difference=describe(values=case.ghb_conductance.ravel()-cond))
    occupied = cond > 0
    expected_head = moment[occupied] / cond[occupied]
    report["ghb_head"] = dict(native=describe(values=ghb["bhead"]),
        difference=describe(values=case.ghb_head.ravel()[occupied]-expected_head))
    ids, drn = entries["DRN"]
    keep = np.flatnonzero(~raw["fixed"].ravel()[ids])
    order = keep[np.argsort(ids[keep], kind="stable")]
    actual_cells = np.ravel_multi_index(np.asarray(case.drn_entries.cells).T, active.shape)
    np.testing.assert_array_equal(actual_cells, ids[order])
    for name, column, values in (("drn_conductance", "cond", case.drn_entries.conductance),
                                 ("drn_elevation", "elev", case.drn_entries.first_level)):
        report[name] = dict(native=describe(values=drn[column]),
                            difference=describe(values=values-drn[column][order]))
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
        budget = {name.decode().strip(): reader.get_data(text=name.decode().strip())[-1]
                  for name in reader.get_unique_record_names()}
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


def package_metrics(*, records, entries, case, head, fields):
    """Verify CBC user-node IDs, CHD suppression and per-entry exchanges.

    :param records: Native final package budget arrays, with inflow positive.
    :param entries: Independent FloPy list records.
    :param case: Effective DarcyWarp operator.
    :param head: Evaluation head field.
    :param fields: CPU reference physical fields.
    """
    cells, summaries = {}, {}
    for kind in ("CHD", "DRN", "GHB", "RCHA"):
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
        if kind == "GHB":
            expected = rows["cond"][entry] * (rows["bhead"][entry]-head.ravel()[ids])
        elif kind == "DRN":
            expected = -rows["cond"][entry] * np.maximum(head.ravel()[ids]-rows["elev"][entry], 0.)
        elif kind == "CHD":
            expected = fields["chd_cell_rate"].ravel()[ids]
        else:
            expected = case.source.ravel()[ids]
        if kind != "CHD":
            overlap = case.fixed.ravel()[ids] != 0
            expected = np.where(overlap, 0., expected)
            summary.update(chd_overlap=int(overlap.sum()), flowing_overlap=int(np.count_nonzero(q[overlap])))
        summary["entry_difference"] = describe(values=expected-q)
        summaries[kind] = summary
    return summaries, cells


def audit(*, archive, directory, output, device, use_darcywarp_heads=False, translation_only=False):
    """Compare translation, native states and both DarcyWarp evaluators.

    :param archive: Original MF6 workspace.
    :param directory: Native workspace containing final budgets and heads.
    :param output: Directory for JSON, raw mapped differences and figures.
    :param device: CUDA device to exercise the production physical kernels.
    :param use_darcywarp_heads: Evaluate a separately accepted importer result.
    :param translation_only: Audit operator inputs without requiring native outputs.
    """
    from DARCY_WARP_PACKAGE.mf6_import import (
        _schedules, _stress_period, import_simulation, load_mf6_workspace,
    )
    from DARCY_WARP_PACKAGE.solvers.npf662_3d import case_from_model, independent_reference
    from DARCY_WARP_PACKAGE.solvers.npf662_newton import NPF662Workspace
    from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D

    started = perf_counter()
    simulation = load_mf6_workspace(directory=archive)
    model = import_simulation(simulation=simulation)
    nz, ny, nx = model.shape
    if nz != 1 or len(model.periods) != 1 or model.periods[0][0] or len(model.periods[0][1]) != 1:
        raise ValueError("This regional audit expects a one-layer, one-step steady model")
    raw, entries = independent_inputs(gwf=simulation.get_model())
    forcing = _stress_period(model=model, schedules=_schedules(model=model), kper=0)
    imported_seconds = perf_counter()-started
    with WarpDarcySolver3D(nx=nx, ny=ny, nz=nz, dx=model.dx, dy=model.dy, device=device,
                           formulation="mf6_6_6_2_npf", implementation="fast", solver="kcycle") as solver:
        started = perf_counter()
        solver.build_from_layers(z_interfaces=model.interfaces, kx_field=model.kx,
                                 ky_field=model.ky, kz_field=model.kz, active=model.active,
                                 initial_head=model.initial, **forcing)
        case = case_from_model(model=solver, top=0., cell_type=model.cell_type)
        build_seconds = perf_counter()-started
        translation = translation_metrics(raw=raw, entries=entries, case=case)
        if translation_only:
            write_audit(output=output, label="translation", report=translation, maps={}, active=raw["active"])
            return
        native_head, keys, native_flows, native_residual, budget = read_native(
            directory=directory, active=raw["active"])
        head = native_head
        if use_darcywarp_heads:
            with flopy.utils.HeadFile(output / "darcywarp.hds") as reader:
                head = reader.get_data()
        reference, _ = independent_reference(model=solver, case=case)
        metrics, cpu = reference.evaluate(head=head)
        initial = np.where(case.fixed, case.values, case.initial)
        initial_saturation = reference.face_terms(head=initial)[3]
        workspace = NPF662Workspace(model=solver, case=case, restart=2,
                                    krylov_budget=0, reference=reference)
        try:
            workspace.head.assign(np.where(case.active, head, 0.).ravel())
            gpu_metrics = workspace.evaluate(trial=False)
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
        packages, package_cells = package_metrics(records=budget, entries=entries,
                                                  case=case, head=head, fields=cpu)
        reference_divergence = (np.bincount(left, weights=native_q, minlength=head.size)
                                - np.bincount(right, weights=native_q, minlength=head.size))
        reassembled_native = sum(package_cells.values()) - reference_divergence.reshape(head.shape)
        physical = {name: metrics[name] for name in ("physical_rms", "head_residual_rms",
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
                      native_residual=describe(values=native_residual[free]),
                      native_residual_reassembly=describe(values=(reassembled_native-native_residual)[free]),
                      head_difference=describe(values=(head-native_head)[raw["active"]]),
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
                "CPU residual difference (m³/day)": cpu["residual"]-native_residual,
                "GHB difference (m³/day)": -cpu["ghb_outflow"]-package_cells["GHB"],
                "DRN difference (m³/day)": -cpu["drn_outflow"]-package_cells["DRN"],
                "CHD difference (m³/day)": cpu["chd_cell_rate"]-package_cells["CHD"]}
        if "DATA-SAT" in budget:
            data = budget["DATA-SAT"]
            saved = np.zeros(head.size)
            saved[data["node"]-1] = data["sat"]
            dh = native_head.ravel()[left]-native_head.ravel()[right]
            donor = np.where((dh > 0) | ((dh == 0) & (reference.axes == 1)), left, right)
            c_native = reference.c0*saved[donor]
            report["saved_saturation_face_check"] = describe(values=c_native*dh-native_q)
            report["cpu_native_conductance"] = describe(values=cpu["conductance"]-c_native)
            saturation_delta = cpu["saturation"]-saved.reshape(head.shape)
            report["saturation_difference"] = describe(values=saturation_delta[raw["active"]])
            maps["Saturated thickness difference (m)"] = saturation_delta*raw["thicknesses"]
        label = "solution" if use_darcywarp_heads else "native_heads"
        write_audit(output=output, label=label, report=report, maps=maps, active=raw["active"])


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

    import matplotlib.pyplot as plt

    np.savez_compressed(output / f"{label}_maps.npz", **maps)
    figure, axes = plt.subplots(nrows=(len(maps)+1)//2, ncols=2, figsize=(11, 12),
                               layout="constrained")
    for axis, (name, values) in zip(axes.flat, maps.items()):
        grid = np.where(active[0], values[0], np.nan)
        limit = np.nanmax(np.abs(grid))
        image = axis.imshow(grid, cmap="RdBu_r", vmin=-limit, vmax=limit)
        axis.set_title(name)
        figure.colorbar(image, ax=axis, shrink=.7)
    for axis in axes.flat[len(maps):]:
        axis.set_visible(False)
    figure.savefig(output / f"{label}_maps.png", dpi=160)
    plt.close(figure)


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
    solver_options = {}
    table_cases = {}

    if run_native:
        print(native_run(archive=archive, directory=native_directory, executable=executable,
                         save_saturation=save_saturation))
    if run_audit:
        audit(archive=archive, directory=native_directory, output=output_directory, device=device,
              translation_only=translation_only)
    if run_darcywarp:
        from DARCY_WARP_PACKAGE.mf6_import import run_mf6_model

        run_mf6_model(model_directory=archive, output_directory=output_directory, device=device,
                      solver_options=solver_options, overwrite=True)
    if compare_solution:
        audit(archive=archive, directory=native_directory, output=output_directory, device=device,
              use_darcywarp_heads=True)
    if table_cases:
        benchmark_table(cases=table_cases, output=output_directory)

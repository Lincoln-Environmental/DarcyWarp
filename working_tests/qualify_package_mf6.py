"""Pinned MF6 twins of the production package qualification fields and schedules."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np


def native_boundary_rows(*, fields, package, stage_offset):
    """Convert authoritative sparse entries to MF6 order without merging entries.

    :param fields: Actual groundwater builder fields.
    :param package: DRN or RIV.
    :param stage_offset: Period-specific river stage adjustment.
    :return: MF6 stress-period rows.
    """
    from DARCY_WARP_PACKAGE.physics.drn_riv_3d import validate_boundary_entries
    entries = validate_boundary_entries(entries=fields[f'{package.lower()}_entries'], package=package,
        shape=fields['active'].shape, active=fields['active'], fixed=fields['bc_mask'])
    if entries is None:
        return []
    rows = []
    for index, cell in enumerate(entries.cells):
        cellid = tuple(int(value) for value in cell)
        if package == 'DRN':
            rows.append((cellid, float(entries.first_level[index]), float(entries.conductance[index])))
        else:
            rows.append((cellid, float(entries.second_level[index] + stage_offset),
                float(entries.conductance[index]), float(entries.third_level[index])))
    return rows


def run_native(*, fields, durations, well_factors, recharge_factors, stage_offsets, transient,
               ss, sy, cell_type, ims_controls, executable, repeats, directory, well_precision='float64'):
    """Build and repeatedly run the exact pinned native reference.

    :param fields: Identical grid, material, sources and entry definitions.
    :param durations: Actual period durations.
    :param well_factors: Actual WEL multipliers.
    :param recharge_factors: Actual recharge multipliers.
    :param stage_offsets: Actual RIV stage changes.
    :param transient: Enable physical STO from the original IC.
    :param ss: Specific storage in inverse length.
    :param sy: Specific yield.
    :param cell_type: NPF cell type.
    :param ims_controls: Existing native solver controls.
    :param executable: Pinned MF6 6.6.2 executable.
    :param repeats: Number of native engine measurements from identical inputs.
    :param directory: Native files and compact evidence destination.
    :param well_precision: Match actual source precision in the numerical model.
    :return: Compact native timing and physical-budget evidence.
    :raises RuntimeError: If MF6 fails or does not terminate normally.
    """
    import flopy
    from DARCY_WARP_PACKAGE.validation.npf662_mf6 import pinned_native_executable
    from DARCY_WARP_PACKAGE.physics.recharge_3d import surface_recharge_cells
    from DARCY_WARP_PACKAGE.physics.wel import validate_well_entries
    binary, version, checksum, build = pinned_native_executable(executable=executable)
    directory.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    # Match the existing pinned adapter's exact float64 text round trip.
    # FloPy defaults to eight digits, which changes physical inputs.
    shape = fields['active'].shape
    nz, ny, nx = shape
    name = 'pkgqual'
    simulation = flopy.mf6.MFSimulation(sim_name=name, sim_ws=str(directory), exe_name=str(binary))
    flopy.mf6.ModflowTdis(simulation, nper=len(durations), time_units='DAYS',
        perioddata=[(float(duration), 1, 1.) for duration in durations])
    simulation.simulation_data.float_precision = 17
    simulation.simulation_data.float_characters = 25
    gwf = flopy.mf6.ModflowGwf(simulation, modelname=name, save_flows=True)
    flopy.mf6.ModflowIms(simulation, **ims_controls)
    flopy.mf6.ModflowGwfdis(gwf, nlay=nz, nrow=ny, ncol=nx, delr=fields['dx'], delc=fields['dy'],
        top=fields['top_elevation'], botm=fields['top_elevation'] - fields['dz'] * (np.arange(nz) + 1),
        idomain=fields['active'])
    flopy.mf6.ModflowGwfic(gwf, strt=fields['initial_head'])
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=cell_type, k=fields['kx_field'], k22=fields['ky_field'],
        k33=fields['kz_field'], save_flows=True)
    if transient:
        flopy.mf6.ModflowGwfsto(gwf, iconvert=cell_type, ss=ss, sy=sy, transient={0: True}, save_flows=True)
    receivers = surface_recharge_cells(active=fields['active'], bc_mask=fields['bc_mask'])
    cells = np.column_stack(np.unravel_index(receivers, shape))
    rate = np.broadcast_to(np.asarray(fields['R_field']), shape[1:]).ravel()[receivers % (ny * nx)]
    recharge = {period: [(tuple(int(value) for value in cell), float(value * recharge_factors[period]))
        for cell, value in zip(cells, rate)] for period in range(len(durations))}
    flopy.mf6.ModflowGwfrch(gwf, maxbound=len(receivers), stress_period_data=recharge, save_flows=True)
    wells = validate_well_entries(entries=fields['wel_entries'], shape=shape, active=fields['active'])
    rate_type = np.dtype(well_precision).type
    well_rows = {period: [(tuple(int(value) for value in cell), float(rate_type(value * well_factors[period])))
        for cell, value in zip(wells.cells, wells.rates)] for period in range(len(durations))}
    flopy.mf6.ModflowGwfwel(gwf, maxbound=len(wells), stress_period_data=well_rows, save_flows=True)
    packages = ['WEL', 'RCH', 'CHD']
    for package in ('DRN', 'RIV'):
        if fields[f'{package.lower()}_entries'] is None:
            continue
        rows = {period: native_boundary_rows(fields=fields, package=package, stage_offset=stage_offsets[period])
                for period in range(len(durations))}
        constructor = flopy.mf6.ModflowGwfdrn if package == 'DRN' else flopy.mf6.ModflowGwfriv
        constructor(gwf, maxbound=len(rows[0]), stress_period_data=rows, save_flows=True)
        packages.append(package)
    conductance = fields.get('ghb_conductance')
    if conductance is not None:
        ghb_cells = np.column_stack(np.nonzero(conductance > 0.))
        boundary_head = np.broadcast_to(fields['ghb_head'], shape)
        rows = [(tuple(int(value) for value in cell), float(boundary_head[tuple(cell)]), float(conductance[tuple(cell)]))
                for cell in ghb_cells]
        flopy.mf6.ModflowGwfghb(gwf, maxbound=len(rows), stress_period_data=rows, save_flows=True)
        packages.append('GHB')
    fixed_cells = np.column_stack(np.nonzero(fields['bc_mask']))
    rows = [(tuple(int(value) for value in cell), float(fields['bc_values'][tuple(cell)])) for cell in fixed_cells]
    flopy.mf6.ModflowGwfchd(gwf, maxbound=len(rows), stress_period_data=rows, save_flows=True)
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord=f'{name}.hds', budget_filerecord=f'{name}.cbc',
        saverecord=[('HEAD', 'ALL'), ('BUDGET', 'ALL')])
    simulation.write_simulation(silent=True)
    construction = perf_counter() - started
    started = perf_counter()
    input_suffixes = {'.nam', '.tdis', '.ims', '.dis', '.ic', '.npf', '.sto', '.rch', '.wel', '.drn', '.riv', '.ghb', '.chd', '.oc'}
    input_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.iterdir() if path.suffix in input_suffixes}
    input_audit_seconds = perf_counter() - started
    manifest = dict(version=version, sha256=checksum, build=build, shape=list(shape), transient=transient,
        well_precision=well_precision, ims_controls=ims_controls, construction_seconds=construction,
        input_precision=17, input_float_characters=25, input_sha256=input_hashes,
        input_audit_seconds=input_audit_seconds, durations=[float(value) for value in durations],
        packages=packages, engine_seconds=[], engine_logs=[], requested_repeats=repeats)
    manifest_path = directory / 'run_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    for repeat in range(repeats):
        started = perf_counter()
        success, lines = simulation.run_simulation(silent=True, report=True)
        elapsed = perf_counter() - started
        log_name = f'engine_{repeat}.txt'
        (directory / log_name).write_text('\n'.join(lines))
        if not success or not any('Normal termination' in line for line in lines):
            raise RuntimeError(f'MF6 failed at repeat {repeat}; see engine log')
        manifest['engine_seconds'].append(elapsed)
        manifest['engine_logs'].append(log_name)
        manifest_path.write_text(json.dumps(manifest, indent=2))
    return extract_completed_native(directory=directory, manifest=manifest)


def extract_completed_native(*, directory, manifest):
    """Extract only the saved native equation whose inputs this manifest owns.

    :param directory: Completed native workspace.
    :param manifest: Input hashes, accepted engine runs and physical time grid.
    :return: Native heads, per-entry budgets and accepted volume evidence.
    :raises AssertionError: If saved equation inputs or accepted time grids differ.
    """
    import flopy
    for name, expected in manifest['input_sha256'].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected, name
    for name in manifest['engine_logs']:
        assert 'Normal termination' in (directory / name).read_text(), name
    durations = manifest['durations']
    packages = manifest['packages']
    transient = manifest['transient']
    engine = manifest['engine_seconds']
    started = perf_counter()
    reader = flopy.utils.HeadFile(str(directory / 'pkgqual.hds'), precision='double')
    heads = reader.get_alldata()
    times = reader.get_times()
    reader.close()
    np.testing.assert_allclose(times, np.cumsum(durations), rtol=0., atol=1.e-12)
    budget = flopy.utils.CellBudgetFile(str(directory / 'pkgqual.cbc'), precision='double')
    arrays = dict(heads=heads, times=np.asarray(times))
    record_names = {name.decode().strip() for name in budget.get_unique_record_names()}
    periods = []
    for period, duration in enumerate(durations):
        rates = {}
        for package in packages:
            data = budget.get_data(text=package, kstpkper=(0, period))[0]
            values = np.asarray(data['q'], dtype=np.float64) if data.dtype.names else np.asarray(data).ravel()
            arrays[f'{package}_{period}'] = values
            rates[package] = dict(inflow=float(np.sum(np.maximum(values, 0.))),
                outflow=float(np.sum(np.maximum(-values, 0.))), net=float(np.sum(values)))
        if transient:
            for storage_name in ('STO-SS', 'STO-SY'):
                if storage_name not in record_names:
                    continue
                records = budget.get_data(text=storage_name, kstpkper=(0, period))
                if not records:
                    continue
                values = records[0]
                if values.dtype.names:
                    values = values['q']
                arrays[f'{storage_name.replace(chr(45), chr(95))}_{period}'] = np.asarray(values)
        periods.append(dict(dt=float(duration), packages=rates,
            volumes={package: {key: value * duration for key, value in rates[package].items()} for package in rates}))
    budget.close()
    np.savez_compressed(directory / 'reference_outputs.npz', **arrays)
    result = dict(manifest, periods=periods,
        engine_summary=(dict(median=float(np.median(engine)), minimum=min(engine), maximum=max(engine)) if engine else None),
        extraction_seconds=perf_counter() - started,
        output_sha256=hashlib.sha256((directory / 'reference_outputs.npz').read_bytes()).hexdigest())
    (directory / 'reference.json').write_text(json.dumps(result, indent=2))
    print(manifest['shape'], 'transient' if transient else 'steady updates', result['engine_summary'], flush=True)
    return result



if __name__ == "__main__":
    os.environ.setdefault('DARCY_FLOAT', 'float64')
    from qualify_package_kcycle import boundary_controls, regional_fields
    from DARCY_WARP_PACKAGE.project_base import require_mf6
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid', choices=('100k', '1M', 'all'), default='all')
    parser.add_argument('--mode', choices=('steady', 'history', 'all'), default='all')
    parser.add_argument('--boundary-case', choices=('packages', 'gaining_losing', 'wel_only'), default='packages')
    parser.add_argument('--profile', choices=('production', 'strict'), default='production')
    parser.add_argument('--strategy', choices=('original', 'ilut8', 'cg_ilu0'), default='original')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, default=Path('/tmp/darcywarp-package-mf6-exact'))
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    controls = json.loads((root / 'working_tests/wel_regional_qualification_controls.json').read_text())
    grids = {'100k': (10, 112, 90), '1M': (5, 450, 450)}
    ims_controls = dict(complexity='COMPLEX', linear_acceleration='BICGSTAB',
        outer_maximum=200, outer_dvclose=1.e-11, inner_maximum=200, inner_dvclose=1.e-12,
        rcloserecord=[1.e-12, 'strict'])
    if arguments.profile == 'production':
        from sanity_3d_controls import read_profiles
        qualified_native = read_profiles(root=root)['mf6']
        steady_ims = dict(qualified_native['steady_confined'])
        history_ims = dict(qualified_native['transient'])
    else:
        steady_ims = dict(ims_controls)
        history_ims = dict(ims_controls)
    arguments.output = arguments.output / arguments.profile
    if arguments.strategy == 'ilut8':
        overrides = dict(under_relaxation='NONE', backtracking_number=0,
            preconditioner_levels=8, preconditioner_drop_tolerance=1.e-4,
            number_orthogonalizations=8, relaxation_factor=0.)
        steady_ims.update(overrides)
        history_ims.update(overrides)
        arguments.output = arguments.output / 'ilut8'
    if arguments.strategy == 'cg_ilu0':
        overrides = dict(complexity='SIMPLE', under_relaxation='NONE', backtracking_number=0,
            linear_acceleration='CG', preconditioner_levels=0, preconditioner_drop_tolerance=0.,
            number_orthogonalizations=0, relaxation_factor=0.)
        steady_ims.update(overrides)
        history_ims.update(overrides)
        arguments.output = arguments.output / 'cg_ilu0'
    controls = boundary_controls(controls=controls, variant=arguments.boundary_case)
    if arguments.boundary_case != 'packages':
        arguments.output = arguments.output / arguments.boundary_case
    executable = require_mf6()
    for label, shape in grids.items():
        if arguments.grid not in ('all', label):
            continue
        fields = regional_fields(shape=shape, abstraction_fraction=controls['abstraction_fraction'],
            packages=arguments.boundary_case != 'wel_only', seed=controls['seed'], regional_controls=controls)
        if arguments.mode in ('all', 'steady'):
            well_factors = [1., *controls['calibration_wel_factors']]
            recharge_factors = [1., *controls['calibration_recharge_factors']]
            stage_offsets = [0., *controls['calibration_stage_offsets']]
            durations = [1.] * len(well_factors)
            run_native(fields=fields, durations=durations, well_factors=well_factors,
                recharge_factors=recharge_factors, stage_offsets=stage_offsets, transient=False,
                ss=controls['specific_storage_per_m'], sy=controls['specific_yield'], cell_type=0,
                ims_controls=steady_ims, executable=executable, repeats=arguments.repeats,
                directory=arguments.output / label / 'steady')
        if arguments.mode in ('all', 'history'):
            steps = controls['history_steps']
            well_factors = np.resize(controls['history_wel_pattern'], steps)
            recharge_factors = np.resize(controls['history_recharge_pattern'], steps)
            durations = [controls['history_dt_days']] * steps
            stage_offsets = [0.] * steps
            run_native(fields=fields, durations=durations, well_factors=well_factors,
                recharge_factors=recharge_factors, stage_offsets=stage_offsets, transient=True,
                ss=controls['specific_storage_per_m'], sy=controls['specific_yield'], cell_type=0,
                ims_controls=history_ims, executable=executable, repeats=arguments.repeats,
                directory=arguments.output / label / 'history')

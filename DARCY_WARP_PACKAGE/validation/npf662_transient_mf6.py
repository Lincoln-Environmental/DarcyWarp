"""Live MF6 NPF/STO histories, with native face/source/storage budget decoding."""
from dataclasses import replace
from pathlib import Path
import hashlib
import subprocess
import numpy as np
from DARCY_WARP_PACKAGE.validation.npf662_reference import NPF662Reference
from DARCY_WARP_PACKAGE.validation.npf662_transient_reference import NPF662TransientReference
from DARCY_WARP_PACKAGE.validation.terrain_case import atomic_json
from DARCY_WARP_PACKAGE.validation.head_checks import finite_diagnostics


def run_transient_mf6(*, case, directory, executable, source_factors, dt, ss, sy,
                       executable_sha256='21efe1534afd1d9f19227938ff70cca4b2f216d326f7d73073b4d8ebe1ed7176',
                       timeout_seconds=600., source_package='wel'):
    """Run one native simulation with one implicit step per stress period.

    :param case: Canonical NPF geometry, materials, boundary and initial state.
    :param directory: New native workspace.
    :param executable: MF6 6.6.2 binary path.
    :param source_factors: Explicit per-period multipliers for the original source.
    :param dt: Step duration in days.
    :param ss: Specific storage in inverse metres.
    :param sy: Specific yield, used only in convertible cells.
    :param executable_sha256: Explicit expected binary build hash.
    :param timeout_seconds: Native process wall-time limit.
    :param source_package: WEL or fixed-cell top-array RCH representation.
    """
    import flopy
    from flopy.mf6.utils.binarygrid_util import MfGrdFile
    case.validate()
    NPF662TransientReference(case=case, previous=case.initial, dt=dt, ss=ss, sy=sy)
    factors = np.asarray(source_factors, dtype=float)
    if factors.ndim != 1 or not factors.size or not np.isfinite(factors).all():
        raise ValueError('Finite nonempty source schedule required')
    executable = Path(executable).resolve()
    version = subprocess.check_output([str(executable), '-v'], text=True).strip()
    checksum = hashlib.sha256(executable.read_bytes()).hexdigest()
    if version != 'mf6: 6.6.2 05/12/2025' or checksum != executable_sha256:
        raise ValueError('MF6 executable does not match the explicit version/build pin')
    if source_package not in ('wel', 'rch'):
        raise ValueError('Source package must be wel or rch')
    if source_package == 'rch' and (np.any(case.source[1:]) or np.any(case.source < 0.) or np.any(factors < 0.)):
        raise ValueError('RCH requires nonnegative top-layer forcing')
    directory.mkdir(parents=True, exist_ok=False)
    nz, ny, nx = case.active.shape
    reverse = (slice(None), slice(None, None, -1), slice(None))
    count = factors.size
    ss = np.broadcast_to(ss, case.active.shape).copy()
    sy = np.broadcast_to(sy, case.active.shape).copy()
    sim = flopy.mf6.MFSimulation(sim_name='transient', sim_ws=str(directory), exe_name=str(executable), verbosity_level=0)
    sim.simulation_data.float_precision = 17
    sim.simulation_data.float_characters = 25
    flopy.mf6.ModflowTdis(sim, time_units='DAYS', nper=count, perioddata=[(dt, 1, 1.)]*count)
    flopy.mf6.ModflowIms(sim, complexity='COMPLEX', print_option='ALL', outer_maximum=500,
        outer_dvclose=1e-9, inner_maximum=2000, inner_dvclose=1e-11,
        rcloserecord=[1e-10, 'STRICT'], linear_acceleration='BICGSTAB',
        under_relaxation='NONE', scaling_method='DIAGONAL', preconditioner_levels=8,
        preconditioner_drop_tolerance=1e-4, relaxation_factor=0., reordering_method='RCM',
        number_orthogonalizations=8, csv_outer_output_filerecord='outer.csv', csv_inner_output_filerecord='inner.csv')
    gwf = flopy.mf6.ModflowGwf(sim, modelname='transient', save_flows=True,
        newtonoptions='UNDER_RELAXATION' if np.any(case.cell_types) else None)
    flopy.mf6.ModflowGwfdis(gwf, length_units='METERS', nlay=nz, nrow=ny, ncol=nx,
        delr=case.dx, delc=case.dy, top=case.tops[0, ::-1], botm=case.bottoms[reverse], idomain=case.active[reverse])
    flopy.mf6.ModflowGwfic(gwf, strt=case.initial[reverse])
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=case.cell_types[reverse], k=case.kx[reverse],
        k22=case.ky[reverse], k33=case.kz[reverse], save_flows=True, save_saturation=True,
        xt3doptions=None, perched=False, thickstrt=False, cvoptions=None, rewet_record=None)
    flopy.mf6.ModflowGwfsto(gwf, iconvert=case.cell_types[reverse], ss=ss[reverse], sy=sy[reverse],
        transient={0: True}, save_flows=True)
    boundary = [((int(k), ny-1-int(j), int(i)), float(case.values[k, j, i]))
        for k, j, i in np.argwhere((case.active != 0)&(case.fixed != 0))]
    if boundary:
        flopy.mf6.ModflowGwfchd(gwf, stress_period_data={0: boundary}, save_flows=True)
    if source_package == 'wel':
        indices = np.argwhere((case.active != 0)&(case.fixed == 0)&(case.source != 0.))
        schedule = {period: [((int(k), ny-1-int(j), int(i)), float(case.source[k, j, i]*factor))
            for k, j, i in indices] for period, factor in enumerate(factors)}
        if indices.size:
            flopy.mf6.ModflowGwfwel(gwf, pname='SOURCE', stress_period_data=schedule, save_flows=True)
    else:
        schedule = {period: (case.source[0]*factor/case.dx/case.dy)[::-1]
            for period, factor in enumerate(factors)}
        flopy.mf6.ModflowGwfrcha(gwf, pname='RECHARGE', readasarrays=True, fixed_cell=True,
            irch=np.zeros((ny, nx), dtype=int), recharge=schedule, save_flows=True)
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord='transient.hds', budget_filerecord='transient.cbc',
        saverecord=[('HEAD', 'ALL'), ('BUDGET', 'ALL')], printrecord=[('BUDGET', 'ALL')])
    sim.write_simulation(silent=True)
    run = subprocess.run([str(executable)], cwd=directory, capture_output=True, text=True,
        timeout=timeout_seconds, check=False)
    directory.joinpath('stdout.log').write_text(run.stdout+run.stderr)
    if run.returncode != 0 or 'Normal termination of simulation' not in run.stdout:
        raise RuntimeError('Native transient simulation failed; inspect retained stdout/listing')
    head_file = flopy.utils.HeadFile(str(directory/'transient.hds'), precision='double')
    times = np.asarray(head_file.get_times())
    if not np.array_equal(times, dt*np.arange(1, count+1)):
        raise ValueError('Native saved times differ from the requested weekly endpoints')
    heads = head_file.get_alldata()[:, :, ::-1, :].copy()
    head_file.close()
    grid = MfGrdFile(str(directory/'transient.dis.grb'), verbose=False)
    ia, ja = grid.ia, grid.ja
    mapping = np.arange(case.active.size).reshape(case.active.shape)[reverse].ravel()
    reference = NPF662Reference(case=case)
    edge_index = {(int(a), int(b)): p for p, (a, b) in enumerate(zip(reference.left, reference.right))}
    positions = []
    destinations = []
    signs = []
    for node in range(case.active.size):
        for p in range(ia[node]+1, ia[node+1]):
            a, b = int(mapping[node]), int(mapping[int(ja[p])])
            if a < b:
                positions.append(p)
                destinations.append(edge_index[(a, b)])
                signs.append(-1.)
    if len(destinations) != reference.left.size or len(set(destinations)) != len(destinations):
        raise ValueError('Native connectivity differs from canonical physical faces')
    positions = np.asarray(positions, dtype=int)
    destinations = np.asarray(destinations, dtype=int)
    signs = np.asarray(signs)
    budget = flopy.utils.CellBudgetFile(str(directory/'transient.cbc'), precision='double')
    record_names = {value.decode().strip() for value in budget.get_unique_record_names()}
    fluxes = []
    storage_rates = []
    rows = []
    previous = case.initial.copy()
    for period, head in enumerate(heads):
        if not np.isfinite(head[case.active != 0]).all() or np.any(np.abs(head[case.active != 0]) >= 1e20):
            raise ValueError('Native transient heads contain invalid or inactive markers')
        connection = budget.get_data(kstpkper=(0, period), text='FLOW-JA-FACE')[0].ravel()
        flux = np.zeros(reference.left.size)
        flux[destinations] = connection[positions]*signs
        storage = np.zeros(case.active.size)
        for name in ['STO-SS', 'STO-SY']:
            records = budget.get_data(kstpkper=(0, period), text=name) if name in record_names else []
            if records:
                storage[mapping] -= records[0].ravel()
        realised = np.zeros(case.active.size)
        source_name = 'WEL' if source_package == 'wel' else 'RCHA'
        for records in budget.get_data(kstpkper=(0, period), text=source_name) if source_name in record_names else []:
            for record in records:
                realised[mapping[int(record['node'])-1]] += float(record['q'])
        step_case = replace(case, source=case.source*factors[period])
        auditor = NPF662TransientReference(case=step_case, previous=previous, dt=dt, ss=ss, sy=sy)
        checks, fields = auditor.evaluate(head=head)
        divergence = (reference.incidence@flux).reshape(case.active.shape)
        native_residual = np.where(reference.free, realised.reshape(case.active.shape)-storage.reshape(case.active.shape)-divergence, 0.)
        native_relative = float(np.sum(np.abs(native_residual[reference.free]))/max(checks['gross_throughput'], np.finfo(float).tiny))
        source_expected = np.where(reference.free, step_case.source, 0.)
        allocation_tolerance = np.maximum(1e-14, 8.*np.finfo(float).eps*np.abs(source_expected))
        allocation_valid = bool(np.all(np.abs(realised.reshape(case.active.shape)-source_expected) <= allocation_tolerance))
        storage_error = float(np.sum(np.abs(storage.reshape(case.active.shape)-fields['storage_rate']))/
            max(checks['gross_throughput'], np.finfo(float).tiny))
        accepted = bool(checks['independent_accepted'] and native_relative <= 1e-8
            and allocation_valid and storage_error <= 1e-8)
        rows.append(dict(period=period+1, time_days=float(times[period]), accepted=accepted,
            physical=checks, native_relative_cell_imbalance=native_relative,
            source_allocation_valid=allocation_valid, storage_difference_relative_throughput=storage_error))
        fluxes.append(flux)
        storage_rates.append(storage.reshape(case.active.shape))
        previous = head.copy()
    budget.close()
    arrays = dict(heads=heads, fluxes=np.asarray(fluxes), storage_rates=np.asarray(storage_rates), times=times)
    temporary = directory/'.decoded.npz.tmp'
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(directory/'decoded.npz')
    result = dict(passed=all(row['accepted'] for row in rows), periods=rows,
        mf6_version=version, mf6_sha256=checksum, source_package=source_package,
        dt_days=dt, source_factors=factors.tolist(), storage='Default MF6 6.6.2 Ss/Sy, no STO optional formulation flags')
    atomic_json(path=directory/'summary.json', value=finite_diagnostics(value=result))
    return arrays, result

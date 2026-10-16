"""Read immutable baseline evidence, add audits/profiling in a NEW directory."""
from pathlib import Path
from time import perf_counter
import csv
import json
import shutil

import flopy
import numpy as np
import scipy
import warp as wp

from DARCY_WARP_PACKAGE.validation.npf662_reference import load_npf662_case, NPF662Reference
from DARCY_WARP_PACKAGE.validation.npf662_matrix_free import NPF662MatrixFreeReference
from DARCY_WARP_PACKAGE.validation.npf662_mf6 import run_reference_mf6
import DARCY_WARP_PACKAGE.solvers_3d as solvers
from benchmark_npf662_baseline import (
    CycleObserver, MemorySampler, NewtonProbe, digest_file, make_linear_problem,
    write_json, save_arrays, archive_solution, compare_solution, snapshot_sources,
)
from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D


def read_json(*, path):
    return json.loads(s=path.read_text())


def audit_native(*, baseline, name, output, native_directory):
    case = load_npf662_case(path=baseline/name/'inputs.npz')
    with np.load(file=native_directory/'result.npz', allow_pickle=False) as saved:
        mapping = saved['node_to_user']
        head = saved['head']; native_flux = saved['flux_native']
        saturation = saved['native_saturation']
    reference = NPF662MatrixFreeReference(case=case)
    checks, fields = reference.evaluate(head=head)
    divergence = reference.divergence(flux=native_flux)
    chd = np.zeros(shape=case.active.size)
    budget = flopy.utils.CellBudgetFile(filename=str(native_directory/'npf662.cbc'), precision='double')
    for record in budget.get_data(text='CHD')[-1]:
        chd[mapping[int(record['node'])-1]] += float(record['q'])
    chd = chd.reshape(case.active.shape)
    full_residual = case.source+chd-divergence
    budget.close()
    output.mkdir(parents=True, exist_ok=False)
    save_arrays(directory=output, arrays=dict(actual_native_chd_exchange=chd,
        native_full_cell_budget_residual=full_residual))
    iterations = {}
    for label in ('outer', 'inner'):
        with (native_directory/f'{label}.csv').open() as stream:
            reader = csv.DictReader(f=stream)
            rows = list(reader)
        iterations[label] = dict(records=len(rows), last=rows[-1] if rows else None)
    qerror = native_flux-fields['flux']
    summary = dict(independent=checks, native_flow_law_max_abs_m3_day=float(np.max(np.abs(qerror))),
        effective_native_solver=read_json(path=native_directory/'summary.json')['native_solver_options'],
        native_saturation_max_error=float(np.max(np.abs(saturation-fields['saturation']))),
        native_chd_allocation_nonfixed_max=float(np.max(np.abs(chd[case.fixed == 0]))),
        native_chd_total_m3_day=float(np.sum(chd)), source_total_m3_day=float(np.sum(case.source)),
        native_chd_inflow_m3_day=float(np.sum(np.maximum(chd, 0.))),
        native_chd_outflow_m3_day=float(np.sum(np.maximum(-chd, 0.))),
        native_full_cell_residual_max_m3_day=float(np.max(np.abs(full_residual))),
        native_full_cell_residual_rms_m3_day=float(np.sqrt(np.mean(full_residual**2))),
        source_cells=int(np.count_nonzero(case.source)), source_top_cells=int(np.count_nonzero(case.source[0])),
        initial_head_min_m=float(case.initial.min()), initial_head_max_m=float(case.initial.max()),
        final_head_min_m=float(head.min()), final_head_max_m=float(head.max()),
        thickness_min_m=float(case.thicknesses.min()), thickness_max_m=float(case.thicknesses.max()),
        conductivity_ranges={key: [float(getattr(case, key).min()), float(getattr(case, key).max())] for key in ('kx', 'ky', 'kz')},
        saturation_min=float(saturation.min()), saturation_max=float(saturation.max()),
        actual_source_check=bool(np.count_nonzero(case.source[1:]) == 0 and np.all(case.source[case.fixed != 0] == 0.)),
        iterations=iterations)
    write_json(path=output/'audit.json', value=summary)
    return summary


class LaunchReadbackProbe:
    """Separate instrumented run: API launch dispatch and .numpy readback waits."""
    def __init__(self):
        self.launches = 0; self.launch_api_seconds = 0.; self.readbacks = 0; self.readback_seconds = 0.
        self.in_cycle = False

    def __getattr__(self, name):
        return getattr(wp, name)

    def launch(self, *args, **kwargs):
        start = perf_counter(); result = wp.launch(*args, **kwargs)
        if self.in_cycle:
            self.launches += 1; self.launch_api_seconds += perf_counter()-start
        return result

    def install(self):
        self.old_wp = solvers.wp; self.old_numpy = wp.array.numpy
        probe = self

        def measured_numpy(array, *args, **kwargs):
            start = perf_counter(); result = probe.old_numpy(array, *args, **kwargs)
            if probe.in_cycle:
                probe.readbacks += 1; probe.readback_seconds += perf_counter()-start
            return result

        solvers.wp = self; wp.array.numpy = measured_numpy

    def restore(self):
        solvers.wp = self.old_wp; wp.array.numpy = self.old_numpy


class ProfileObserver(CycleObserver):
    def __init__(self, *, device, probe):
        super().__init__(device=device); self.probe = probe

    def __call__(self, *, event, **values):
        if event == 'cycle_start':
            self.probe.in_cycle = True
        super().__call__(event=event, **values)
        if event == 'cycle_end':
            self.probe.in_cycle = False


def profile_cycle(*, baseline, output, device, levels):
    settings = read_json(path=baseline/'frozen_settings.json')['settings']
    case = load_npf662_case(path=baseline/'confined'/'inputs.npz')
    linear, exact, faces = make_linear_problem(case=case, settings=settings['linear_settings'])
    options = dict(settings['kcycle_options'], min_coarse_n=settings['min_coarse_n'], max_cycles=1)
    probe = LaunchReadbackProbe(); observer = ProfileObserver(device=device, probe=probe)
    # Use the existing compiled cache, not another cold compilation campaign.
    wp.load_module(module='DARCY_WARP_PACKAGE.kernels_3d', device=device)
    probe.install()
    try:
        head, info = solvers.solve_multigrid_kcycle_7point_3d(tx_p=faces[0], tx_m=faces[1], ty_p=faces[2],
            ty_m=faces[3], tz_p=faces[4], tz_m=faces[5], rhs=linear.source, active=case.active,
            bc_mask=case.fixed, bc_values=case.values, initial_head=case.initial,
            max_levels=levels, device=device, benchmark_observer=observer, **options)
    finally:
        probe.restore()
    row = dict(levels=info['n_levels'], cycle_seconds=observer.cycle_times,
        explicit_kernel_launches=probe.launches, launch_api_wall_seconds=probe.launch_api_seconds,
        scalar_numpy_readbacks=probe.readbacks, numpy_readback_wait_wall_seconds=probe.readback_seconds,
        caveat='Profiling overhead included; dispatch is host API wall, readback includes outstanding GPU work, synchronization and scalar copies. Not additive estimates of pure overhead. Fills/copies are not counted as wp.launch calls.')
    output.mkdir(parents=True, exist_ok=False)
    write_json(path=output/'profile.json', value=row)
    return row


def check_kcycle_linearity(*, baseline, output, device, levels):
    settings = read_json(path=baseline/'frozen_settings.json')['settings']
    case = load_npf662_case(path=baseline/'confined'/'inputs.npz')
    linear, exact, faces = make_linear_problem(case=case, settings=settings['linear_settings'])
    reference = NPF662Reference(case=case)
    k, j, i = np.indices(dimensions=case.active.shape)
    second = np.sin(2.*np.pi*i/(case.active.shape[2]-1))*np.cos(np.pi*j/(case.active.shape[1]-1))*(1.+k/case.active.shape[0])
    second[case.fixed != 0] = 0.
    flux = reference.c0*(second.ravel()[reference.left]-second.ravel()[reference.right])
    rhs_b = (reference.incidence@flux).reshape(case.active.shape); rhs_b[~reference.free] = 0.
    options = dict(settings['kcycle_options'], min_coarse_n=settings['min_coarse_n'], max_cycles=1)
    heads = {}
    for name, rhs in (('a', linear.source), ('b', rhs_b), ('a_plus_b', linear.source+rhs_b), ('twice_a', 2.*linear.source)):
        head, info = solvers.solve_multigrid_kcycle_7point_3d(tx_p=faces[0], tx_m=faces[1], ty_p=faces[2],
            ty_m=faces[3], tz_p=faces[4], tz_m=faces[5], rhs=rhs, active=case.active,
            bc_mask=case.fixed, bc_values=np.zeros_like(a=case.values), initial_head=np.zeros_like(a=case.initial),
            max_levels=levels, device=device, **options)
        heads[name] = head
    scale = max(np.linalg.norm(heads['a_plus_b']), np.linalg.norm(heads['a'])+np.linalg.norm(heads['b']))
    result = dict(levels=levels, fixed_cycles=1, homogeneous_CHD=True,
        relative_additivity_error=float(np.linalg.norm(heads['a_plus_b']-heads['a']-heads['b'])/scale),
        relative_homogeneity_error=float(np.linalg.norm(heads['twice_a']-2.*heads['a'])/np.linalg.norm(2.*heads['a'])),
        interpretation='Residual-dependent coarse alpha makes this generally nonlinear in the RHS even with fixed sweep counts; flexible GMRES must store preconditioned basis vectors.')
    output.mkdir(parents=True, exist_ok=False)
    save_arrays(directory=output, arrays=heads)
    write_json(path=output/'linearity.json', value=result)
    return result


def measure_resident_runs(*, baseline, output, name, device, repeats, native_directory):
    settings = read_json(path=baseline/'frozen_settings.json')['settings']
    case = load_npf662_case(path=baseline/name/'inputs.npz')
    with np.load(file=native_directory/'result.npz', allow_pickle=False) as saved:
        native_head = saved['head']; native_fields = dict(flux_native=saved['flux_native'])
    nz, ny, nx = case.active.shape
    start = perf_counter()
    model = WarpDarcySolver3D(nx=nx, ny=ny, nz=nz, dx=case.dx, dy=case.dy,
        device=device, solver='kcycle', implementation='fast', formulation='mf6_6_6_2_npf')
    model.build_from_layers(z_interfaces=case.z_interfaces, kx_field=case.kx, ky_field=case.ky,
        kz_field=case.kz, active=case.active, bc_mask=case.fixed, bc_values=case.values,
        rhs=case.source, initial_head=case.initial)
    construction = perf_counter()-start
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    try:
        for index in range(repeats+1):
            np.testing.assert_array_equal(actual=model._session.initial_head, desired=case.initial)
            memory = MemorySampler(**settings['memory_options']); memory.start()
            probe = NewtonProbe(device=device); probe.install()
            wp.synchronize_device(device=device); start = perf_counter()
            try:
                head, info = model.solve(unconfined=bool(np.any(case.cell_types)), npf_cell_type=case.cell_types,
                    **settings['newton_options'])
                wp.synchronize_device(device=device); seconds = perf_counter()-start
            finally:
                probe.restore()
            peak = memory.stop()
            info.get('effective_options', {}).pop('cell_type', None)
            checks, fields = NPF662MatrixFreeReference(case=case).evaluate(head=head)
            directory = output/('warmup' if index == 0 else f'repeat_{index-1}')
            archive_solution(directory=directory/'arrays', case=case, head=head, fields=fields)
            row = dict(accepted=bool(info['converged'] and checks['independent_accepted']),
                solve_only_seconds=info['seconds'], public_solve_wall_seconds=seconds,
                initialization_seconds=probe.workspace_seconds,
                upload_and_packing_seconds=probe.upload_seconds, memory=peak,
                independent=checks, info=info, cold_head_verified=True,
                comparison=compare_solution(case=case, head=head, fields=fields, native_head=native_head, native_fields=native_fields),
                evaluation_history=probe.evaluations)
            write_json(path=directory/'result.json', value=row)
            if index > 0:
                if not info['workspace_reused']:
                    raise AssertionError('Resident timing did not use the existing workspace')
                rows.append(row)
            print(f"Resident {name} run {index}: accepted={row['accepted']}, solver={info['seconds']:.6g} s", flush=True)
    finally:
        model.close()
    result = dict(construction_seconds=construction, repeats=rows,
        timing='Model/workspace/graph reuse; each call resets the exact canonical cold head. Public wall excludes shared construction.')
    write_json(path=output/'resident.json', value=result)
    return result


def write_report(*, baseline, output, result, audits, profiles, linearity, resident):
    lines = ['Measured local results', '======================', '',
        f"Baseline source HEAD: {result['metadata']['revision']}",
        f"Device: {result['metadata']['device']}",
        f"Runtime: {result['metadata']['python'].splitlines()[0]}; Warp {result['metadata']['warp']}; "
        f"NumPy {result['metadata']['numpy']}; SciPy {scipy.__version__}; FloPy {flopy.__version__}.", '',
        'Every successful median below uses three independently accepted runs from the exact cold initial state.',
        'Public wall includes final independent host acceptance; the backend timer excludes it.', '',
        'Variant       MF6 process   warmed solver   warmed public   warmed end-to-end   cold end-to-end + explicit compile',
        '------------- ------------- --------------- --------------- ------------------- -----------------------------------']
    compile_seconds = sum(item['seconds'] for item in result['explicit_module_compilation']
        if item['module'] != 'DARCY_WARP_PACKAGE.kernels_3d')
    for name, record in result['variants'].items():
        successful = [row for row in record['warm_repeats'] if row['accepted']]
        if successful:
            solver = float(np.median([row['solve_only_seconds'] for row in successful]))
            public = float(np.median([row['public_solve_wall_seconds'] for row in successful]))
            end = float(np.median([row['end_to_end_seconds'] for row in successful]))
            cold = record['cold_first']['end_to_end_seconds']+compile_seconds
            lines.append(f"{name:13} {record['mf6']['seconds']:13.4f} {solver:15.4f} {public:15.4f} {end:19.4f} {cold:35.4f}")
        else:
            lines.append(f'{name}: NO successful timing median; all failures retained.')
        audit = audits[name]
        lines.extend(['', f"{name}: native accepted={record['mf6_accepted']}; native physical RMS="
            f"{audit['independent']['physical_rms']:.6g}; relative cell imbalance={audit['independent']['relative_cell_imbalance']:.6g}.",
            f"Recharge {audit['source_total_m3_day']:.9g} m3/day in {audit['source_cells']} top-layer RCH cells; "
            f"native CHD exchange {audit['native_chd_total_m3_day']:.9g} m3/day (injection sign).",
            f"Native full-cell maximum budget residual {audit['native_full_cell_residual_max_m3_day']:.6g} m3/day; "
            f"constitutive/native face-flow mismatch {audit['native_flow_law_max_abs_m3_day']:.6g} m3/day."])
        for index, row in enumerate(record['warm_repeats']):
            info = row['info']; comparison = row['comparison']
            fallback_work = sum(attempt['krylov'] for outer in info['outer_history'] for attempt in outer['attempts'] if attempt['direction'] != 'newton')
            lines.append(f"Repeat {index}: accepted={row['accepted']}, Newton={info['outer_iterations']}, Krylov={info['krylov_iterations']}, "
                f"Jv={info['jacobian_vector_products']}, candidates={info['candidate_evaluations']}, fallback Krylov={fallback_work}; "
                f"head RMSE/max={comparison.get('head_rmse_m')}/{comparison.get('head_max_abs_m')} m; "
                f"face relative L2={comparison.get('flux_relative_l2')}; device sample peak={row['memory']['device_sampled_peak_mib']} MiB.")
    lines.extend(['', 'Standalone full-saturated K-cycle', '----------------------------------', '',
        'These are linear A0 timings, not preconditioned Newton timings.', '',
        'Levels Smoother                    Cycles Accepted Wall_s       Cycle_s      Hierarchy_s',
        '------ --------------------------- ------ -------- ------------ ------------ ------------'])
    for row in result['kcycle']:
        hierarchy = (row['hierarchy_host_seconds'] or 0.)+(row['hierarchy_device_upload_and_init_seconds'] or 0.)
        lines.append(f"{row['info']['n_levels']:6} {row['info']['smoother']:27} {row['info']['n_cycles_used']:6} "
            f"{str(row['accepted']):8} {row['wall_seconds']:12.5f} {(row['cycles_total_seconds'] or 0.):12.5f} {hierarchy:12.5f}")
    accepted = [row for row in result['kcycle'] if row['accepted']]
    accepted.sort(key=kcycle_wall)
    lines.extend(['', 'Candidates for the NEXT integration task:'])
    for row in accepted[:3]:
        lines.append(f"* L{row['info']['n_levels']} {row['info']['smoother']}: {row['wall_seconds']:.5f} s total, "
            f"{row['info']['n_cycles_used']} cycles; single observation, not a repeated performance ranking.")
    if not accepted:
        lines.append('No hierarchy reached the target; no successful configuration can be recommended from these measurements.')
    lines.extend(['', 'Resident workspace/graph reuse (each run still starts from the exact cold head):'])
    for name, record in resident.items():
        successful = [row for row in record['repeats'] if row['accepted']]
        if successful:
            lines.append(f"* {name}: {len(successful)} accepted repeats; median backend solver "
                f"{float(np.median([row['solve_only_seconds'] for row in successful])):.6g} s; "
                f"median public wall {float(np.median([row['public_solve_wall_seconds'] for row in successful])):.6g} s. "
                'Shared construction is outside these per-call times.')
        else:
            lines.append(f'* {name}: no accepted resident repeat; no successful median.')
    lines.extend(['', 'Launch and readback profiling is a separate one-cycle run and excluded from timing medians:'])
    for profile in profiles:
        lines.append(f"* L{profile['levels']}: {profile['explicit_kernel_launches']} explicit wp.launch calls, "
            f"{profile['scalar_numpy_readbacks']} scalar readbacks; launch API wall {profile['launch_api_wall_seconds']:.6g} s, "
            f"readback/wait wall {profile['numpy_readback_wait_wall_seconds']:.6g} s.")
    lines.extend(['', f"One-cycle L{linearity['levels']} additivity error: {linearity['relative_additivity_error']:.6g}; "
        f"homogeneity error: {linearity['relative_homogeneity_error']:.6g}. "
        'A homogeneous map can still be nonlinear: this is evidence against treating this cycle as one fixed linear inverse.'])
    lines.extend(['', 'Evidence and interpretation', '----------------------------', '',
        'The initial CG attempt is retained under benchmark_artifacts/npf662_250x250x5_v1/run_003.',
        'Earlier namespaces preserve setup failures, CG/BICGSTAB diagnostics, excessive ILUT fill setup and superseded source mappings.',
        'The final namespace uses the user-selected 100 m cells and 150 mm/year native RCH recharge.',
        'No failed or superseded run contributes to a successful timing median.',
        'The cold compilation sum above includes the NPF and Warp linear modules; K-cycle compilation is separate. First solve also captures specialized kernels/graphs.',
        'Cold totals exclude interpreter/import/context startup and report-writing. Warm totals still reconstruct the model/workspace.',
        'Memory samples include display/context; pool high-water values are process cumulative. Exact hierarchy allocation bytes are in JSON.',
        'One native solve and one K-cycle observation per configuration limit timing precision; these are first-pass rankings.',
        'See docs/NPF662_250_BASELINE.rst for array signs, equations, acceptance criteria and scaling/reuse/FGMRES integration requirements.', ''])
    (output/'REPORT.rst').write_text(data='\n'.join(lines))


def plot_breakdowns(*, output, result):
    import matplotlib
    matplotlib.use(backend='Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    for name, record in result['variants'].items():
        for index, row in enumerate(record['warm_repeats']):
            history = row['info']['outer_history']
            work = np.cumsum([item['krylov'] for item in history])
            axes[0].semilogy(work, [item['metrics']['physical_rms'] for item in history], label=f'{name} {index}')
    axes[0].set_xlabel('Cumulative Krylov iterations'); axes[0].set_ylabel('Physical RMS [m³/day]'); axes[0].legend(fontsize='small')
    base = [row for row in result['kcycle'] if row['info']['smoother'] == 'chebyshev']
    x = np.arange(len(base))
    host = np.array([row['hierarchy_host_seconds'] for row in base])
    upload = np.array([row['hierarchy_device_upload_and_init_seconds'] for row in base])
    cycle = np.array([row['cycles_total_seconds'] for row in base])
    remainder = np.array([row['wall_seconds'] for row in base])-host-upload-cycle
    axes[1].bar(x=x, height=host, label='Host hierarchy')
    axes[1].bar(x=x, height=upload, bottom=host, label='Device hierarchy/upload/init')
    axes[1].bar(x=x, height=cycle, bottom=host+upload, label='Cycles')
    axes[1].bar(x=x, height=remainder, bottom=host+upload+cycle, label='Checks/final budget/release')
    axes[1].set_xticks(ticks=x, labels=[str(row['info']['n_levels']) for row in base])
    axes[1].set_xlabel('Actual levels'); axes[1].set_ylabel('Wall seconds'); axes[1].legend(fontsize='small')
    for axis in axes:
        axis.grid(visible=True, axis='y')
    fig.tight_layout(); fig.savefig(fname=output/'work_and_time_breakdown.png', dpi=160); plt.close(fig=fig)


def kcycle_wall(row):
    return row['wall_seconds']


def write_combined_csv(*, baseline, output, resident, control, result):
    with (baseline/'summary.csv').open() as stream:
        rows = list(csv.DictReader(f=stream))
    for row in rows:
        if row['backend'] == 'MF6_process':
            name = row['experiment']
            row['seconds'] = result['variants'][name]['mf6']['seconds']
            row['accepted'] = result['variants'][name]['mf6_accepted']
    for name, record in result['variants'].items():
        diagnostic = record['diagnostic_mf6']
        rows.append(dict(experiment=f'{name}_initial_native_unaccepted', backend='MF6_diagnostic',
            accepted=False, seconds=diagnostic['seconds']))
    for name, record in resident.items():
        successful = [row for row in record['repeats'] if row['accepted']]
        for index, row in enumerate(record['repeats']):
            rows.append(dict(experiment=f'{name}_resident_{index}', backend='resident_device_newton_gmres_diagonal',
                accepted=row['accepted'], seconds=row['solve_only_seconds'], public_wall_seconds=row['public_solve_wall_seconds'],
                peak_mib=row['memory']['device_sampled_peak_mib'], iterations=row['info']['krylov_iterations']))
        for metric in ('solve_only_seconds', 'public_solve_wall_seconds'):
            rows.append(dict(experiment=f'{name}_resident_successful_median_{metric}',
                backend='resident_device_newton_gmres_diagonal', accepted=bool(successful),
                seconds=float(np.median([row[metric] for row in successful])) if successful else None))
    rows.append(dict(experiment='observer_disabled_control', backend='standalone_kcycle_linear_A0',
        accepted=control['accepted'], seconds=control['wall_seconds'], iterations=control['info']['n_cycles_used']))
    columns = sorted(set().union(*(row.keys() for row in rows)))
    with (output/'summary.csv').open(mode='w', newline='') as stream:
        writer = csv.DictWriter(f=stream, fieldnames=columns); writer.writeheader(); writer.writerows(rows)


def refine_and_refresh_reference(*, baseline, output, result, native_options, timeout_seconds, head_tolerance, reuse_native):
    directories = {}
    for name, record in result['variants'].items():
        case = load_npf662_case(path=baseline/name/'inputs.npz')
        directory = reuse_native.get(name, output/'native_refined'/name)
        print(f'Refined native RCH reference: {name}', flush=True)
        if name in reuse_native:
            native = read_json(path=directory/'summary.json')
            if not (native['native_success'] and native['independent_conservation'] and native['independent']['independent_accepted']
                    and native['input_hash'] == case.digest() and native['mf6_sha256'] == result['metadata']['mf6_sha256']
                    and native['source_package'] == 'rch'):
                raise ValueError('Reused native reference identity or acceptance mismatch')
            with np.load(file=directory/'result.npz', allow_pickle=False) as saved:
                fields = {key: saved[key].copy() for key in saved.files}
                head = fields['head']
        else:
            head, native, fields = run_reference_mf6(case=case, directory=directory,
                executable=Path(result['metadata']['executable']), source_package='rch',
                head_tolerance_m=head_tolerance, native_solver_options=native_options, timeout_seconds=timeout_seconds)
        if head is None:
            raise RuntimeError(f'Refined native reference failed; retain {directory}')
        record['diagnostic_mf6'] = record['mf6']
        record['mf6'] = native
        record['mf6_accepted'] = bool(native['native_success'] and native['independent_conservation'] and native['independent']['independent_accepted'])
        directories[name] = directory
        reference = NPF662MatrixFreeReference(case=case)
        archive_solution(directory=output/'reference_arrays'/name, case=case, head=head, fields=fields, native=True)
        for label, row in [('cold_first', record['cold_first'])]+[(f'repeat_{index}', row) for index, row in enumerate(record['warm_repeats'])]:
            solved = np.load(file=baseline/name/label/'arrays'/'head.npy', allow_pickle=False)
            checks, evaluated = reference.evaluate(head=solved)
            row['independent_checks'] = checks
            row['accepted'] = bool(row['info']['converged'] and checks['independent_accepted'])
            row['comparison'] = compare_solution(case=case, head=solved, fields=evaluated, native_head=head, native_fields=fields)
        print(f"Refined {name}: native accepted={record['mf6_accepted']}, native seconds={native['seconds']:.6g}", flush=True)
    return directories


def analyze(*, baseline, output, device, profile_levels, native_options, native_timeout, native_head_tolerance, reuse_native):
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(src=Path(__file__), dst=output/Path(__file__).name)
    source_paths = [Path('working_tests/benchmark_npf662_baseline.py'), Path(__file__)]
    source_paths += list(Path('DARCY_WARP_PACKAGE').rglob(pattern='*.py'))
    analysis_source_hashes = snapshot_sources(root=output, paths=source_paths)
    write_json(path=output/'frozen_refinement_settings.json', value=dict(native_options=native_options,
        native_timeout=native_timeout, native_head_tolerance=native_head_tolerance,
        reused_native={name:str(path) for name,path in reuse_native.items()},
        source_sha256=analysis_source_hashes))
    result = read_json(path=baseline/'results.json')
    native_directories = refine_and_refresh_reference(baseline=baseline, output=output, result=result,
        native_options=native_options, timeout_seconds=native_timeout, head_tolerance=native_head_tolerance,
        reuse_native=reuse_native)
    audits = {name: audit_native(baseline=baseline, name=name, output=output/name,
        native_directory=native_directories[name]) for name in result['variants']}
    resident = {name: measure_resident_runs(baseline=baseline, output=output/f'{name}_resident', name=name,
        device=device, repeats=result['settings']['repeats'], native_directory=native_directories[name]) for name in result['variants']}
    profiles = [profile_cycle(baseline=baseline, output=output/f'profile_L{levels}', device=device, levels=levels) for levels in profile_levels]
    linearity = check_kcycle_linearity(baseline=baseline, output=output/'linearity', device=device,
        levels=result['settings']['smoother_comparison_levels'])
    # Prove both variants and failed attempts use identical shared physical arrays.
    cases = [load_npf662_case(path=baseline/name/'inputs.npz') for name in result['variants']]
    for name in cases[0].__dataclass_fields__:
        if name != 'cell_type':
            np.testing.assert_array_equal(actual=getattr(cases[0], name), desired=getattr(cases[1], name))
    earlier_root = Path('benchmark_artifacts/npf662_250x250x5_v1/run_003')
    for name in result['variants']:
        earlier = load_npf662_case(path=earlier_root/name/'inputs.npz')
        current = load_npf662_case(path=baseline/name/'inputs.npz')
        for field in ('z_interfaces', 'kx', 'ky', 'kz', 'active', 'fixed', 'values', 'initial', 'cell_type'):
            np.testing.assert_array_equal(actual=getattr(earlier, field), desired=getattr(current, field))
    shutil.copyfile(src=Path(result['metadata']['executable']), dst=output/'mf6_6.6.2')
    analysis = dict(native_audits=audits, launch_profiles=profiles, linearity=linearity,
        source_sha256=analysis_source_hashes, native_refinement_options=native_options,
        resident_workspace=resident,
        archived_mf6_sha256=digest_file(path=output/'mf6_6.6.2'),
        shared_physical_inputs_verified=True, initial_geometry_material_boundary_head_arrays_preserved=True,
        user_requested_changes=dict(dx_m=100.,dy_m=100.,recharge_mm_year=150.,native_source_package='RCH'),
        flopy=flopy.__version__, scipy=scipy.__version__)
    write_json(path=output/'analysis.json', value=analysis)
    write_json(path=output/'complete_results.json', value=dict(primary=result, analysis=analysis))
    write_combined_csv(baseline=baseline, output=output, resident=resident, control=result['observer_disabled_control'], result=result)
    write_report(baseline=baseline, output=output, result=result, audits=audits, profiles=profiles, linearity=linearity, resident=resident)
    plot_breakdowns(output=output, result=result)
    manifest = {str(path.relative_to(output)): dict(bytes=path.stat().st_size, sha256=digest_file(path=path))
        for path in output.rglob(pattern='*') if path.is_file()}
    write_json(path=output/'manifest.json', value=manifest)


if __name__ == "__main__":
    baseline = Path('benchmark_artifacts/npf662_250x250x5_rch150_dx100_v1/run_001')
    output = Path('benchmark_artifacts/npf662_250x250x5_rch150_dx100_v1/analysis_003')
    device = 'cuda:0'
    profile_levels = [3, 5, 8]
    native_options = dict(outer_maximum=500, inner_maximum=2000, outer_dvclose=1e-6, inner_dvclose=1e-8,
        rcloserecord=[1e-7, 'STRICT'], linear_acceleration='BICGSTAB', preconditioner_levels=8,
        preconditioner_drop_tolerance=1e-4, relaxation_factor=0., reordering_method='RCM',
        number_orthogonalizations=8, scaling_method='DIAGONAL', under_relaxation='NONE')
    reuse_native = dict(confined=Path('benchmark_artifacts/npf662_250x250x5_rch150_dx100_v1/analysis_002/native_refined/confined'))
    native_timeout = 300
    native_head_tolerance = 1e-6
    wp.config.kernel_cache_dir = str((baseline/'kernel_cache').resolve())
    wp.init()
    analyze(baseline=baseline, output=output, device=device, profile_levels=profile_levels,
        native_options=native_options, native_timeout=native_timeout, native_head_tolerance=native_head_tolerance,
        reuse_native=reuse_native)

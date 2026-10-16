"""Local immutable NPF662 baseline. Edit only the explicit configuration below.

Run from the repository root with DARCY_FLOAT=float64 and the darcywarp Python.
No sparse direct solve is used. Each invocation creates a new evidence directory.
"""
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import csv
import gc
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading

import numpy as np
import warp as wp

from DARCY_WARP_PACKAGE.validation.head_checks import finite_diagnostics
from DARCY_WARP_PACKAGE.validation.npf662_reference import (
    NPF662LayeredCase, NPF662Reference, load_npf662_case, save_npf662_case,
)
from DARCY_WARP_PACKAGE.validation.npf662_mf6 import run_reference_mf6
from DARCY_WARP_PACKAGE.validation.npf662_matrix_free import NPF662MatrixFreeReference
from DARCY_WARP_PACKAGE.validation.terrain_case import atomic_json
from DARCY_WARP_PACKAGE.physics.npf662_layers import build_layered_dis_faces
from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D
from DARCY_WARP_PACKAGE.solvers import npf662_newton
from DARCY_WARP_PACKAGE.solvers_3d import (
    solve_multigrid_kcycle_7point_3d, _choose_horizontal_coarsening,
)


def json_value(*, value):
    if isinstance(value, np.ndarray):
        return json_value(value=value.tolist())
    if isinstance(value, dict):
        return {key: json_value(value=item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(value=item) for item in value]
    if isinstance(value, np.generic):
        return json_value(value=value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(*, path, value):
    atomic_json(path=path, value=json_value(value=value))


def command(*, arguments):
    return subprocess.check_output(args=arguments, text=True).strip()


def digest_file(*, path):
    checksum = hashlib.sha256()
    with path.open(mode='rb') as stream:
        for block in iter(stream.read, b''):
            checksum.update(block)
            # read() without a size returns the complete file once.
    return checksum.hexdigest()


def snapshot_sources(*, root, paths):
    destination = root / 'source_snapshot'
    hashes = {}
    for path in paths:
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src=path, dst=target)
        hashes[str(path)] = digest_file(path=path)
    return hashes


def generate_case(*, settings, convertible):
    nz, ny, nx = settings['shape']
    rng = np.random.default_rng(seed=settings['seed'])
    y, x = np.meshgrid(np.linspace(start=0., stop=1., num=ny),
                       np.linspace(start=0., stop=1., num=nx), indexing='ij')
    surface = settings['surface_base'] + settings['surface_rise'] * x
    surface += settings['surface_wave'] * np.sin(2.*np.pi*y) * np.sin(np.pi*x)
    layers = np.arange(nz)[:, None, None]
    thickness = settings['layer_thickness'] + settings['layer_increment']*layers
    thickness = thickness + settings['thickness_wave'] * np.sin(
        2.*np.pi*x[None] + settings['layer_phase']*layers) * np.cos(
        2.*np.pi*y[None] - settings['layer_phase']*layers)
    interfaces = np.concatenate([surface[None], surface[None]-np.cumsum(a=thickness, axis=0)], axis=0)
    pattern = settings['logk_wave']*np.sin(2.*np.pi*x[None]+layers) * np.cos(2.*np.pi*y[None])
    noise = rng.uniform(low=-settings['logk_noise'], high=settings['logk_noise'], size=(nz, ny, nx))
    kx = settings['layer_k'][:, None, None] * np.exp(pattern+noise)
    ky = kx * (settings['ky_ratio'] + settings['ky_wave']*np.sin(2.*np.pi*y[None]+layers))
    kz = kx * (settings['kz_ratio'] + settings['kz_wave']*np.cos(2.*np.pi*x[None]-layers))
    active = np.ones(shape=(nz, ny, nx), dtype=np.int32)
    fixed = np.zeros_like(a=active); fixed[:, :, 0] = 1
    values = np.full(shape=active.shape, fill_value=settings['coast_head'], dtype=np.float64)
    source = np.zeros_like(a=kx)
    source[0, :, 1:] = settings['recharge']*settings['dx']*settings['dy']*(
        1.+settings['recharge_wave']*np.sin(2.*np.pi*y[:, 1:])*np.sin(np.pi*x[:, 1:]))
    initial = settings['coast_head'] + settings['cold_rise']*np.broadcast_to(array=x, shape=active.shape)
    initial[fixed != 0] = values[fixed != 0]
    types = np.zeros_like(a=active)
    types[convertible] = 1
    case = NPF662LayeredCase(kx=kx, ky=ky, kz=kz, active=active, fixed=fixed,
        values=values, source=source, initial=initial, dx=settings['dx'], dy=settings['dy'],
        z_interfaces=interfaces, cell_type=types)
    case.validate()
    return case


def save_arrays(*, directory, arrays):
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in arrays.items():
        np.save(file=directory/f'{name}.npy', arr=value, allow_pickle=False)


def archive_case(*, directory, case):
    directory.mkdir(parents=True, exist_ok=False)
    save_npf662_case(case=case, path=directory/'inputs.npz')
    arrays = {name: getattr(case, name) for name in case.__dataclass_fields__}
    arrays.update(interfaces=case.z_interfaces, top=case.tops, bottom=case.bottoms,
        thickness=case.thicknesses, centres=(case.tops+case.bottoms)/2.,
        cold_initial_head=case.initial, prescribed_values=case.values)
    save_arrays(directory=directory/'arrays', arrays=arrays)
    write_json(path=directory/'identity.json', value=dict(case_sha256=case.digest(),
        inputs_sha256=digest_file(path=directory/'inputs.npz'), shape=case.active.shape,
        convertible_layers=np.flatnonzero(np.any(case.cell_types, axis=(1, 2))).tolist()))


def face_arrays(*, reference, flux):
    result = {}
    for name, axis in (('qx', 2), ('qy', 1), ('qz', 0)):
        shape = list(reference.shape); shape[axis] -= 1
        # Reference face ordering is C order within each axis, x then y then z.
        result[name] = flux[reference.axes == axis].reshape(shape)
    return result


def archive_solution(*, directory, case, head, fields, native=False):
    reference = NPF662MatrixFreeReference(case=case)
    checks, evaluated = reference.evaluate(head=head)
    flux = fields['flux_native'] if native else evaluated['flux']
    divergence = reference.divergence(flux=flux)
    reservoir = np.where(case.fixed != 0, divergence-case.source, 0.)
    physical = case.source+reservoir-divergence
    arrays = dict(head=head, saturation=fields['native_saturation'] if native else evaluated['saturation'],
        cellwise_budget_residual=physical, free_cell_residual=np.where(reference.free, physical, 0.),
        boundary_exchange=reservoir, source_realised=fields['source_realised'] if native else case.source,
        **face_arrays(reference=reference, flux=flux))
    save_arrays(directory=directory, arrays=arrays)
    return checks


class MemorySampler:
    """100 ms device-wide NVML samples via nvidia-smi; pool data is separate."""
    def __init__(self, *, device_index, interval_ms):
        self.device_index = device_index; self.interval_ms = interval_ms
        self.samples = []; self.process = None; self.thread = None

    def collect(self):
        for line in self.process.stdout:
            try:
                self.samples.append(float(line.strip()))
            except ValueError:
                continue

    def start(self):
        self.pool_before = wp.get_mempool_used_mem_current(device=f'cuda:{self.device_index}')
        self.high_before = wp.get_mempool_used_mem_high(device=f'cuda:{self.device_index}')
        self.process = subprocess.Popen(args=['nvidia-smi', f'--id={self.device_index}',
            '--query-gpu=memory.used', '--format=csv,noheader,nounits', '-lms', str(self.interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self.thread = threading.Thread(target=self.collect, daemon=True); self.thread.start()

    def stop(self):
        self.process.terminate(); self.process.wait(); self.thread.join()
        return dict(device_sampled_peak_mib=max(self.samples) if self.samples else None,
            device_first_sample_mib=self.samples[0] if self.samples else None,
            sample_interval_ms=self.interval_ms, sample_count=len(self.samples),
            pool_current_before_bytes=self.pool_before,
            pool_high_before_bytes=self.high_before,
            pool_current_after_bytes=wp.get_mempool_used_mem_current(device=f'cuda:{self.device_index}'),
            pool_process_cumulative_high_bytes=wp.get_mempool_used_mem_high(device=f'cuda:{self.device_index}'))


class NewtonProbe:
    """Passive timings; synchronized uploads include NumPy packing and API work."""
    def __init__(self, *, device):
        self.device = device; self.upload_seconds = 0.; self.evaluations = []
        self.workspace_seconds = 0.; self.origin = perf_counter()

    def __getattr__(self, name):
        return getattr(wp, name)

    def array(self, *args, **kwargs):
        wp.synchronize_device(device=self.device); start = perf_counter()
        result = wp.array(*args, **kwargs)
        wp.synchronize_device(device=self.device)
        self.upload_seconds += perf_counter()-start
        return result

    def install(self):
        self.old_wp = npf662_newton.wp
        self.old_load = npf662_newton.NPF662Workspace.load
        self.old_init = npf662_newton.NPF662Workspace.__init__
        self.old_evaluate = npf662_newton.NPF662Workspace.evaluate
        probe = self

        def measured_load(workspace, **kwargs):
            wp.synchronize_device(device=probe.device); start = perf_counter()
            result = probe.old_load(workspace, **kwargs)
            wp.synchronize_device(device=probe.device)
            probe.upload_seconds += perf_counter()-start
            return result

        def measured_init(workspace, **kwargs):
            start = perf_counter(); result = probe.old_init(workspace, **kwargs)
            wp.synchronize_device(device=probe.device)
            probe.workspace_seconds += perf_counter()-start
            return result

        def measured_evaluate(workspace, **kwargs):
            result = probe.old_evaluate(workspace, **kwargs)
            probe.evaluations.append(dict(wall_seconds=perf_counter()-probe.origin,
                evaluation=len(probe.evaluations)+1, candidate=kwargs.get('trial', False), **result))
            return result

        npf662_newton.wp = self
        npf662_newton.NPF662Workspace.load = measured_load
        npf662_newton.NPF662Workspace.__init__ = measured_init
        npf662_newton.NPF662Workspace.evaluate = measured_evaluate

    def restore(self):
        npf662_newton.wp = self.old_wp
        npf662_newton.NPF662Workspace.load = self.old_load
        npf662_newton.NPF662Workspace.__init__ = self.old_init
        npf662_newton.NPF662Workspace.evaluate = self.old_evaluate


def run_newton(*, case_path, output, device, options, memory_options, native_head, native_fields):
    output.mkdir(parents=True, exist_ok=False)
    memory = MemorySampler(**memory_options); memory.start()
    wp.synchronize_device(device=device); start = perf_counter()
    case = load_npf662_case(path=case_path)
    loaded = perf_counter()
    nz, ny, nx = case.active.shape
    model = WarpDarcySolver3D(nx=nx, ny=ny, nz=nz, dx=case.dx, dy=case.dy,
        device=device, solver='kcycle', implementation='fast', formulation='mf6_6_6_2_npf')
    model.build_from_layers(z_interfaces=case.z_interfaces, kx_field=case.kx, ky_field=case.ky,
        kz_field=case.kz, active=case.active, bc_mask=case.fixed, bc_values=case.values,
        rhs=case.source, initial_head=case.initial)
    wp.synchronize_device(device=device); built = perf_counter()
    probe = NewtonProbe(device=device); probe.install()
    try:
        head, info = model.solve(unconfined=bool(np.any(case.cell_types)), npf_cell_type=case.cell_types, **options)
        wp.synchronize_device(device=device); solved = perf_counter()
    finally:
        probe.restore()
        model.close()
    peak = memory.stop()
    # Do not replicate the 312,500-entry declared type array in every JSON run.
    info.get('effective_options', {}).pop('cell_type', None)
    checks, fields = NPF662MatrixFreeReference(case=case).evaluate(head=head)
    checks = archive_solution(directory=output/'arrays', case=case, head=head, fields=fields)
    comparison = compare_solution(case=case, head=head, fields=fields,
        native_head=native_head, native_fields=native_fields)
    row = dict(kind='device_newton_gmres_diagonal', case_hash=case.digest(),
        accepted=bool(info['converged'] and checks['independent_accepted']),
        load_seconds=loaded-start, construction_seconds=built-loaded,
        workspace_initialization_seconds=probe.workspace_seconds,
        synchronized_host_to_device_and_packing_seconds=probe.upload_seconds,
        public_solve_wall_seconds=solved-built, solve_only_seconds=info['seconds'],
        end_to_end_seconds=solved-start, independent_checks=checks,
        comparison=comparison, memory=peak, info=info,
        initial_state='exact_saved_cold_head', upload_timing='neighbor arrays plus workspace load; includes packing/API/synchronization',
        evaluation_history=probe.evaluations)
    write_json(path=output/'result.json', value=row)
    print(json.dumps(dict(output=str(output), accepted=row['accepted'], seconds=row['solve_only_seconds'],
        outers=info['outer_iterations'], krylov=info['krylov_iterations'], reason=info['stop_reason'])), flush=True)
    gc.collect()
    return row


def compare_solution(*, case, head, fields, native_head, native_fields):
    if native_head is None:
        return dict(available=False, reason='MF6 native reference failed; retained evidence')
    active = case.active != 0
    difference = head[active]-native_head[active]
    qdifference = fields['flux']-native_fields['flux_native']
    axes = NPF662MatrixFreeReference(case=case).axes
    return dict(available=True, head_rmse_m=float(np.sqrt(np.mean(difference**2))),
        head_max_abs_m=float(np.max(np.abs(difference))),
        flux_rmse_m3_day=float(np.sqrt(np.mean(qdifference**2))),
        flux_max_abs_m3_day=float(np.max(np.abs(qdifference))),
        flux_relative_l2=float(np.linalg.norm(qdifference)/max(np.linalg.norm(native_fields['flux_native']), np.finfo(float).tiny)),
        **{f'{axis}_flux_max_abs_m3_day': float(np.max(np.abs(qdifference[axes == number])))
           for axis, number in (('x', 2), ('y', 1), ('z', 0))})


class CycleObserver:
    def __init__(self, *, device):
        self.device = device; self.origin = perf_counter(); self.history = []; self.cycle_times = []
        self.host_hierarchy_seconds = None; self.device_hierarchy_seconds = None

    def __call__(self, *, event, **values):
        if event == 'host_hierarchy':
            self.host_hierarchy_seconds = perf_counter()-self.origin
            self.host_bytes = sum(value.nbytes for level in values['levels'] for value in level.values() if isinstance(value, np.ndarray))
        elif event == 'device_hierarchy':
            wp.synchronize_device(device=self.device)
            self.device_hierarchy_seconds = perf_counter()-self.origin-self.host_hierarchy_seconds
            seen = set(); self.device_bytes = 0
            for level in values['levels']:
                for value in level.values():
                    if isinstance(value, wp.array) and value.ptr not in seen:
                        seen.add(value.ptr); self.device_bytes += value.capacity
        elif event == 'cycle_start':
            wp.synchronize_device(device=self.device); self.cycle_start = perf_counter()
        elif event == 'cycle_end':
            wp.synchronize_device(device=self.device); self.cycle_times.append(perf_counter()-self.cycle_start)
        elif event in ('initial_residual', 'residual'):
            self.history.append(dict(cycle=values.get('cycle', 0), residual_rms=values['residual_rms'],
                wall_seconds=perf_counter()-self.origin,
                cycle_seconds=sum(self.cycle_times)))


def hierarchy_signature(*, shape, max_levels, min_coarse_n):
    levels = [tuple(shape)]
    for unused in range(1, max_levels):
        nz, ny, nx = levels[-1]
        fy, fx = _choose_horizontal_coarsening(ny=ny, nx=nx, min_coarse_n=min_coarse_n)
        if (fy, fx) == (1, 1):
            break
        levels.append((nz, (ny+fy-1)//fy, (nx+fx-1)//fx))
    return tuple(levels)


def make_linear_problem(*, case, settings):
    reference = NPF662Reference(case=case)
    nz, ny, nx = case.active.shape
    k, j, i = np.indices(dimensions=case.active.shape)
    x = i/(nx-1); y = j/(ny-1)
    exact = case.values + settings['amplitude']*np.sin(settings['x_frequency']*np.pi*x/2.)*(
        1.+settings['y_amplitude']*np.cos(settings['y_frequency']*np.pi*y))*(1.+settings['layer_amplitude']*k/max(nz-1, 1))
    exact[case.fixed != 0] = case.values[case.fixed != 0]
    flux = reference.c0*(exact.ravel()[reference.left]-exact.ravel()[reference.right])
    rhs = (reference.incidence @ flux).reshape(case.active.shape)
    rhs[~reference.free] = 0.
    linear_case = replace(case, source=rhs)
    faces, unused = build_layered_dis_faces(z_interfaces=case.z_interfaces, kx=case.kx, ky=case.ky,
        kz=case.kz, active=case.active, dx=case.dx, dy=case.dy)
    # Independent exact conductance agreement, before benchmarking.
    for face, axis in zip((faces[0], faces[2], faces[4]), (2, 1, 0)):
        selected = reference.axes == axis
        if not np.allclose(a=face.ravel()[reference.left[selected]], b=reference.c0[selected], rtol=1e-14, atol=0.):
            raise AssertionError('Production A0 differs from independently built full-thickness conductances')
    return linear_case, exact, faces


def run_kcycle(*, case, exact, faces, output, device, options, requested_levels, memory_options, observe=True):
    output.mkdir(parents=True, exist_ok=False)
    memory = MemorySampler(**memory_options); memory.start()
    observer = CycleObserver(device=device)
    start = perf_counter()
    head, info = solve_multigrid_kcycle_7point_3d(tx_p=faces[0], tx_m=faces[1], ty_p=faces[2],
        ty_m=faces[3], tz_p=faces[4], tz_m=faces[5], rhs=case.source, active=case.active,
        bc_mask=case.fixed, bc_values=case.values, initial_head=case.initial,
        max_levels=requested_levels, device=device, benchmark_observer=observer if observe else None, **options)
    wp.synchronize_device(device=device); elapsed = perf_counter()-start
    peak = memory.stop()
    checks, fields = NPF662MatrixFreeReference(case=case).evaluate(head=head)
    difference = head[case.active != 0]-exact[case.active != 0]
    row = dict(kind='standalone_kcycle_linear_A0', accepted=bool(info['converged'] and checks['independent_accepted']),
        requested_levels=requested_levels, options=options, info=info, independent_checks=checks,
        wall_seconds=elapsed, hierarchy_host_seconds=observer.host_hierarchy_seconds,
        hierarchy_device_upload_and_init_seconds=observer.device_hierarchy_seconds,
        hierarchy_host_array_bytes=getattr(observer, 'host_bytes', None),
        hierarchy_device_array_bytes=getattr(observer, 'device_bytes', None),
        individual_cycle_seconds=observer.cycle_times, cycles_total_seconds=sum(observer.cycle_times) if observe else None,
        history=observer.history, memory=peak, observer_enabled=observe,
        exact_head_rmse_m=float(np.sqrt(np.mean(difference**2))), exact_head_max_abs_m=float(np.max(np.abs(difference))),
        residual_reduction_per_cycle=[observer.history[n]['residual_rms']/max(observer.history[n-1]['residual_rms'], np.finfo(float).tiny)
            for n in range(1, len(observer.history))])
    np.save(file=output/'head.npy', arr=head)
    write_json(path=output/'result.json', value=row)
    print(json.dumps(dict(output=str(output), accepted=row['accepted'], levels=info['n_levels'],
        seconds=elapsed, cycles=info['n_cycles_used'], residual=info['r_rms_end'])), flush=True)
    gc.collect()
    return row


def compile_modules(*, device, modules):
    results = []
    for module in modules:
        wp.synchronize_device(device=device); start = perf_counter()
        wp.load_module(module=module, device=device)
        wp.synchronize_device(device=device)
        results.append(dict(module=module, seconds=perf_counter()-start))
    return results


def successful_median(*, rows, field):
    selected = [row[field] for row in rows if row['accepted']]
    return float(np.median(a=selected)) if selected else None


def summarize(*, root, result):
    rows = []
    for variant, record in result['variants'].items():
        native = record['mf6']
        rows.append(dict(experiment=variant, backend='MF6_process', accepted=record['mf6_accepted'],
            seconds=native['seconds'], construction_seconds=None, peak_mib=None, iterations=None))
        for label in ('cold_first', 'warm_repeats'):
            runs = record[label] if label == 'warm_repeats' else [record[label]]
            for index, run in enumerate(runs):
                rows.append(dict(experiment=f'{variant}_{label}_{index}', backend=run['kind'], accepted=run['accepted'],
                    seconds=run['solve_only_seconds'], end_to_end_seconds=run['end_to_end_seconds'],
                    public_wall_seconds=run['public_solve_wall_seconds'], construction_seconds=run['construction_seconds'],
                    peak_mib=run['memory']['device_sampled_peak_mib'], iterations=run['info']['krylov_iterations']))
        for metric in ('solve_only_seconds', 'end_to_end_seconds', 'public_solve_wall_seconds'):
            rows.append(dict(experiment=f'{variant}_successful_median_{metric}', backend='device_newton_gmres_diagonal',
                accepted=any(run['accepted'] for run in record['warm_repeats']),
                seconds=successful_median(rows=record['warm_repeats'], field=metric)))
    for index, run in enumerate(result['kcycle']):
        rows.append(dict(experiment=f"A0_{run['info']['smoother']}_L{run['info']['n_levels']}_{index}",
            backend=run['kind'], accepted=run['accepted'], seconds=run['wall_seconds'],
            cycle_seconds=run['cycles_total_seconds'], construction_seconds=(run['hierarchy_host_seconds'] or 0.)+(
                run['hierarchy_device_upload_and_init_seconds'] or 0.),
            peak_mib=run['memory']['device_sampled_peak_mib'], iterations=run['info']['n_cycles_used']))
    columns = sorted(set().union(*(row.keys() for row in rows)))
    with (root/'summary.csv').open(mode='w', newline='') as stream:
        writer = csv.DictWriter(f=stream, fieldnames=columns); writer.writeheader(); writer.writerows(rows)
    write_json(path=root/'results.json', value=result)
    plot_results(root=root, result=result)


def plot_results(*, root, result):
    import matplotlib
    matplotlib.use(backend='Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    for variant, record in result['variants'].items():
        for index, row in enumerate(record['warm_repeats']):
            history = row['evaluation_history']
            if history:
                axes[0].semilogy([item['evaluation'] for item in history], [item['physical_rms'] for item in history], label=f'{variant} repeat {index}')
                axes[1].semilogy([item['wall_seconds'] for item in history], [item['physical_rms'] for item in history], label=f'{variant} repeat {index}')
    axes[0].set_xlabel('Physical residual evaluations (including candidates)'); axes[1].set_xlabel('Seconds within public solve')
    for axis in axes:
        axis.set_ylabel('Physical residual RMS [m³/day]'); axis.legend(fontsize='small'); axis.grid(visible=True)
    fig.tight_layout(); fig.savefig(fname=root/'newton_residual_history.png', dpi=160); plt.close(fig=fig)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    for row in result['kcycle']:
        history = row['history']
        label = f"L{row['info']['n_levels']} {row['info']['smoother']}"
        if history:
            axes[0].semilogy([item['cycle'] for item in history], [item['residual_rms'] for item in history], label=label)
            axes[1].semilogy([item['wall_seconds'] for item in history], [item['residual_rms'] for item in history], label=label)
    axes[0].set_xlabel('K cycles'); axes[1].set_xlabel('Seconds including hierarchy')
    for axis in axes:
        axis.set_ylabel('Linear residual RMS [m³/day]'); axis.legend(fontsize='small'); axis.grid(visible=True)
    fig.tight_layout(); fig.savefig(fname=root/'kcycle_residual_history.png', dpi=160); plt.close(fig=fig)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
    for row in result['kcycle']:
        label = row['info']['smoother']+(' accepted' if row['accepted'] else ' FAILED target')
        level = row['info']['n_levels']; marker = 'o' if row['accepted'] else 'x'
        axes[0].scatter([level], [row['wall_seconds']], label=label, marker=marker)
        axes[1].scatter([level], [(row['hierarchy_host_seconds'] or 0.)+(row['hierarchy_device_upload_and_init_seconds'] or 0.)], marker=marker)
        axes[2].scatter([level], [row['memory']['device_sampled_peak_mib'] or 0.], marker=marker)
    for axis in axes:
        axis.set_xlabel('Actual hierarchy levels'); axis.grid(visible=True)
    axes[0].set_ylabel('Total seconds (failures are budget time)'); axes[0].legend(fontsize='x-small')
    axes[1].set_ylabel('Hierarchy construction + upload [s]'); axes[2].set_ylabel('Sampled device peak [MiB]')
    fig.tight_layout(); fig.savefig(fname=root/'kcycle_levels_time_memory.png', dpi=160); plt.close(fig=fig)


def run_benchmark(*, root, geology, variants, executable, native_options, native_timeout, native_head_tolerance,
                  device, newton_options, repeats, memory_options, compilation_modules, linear_settings,
                  level_counts, min_coarse_n, kcycle_options, smoother_comparison_levels, source_paths, native_source_package):
    root.mkdir(parents=True, exist_ok=False)
    wp.init()
    source_hashes = snapshot_sources(root=root, paths=source_paths)
    metadata = dict(revision=command(arguments=['git', 'rev-parse', 'HEAD']),
        status=command(arguments=['git', 'status', '--short', '--untracked-files=no']),
        source_sha256=source_hashes, interpreter=sys.executable, python=sys.version, platform=platform.platform(),
        cpu=command(arguments=['lscpu']), gpu=command(arguments=['nvidia-smi', '-q']),
        compute_processes=command(arguments=['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader']),
        warp=wp.__version__, numpy=np.__version__, precision=os.environ.get('DARCY_FLOAT'),
        executable=str(executable.resolve()), mf6_version=command(arguments=[str(executable), '-v']),
        mf6_sha256=digest_file(path=executable), device=wp.get_device(ident=device).name,
        runtime_environment={name: os.environ.get(name) for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'CUDA_VISIBLE_DEVICES')})
    settings = dict(geology=geology, variants=variants, native_options=native_options, native_timeout=native_timeout,
        native_head_tolerance=native_head_tolerance, native_source_package=native_source_package,
        newton_options=newton_options, repeats=repeats,
        memory_options=memory_options, linear_settings=linear_settings, level_counts=level_counts,
        min_coarse_n=min_coarse_n, kcycle_options=kcycle_options, smoother_comparison_levels=smoother_comparison_levels)
    # This file is committed to disk BEFORE any solve and is never updated.
    write_json(path=root/'frozen_settings.json', value=dict(metadata=metadata, settings=settings))
    result = dict(metadata=metadata, settings=settings, variants={}, kcycle=[], equivalent_hierarchies=[])
    for name, convertible in variants.items():
        case = generate_case(settings=geology, convertible=convertible)
        archive_case(directory=root/name, case=case)
    result['explicit_module_compilation'] = compile_modules(device=device, modules=compilation_modules)
    for name in variants:
        directory = root/name
        case = load_npf662_case(path=directory/'inputs.npz')
        print(f'Running pinned MF6: {name}', flush=True)
        start = perf_counter()
        native_head, native, fields = run_reference_mf6(case=case, directory=directory/'mf6', executable=executable,
            head_tolerance_m=native_head_tolerance, native_solver_options=native_options,
            timeout_seconds=native_timeout,source_package=native_source_package)
        native['write_run_decode_wall_seconds'] = perf_counter()-start
        native.get('settings', {}).pop('icelltype', None)
        native_accepted = bool(native['native_success'] and native.get('independent_conservation') and native.get('independent', {}).get('independent_accepted'))
        if native_head is not None:
            archive_solution(directory=directory/'mf6_arrays', case=case, head=native_head, fields=fields, native=True)
        write_json(path=directory/'mf6_benchmark.json', value=native)
        record = dict(mf6=native, mf6_accepted=native_accepted, warm_repeats=[])
        result['variants'][name] = record
        record['cold_first'] = run_newton(case_path=directory/'inputs.npz', output=directory/'cold_first', device=device,
            options=newton_options, memory_options=memory_options, native_head=native_head, native_fields=fields)
        for repeat in range(repeats):
            record['warm_repeats'].append(run_newton(case_path=directory/'inputs.npz', output=directory/f'repeat_{repeat}',
                device=device, options=newton_options, memory_options=memory_options, native_head=native_head, native_fields=fields))
            write_json(path=root/'checkpoint.json', value=result)
    case = load_npf662_case(path=root/'confined'/'inputs.npz')
    linear_case, exact, faces = make_linear_problem(case=case, settings=linear_settings)
    archive_case(directory=root/'linear_A0', case=linear_case)
    np.save(file=root/'linear_A0'/'manufactured_head.npy', arr=exact)
    options = dict(kcycle_options, min_coarse_n=min_coarse_n)
    seen = set()
    for requested in level_counts:
        signature = hierarchy_signature(shape=case.active.shape, max_levels=requested, min_coarse_n=min_coarse_n)
        if signature in seen:
            result['equivalent_hierarchies'].append(dict(requested=requested, signature=signature)); continue
        seen.add(signature)
        row = run_kcycle(case=linear_case, exact=exact, faces=faces, output=root/'linear_A0'/f'levels_{requested}',
            device=device, options=options, requested_levels=requested, memory_options=memory_options)
        result['kcycle'].append(row)
        write_json(path=root/'checkpoint.json', value=result)
    for smoother in ('vertical_line', 'chebyshev_vertical_line'):
        row = run_kcycle(case=linear_case, exact=exact, faces=faces, output=root/'linear_A0'/smoother,
            device=device, options=dict(options, smoother=smoother), requested_levels=smoother_comparison_levels, memory_options=memory_options)
        result['kcycle'].append(row)
        write_json(path=root/'checkpoint.json', value=result)
    # The original algorithm with the observer disabled is the numerical/timing control.
    result['observer_disabled_control'] = run_kcycle(case=linear_case, exact=exact, faces=faces,
        output=root/'linear_A0'/'observer_disabled_control', device=device, options=options,
        requested_levels=smoother_comparison_levels, memory_options=memory_options, observe=False)
    summarize(root=root, result=result)
    manifest = {}
    for path in sorted(root.rglob(pattern='*')):
        if path.is_file() and 'kernel_cache' not in path.parts:
            manifest[str(path.relative_to(root))] = dict(bytes=path.stat().st_size, sha256=digest_file(path=path))
    write_json(path=root/'manifest.json', value=manifest)
    return result


if __name__ == "__main__":
    root = Path('benchmark_artifacts/npf662_250x250x5_rch150_dx100_v1/run_001')
    executable = Path('/bin/modflow/mf6')
    device = 'cuda:0'
    # Physical inputs are selected once, before MF6 or DarcyWarp is run.
    recharge_annual_mm = 150.
    days_per_year = 365.
    geology = dict(shape=(5, 250, 250), seed=6622505, dx=100., dy=100.,
        surface_base=8., surface_rise=12., surface_wave=1.,
        layer_thickness=24., layer_increment=3., thickness_wave=2., layer_phase=.7,
        layer_k=np.array([12., 8., 16., 6., 10.]), logk_wave=.45, logk_noise=.25,
        ky_ratio=.8, ky_wave=.1, kz_ratio=.15, kz_wave=.025,
        coast_head=0., recharge=recharge_annual_mm/1000./days_per_year,
        recharge_annual_mm=recharge_annual_mm, days_per_year=days_per_year,
        recharge_wave=.2, cold_rise=2.)
    variants = dict(confined=[], unconfined=[0])
    native_source_package = 'rch'
    native_options = dict(outer_maximum=500, inner_maximum=2000,
        linear_acceleration='BICGSTAB',
        outer_dvclose=1e-6, inner_dvclose=1e-8,
        rcloserecord=[1e-9, 'STRICT'], preconditioner_levels=5,
        preconditioner_drop_tolerance=1e-4, under_relaxation='DBD')
    native_timeout = 1800
    native_head_tolerance = 1e-6
    newton_options = dict(max_outer=80, max_cycles=4000, krylov_restart=48, return_info=True)
    repeats = 3
    memory_options = dict(device_index=0, interval_ms=100)
    compilation_modules = ['DARCY_WARP_PACKAGE.solvers.npf662_kernels', 'warp.optim.linear',
        'DARCY_WARP_PACKAGE.kernels_3d']
    linear_settings = dict(amplitude=3., x_frequency=1., y_amplitude=.2, y_frequency=2., layer_amplitude=.1)
    level_counts = [3, 4, 5, 6, 7, 8]
    min_coarse_n = 2
    kcycle_options = dict(max_cycles=100, nu_pre=2, nu_post=2, nu_coarse=10, smoother='chebyshev',
        omega=.8, line_omega=.8, line_sweeps_pre=1, line_sweeps_post=1, line_sweeps_coarse=1,
        cheby_lambda_min=.05, cheby_lambda_max=1.95, rel_tol=0., abs_tol_min=1e-10,
        dh_rms_tol=None, dh_max_tol=None, check_every_no=1, diag_preconditioner_backend='device', return_info=True)
    smoother_comparison_levels = 5
    source_paths = [Path('working_tests/benchmark_npf662_baseline.py')]
    source_paths += list(Path('DARCY_WARP_PACKAGE').rglob(pattern='*.py'))
    # Force a private empty cache so reported module compilation is reproducible.
    cache_directory = root/'kernel_cache'
    wp.config.kernel_cache_dir = str(cache_directory.resolve())
    if os.environ.get('DARCY_FLOAT') != 'float64':
        raise RuntimeError('Launch with DARCY_FLOAT=float64 before importing WarpDarcy')
    results = run_benchmark(root=root, geology=geology, variants=variants, executable=executable,
        native_options=native_options, native_timeout=native_timeout, native_head_tolerance=native_head_tolerance,
        device=device, newton_options=newton_options, repeats=repeats, memory_options=memory_options,
        compilation_modules=compilation_modules, linear_settings=linear_settings, level_counts=level_counts,
        min_coarse_n=min_coarse_n, kcycle_options=kcycle_options,
        smoother_comparison_levels=smoother_comparison_levels, source_paths=source_paths,
        native_source_package=native_source_package)

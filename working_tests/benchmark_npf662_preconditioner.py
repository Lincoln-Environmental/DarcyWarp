"""Compare the implemented Newton preconditioners using immutable saved arrays."""
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import csv
import gc
import hashlib
import json
import platform
import subprocess

import numpy as np
import warp as wp

from DARCY_WARP_PACKAGE.solvers.npf662_kcycle import NPF662KCycleControls
from DARCY_WARP_PACKAGE.validation.npf662_reference import load_npf662_case
from DARCY_WARP_PACKAGE.validation.npf662_matrix_free import NPF662MatrixFreeReference
from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D
from benchmark_npf662_baseline import MemorySampler, write_json


def build_model(*, case, device):
    nz, ny, nx = case.active.shape
    model = WarpDarcySolver3D(nx=nx, ny=ny, nz=nz, dx=case.dx, dy=case.dy,
        device=device, solver='kcycle', implementation='fast', formulation='mf6_6_6_2_npf')
    model.build_from_layers(z_interfaces=case.z_interfaces, kx_field=case.kx,
        ky_field=case.ky, kz_field=case.kz, active=case.active, bc_mask=case.fixed,
        bc_values=case.values, rhs=case.source, initial_head=case.initial)
    return model


def measure(*, baseline, variant, label, device, options, controls, phase, repeat, resident=None):
    gc.collect()
    memory = MemorySampler(device_index=int(device.split(':')[-1]), interval_ms=100)
    memory.start()
    wp.synchronize_device(device=device)
    start = perf_counter()
    case = load_npf662_case(path=baseline / variant / 'inputs.npz')
    loaded = perf_counter()
    model = build_model(case=case, device=device) if resident is None else resident
    model.update_initial_head(initial_head=case.initial)
    wp.synchronize_device(device=device)
    built = perf_counter()
    kwargs = dict(npf_preconditioner='diagonal' if label == 'diagonal_fgmres' else label,
                  unconfined=bool(np.any(case.cell_types)),
                  npf_cell_type=case.cell_types, **options)
    if label == 'diagonal_fgmres':kwargs['npf_krylov_method'] = 'fgmres'
    if label == 'kcycle':kwargs['npf_kcycle_controls'] = controls
    try:
        head, info = model.solve(**kwargs)
        wp.synchronize_device(device=device)
        stopped = perf_counter()
    finally:
        peak = memory.stop()
        if resident is None:model.close()
    info.get('effective_options', {}).pop('cell_type', None)
    reference = NPF662MatrixFreeReference(case=case)
    checks, fields = reference.evaluate(head=head)
    native_head = np.load(file=baseline / variant / 'mf6' / 'head.npy', allow_pickle=False)
    native_flux = np.concatenate([np.load(file=baseline / variant / 'mf6' / f'{name}.npy', allow_pickle=False).ravel()
                                  for name in ('qx', 'qy', 'qz')])
    delta = head[reference.free] - native_head[reference.free]
    difference = fields['flux'] - native_flux
    row = dict(variant=variant, preconditioner=label, phase=phase, repeat=repeat,
        accepted=bool(info['converged'] and checks['independent_accepted']),
        load_seconds=loaded-start, construction_seconds=built-loaded,
        end_to_end_seconds=stopped-start, public_solve_seconds=stopped-built,
        solve_only_seconds=info['seconds'], memory=peak, info=info, physical=checks,
        comparison=dict(head_rmse_m=float(np.sqrt(np.mean(delta**2))), head_max_m=float(np.max(np.abs(delta))),
                        flux_relative_l2=float(np.linalg.norm(difference)/np.linalg.norm(native_flux)),
                        flux_max_m3_day=float(np.max(np.abs(difference)))))
    print(f"{variant} {label} {phase} {repeat}: accepted={row['accepted']} "
          f"e2e={row['end_to_end_seconds']:.4f}s solve={row['solve_only_seconds']:.4f}s "
          f"Newton={info['outer_iterations']} Krylov={info['krylov_iterations']}", flush=True)
    return row


def save_results(*, output, results):
    write_json(path=output / 'summary.json', value=results)
    rows = []
    for row in results['runs']:
        mg = row['info'].get('multigrid', {})
        rows.append(dict(variant=row['variant'], preconditioner=row['preconditioner'], phase=row['phase'],
            repeat=row['repeat'], accepted=row['accepted'], end_to_end_seconds=row['end_to_end_seconds'],
            solve_only_seconds=row['solve_only_seconds'], public_solve_seconds=row['public_solve_seconds'],
            construction_seconds=row['construction_seconds'], hierarchy_seconds=mg.get('construction_seconds', 0.),
            mg_device_seconds=mg.get('device_apply_seconds', 0.), sync_seconds=row['info'].get('krylov_synchronization_seconds'),
            mg_enqueue_seconds=mg.get('preconditioner_enqueue_seconds', 0.),
            newton=row['info']['outer_iterations'], krylov=row['info']['krylov_iterations'],
            restarts=row['info'].get('krylov_restarts'), applications=row['info'].get('preconditioner_applications'),
            cycles=row['info'].get('preconditioner_cycles'),
            hierarchy_reused=mg.get('hierarchy_reused'), peak_mib=row['memory']['device_sampled_peak_mib'],
            pool_live_delta_bytes=row['memory']['pool_current_after_bytes']-row['memory']['pool_current_before_bytes'],
            head_rmse_m=row['comparison']['head_rmse_m'], flux_relative_l2=row['comparison']['flux_relative_l2']))
    if rows:
        with (output / 'summary.csv').open(mode='w', newline='') as stream:
            writer = csv.DictWriter(f=stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(rows)
    histories = [dict(variant=row['variant'], preconditioner=row['preconditioner'], phase=row['phase'], repeat=row['repeat'],
                      nonlinear=row['info']['outer_history'], linear=row['info'].get('linear_history')) for row in results['runs']]
    write_json(path=output / 'histories.json', value=histories)
    lines = ['Observed 3D Newton preconditioner comparison', '===========================================', '',
             'Saved 250x250x5 inputs and MF6 arrays were read unchanged. No model regeneration or native solve is part of this driver.',
             'Fresh model timings include NPZ load, construction, hierarchy/workspaces, solve and public independent acceptance.',
             'Warmup is excluded; successful medians require all three accepted repetitions. Resident runs also reset the cold head.', '',
             'Multigrid device events measure scaling + graph + output copy; scalar-read wait time overlaps that GPU work.',
             'These costs must not be added as disjoint runtime fractions. Hierarchy time includes graph setup.', '']
    for variant in results['variants']:
        medians = {}
        for label in results['labels']:
            selected = [row for row in results['runs'] if row['variant'] == variant and row['preconditioner'] == label
                        and row['phase'] == 'timed' and row['accepted']]
            if len(selected) != results['repeats']:
                lines.append(f'{variant} {label}: no successful median ({len(selected)} accepted repetitions).')
                continue
            medians[label] = float(np.median([row['end_to_end_seconds'] for row in selected]))
            row = selected[0]; mg = row['info'].get('multigrid', {})
            lines.append(f"{variant} {label}: median end-to-end {medians[label]:.6f}s; "
                         f"Newton {[r['info']['outer_iterations'] for r in selected]}; "
                         f"Krylov {[r['info']['krylov_iterations'] for r in selected]}; "
                         f"restarts {[r['info'].get('krylov_restarts') for r in selected]}.")
            if mg:
                for key in ('construction_seconds', 'device_apply_seconds', 'synchronization_seconds', 'preconditioner_enqueue_seconds'):
                    value = float(np.median([r['info']['multigrid'][key] for r in selected]))
                    lines.append(f'  {key}: {value:.6f}s ({100*value/medians[label]:.2f}% of end-to-end; overlapping categories).')
        if 'diagonal' in medians and 'kcycle' in medians:
            lines.append(f"{variant} end-to-end speedup: {medians['diagonal']/medians['kcycle']:.3f}x.")
        if 'diagonal_fgmres' in medians and 'kcycle' in medians:
            lines.append(f"{variant} K-cycle versus diagonal FGMRES speedup: {medians['diagonal_fgmres']/medians['kcycle']:.3f}x.")
    (output / 'REPORT.rst').write_text(data='\n'.join(lines) + '\n')


def benchmark(*, baseline, output, variants, labels, device, options, controls, repeats, resident_repeats):
    output.mkdir(parents=True, exist_ok=False)
    wp.init()
    source_paths = [Path(__file__), Path('DARCY_WARP_PACKAGE/solvers/npf662_newton.py'),
                    Path('DARCY_WARP_PACKAGE/solvers/npf662_3d.py'), Path('DARCY_WARP_PACKAGE/solvers/npf662_kcycle.py'),
                    Path('DARCY_WARP_PACKAGE/solvers/npf662_fgmres.py')]
    baseline_hashes = {str(path.relative_to(baseline)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in [baseline / variant / 'inputs.npz' for variant in variants]}
    results = dict(baseline=str(baseline), baseline_input_sha256=baseline_hashes, variants=variants, labels=labels,
        repeats=repeats, device=device, options=options, multigrid_controls=asdict(controls),
        revision=subprocess.check_output(args=['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
        python=platform.python_version(), warp=wp.__version__, hardware=subprocess.check_output(
            args=['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'], text=True).strip(), runs=[])
    for variant in variants:
        for label in labels:
            row = measure(baseline=baseline, variant=variant, label=label, device=device, options=options,
                          controls=controls, phase='warmup', repeat=0)
            results['runs'].append(row); save_results(output=output, results=results)
            if not row['accepted']:continue
            for repeat in range(repeats):
                row = measure(baseline=baseline, variant=variant, label=label, device=device, options=options,
                              controls=controls, phase='timed', repeat=repeat)
                results['runs'].append(row); save_results(output=output, results=results)
            if label == 'kcycle' and resident_repeats:
                case = load_npf662_case(path=baseline / variant / 'inputs.npz')
                with build_model(case=case, device=device) as model:
                    for repeat in range(resident_repeats + 1):
                        row = measure(baseline=baseline, variant=variant, label=label, device=device, options=options,
                            controls=controls, phase='resident_warmup' if repeat == 0 else 'resident', repeat=repeat,
                            resident=model)
                        results['runs'].append(row); save_results(output=output, results=results)
    for relative, digest in baseline_hashes.items():
        assert hashlib.sha256((baseline / relative).read_bytes()).hexdigest() == digest


if __name__ == "__main__":
    baseline = Path('baselines/npf662_250x250x5_rch150_dx100_v1')
    output = Path('benchmark_results/npf662_stabilization_001')
    variants = ('confined', 'unconfined')
    labels = ('kcycle', 'diagonal', 'diagonal_fgmres')
    device = 'cuda:0'
    options = dict(max_outer=80, max_cycles=4000, krylov_restart=48, return_info=True)
    controls = NPF662KCycleControls(levels=5, pre_sweeps=2, post_sweeps=2, coarse_sweeps=20, cycles=1, omega=.8)
    repeats = 3
    resident_repeats = 3
    benchmark(baseline=baseline, output=output, variants=variants, labels=labels, device=device,
              options=options, controls=controls, repeats=repeats, resident_repeats=resident_repeats)

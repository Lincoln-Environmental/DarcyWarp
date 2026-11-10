"""Repeated, matched 3D NPF scaling comparisons with live pinned MF6.

Run from the repository root. --dry-run requires only NumPy. Timing runs
require the darcywarp environment and the existing pinned MF6 6.6.2 binary.
"""
import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter

import numpy as np


def suite_settings(*, sizes, variants):
    """Describe the cases without importing Warp or allocating grid arrays.

    :param sizes: Horizontal cell counts for square grids.
    :param variants: Confined, convertible or phreatic recharge cases.
    """
    cases = []
    for size in sizes:
        if size < 2:
            raise ValueError('Grid sizes must be at least two cells')
        for variant in variants:
            recharge_mm = 5. if variant == 'phreatic' else 150.
            if variant not in ('confined', 'convertible', 'phreatic'):
                raise ValueError(f'Unknown variant: {variant}')
            geology = dict(shape=[5, size, size], seed=6622505,
                dx=25000./size, dy=25000./size,
                surface_base=8., surface_rise=12., surface_wave=1.,
                layer_thickness=24., layer_increment=3., thickness_wave=2., layer_phase=.7,
                layer_k=[12., 8., 16., 6., 10.], logk_wave=.45, logk_noise=.25,
                ky_ratio=.8, ky_wave=.1, kz_ratio=.15, kz_wave=.025,
                coast_head=0., recharge=recharge_mm/1000./365.,
                recharge_annual_mm=recharge_mm, days_per_year=365.,
                recharge_wave=.2, cold_rise=2.)
            cases.append(dict(case_id=f'{variant}_{size}x{size}x5', variant=variant,
                geology=geology, convertible=[] if variant == 'confined' else [0]))
    return cases


def parity_metrics(*, head, flux, native_head, native_flux, free, gates):
    """Evaluate head and physical face-flow differences on matched arrays.

    :param head: DarcyWarp head array in canonical layer, y, x order.
    :param flux: Independently evaluated DarcyWarp physical face flows.
    :param native_head: Decoded native MF6 heads.
    :param native_flux: Decoded native MF6 FLOW-JA-FACE flows.
    :param free: Boolean free-cell mask.
    :param gates: Maximum head RMSE, head difference and relative flux limits.
    """
    delta = np.asarray(head)[free]-np.asarray(native_head)[free]
    difference = np.asarray(flux)-np.asarray(native_flux)
    head_rmse = float(np.sqrt(np.mean(delta**2)))
    head_max = float(np.max(np.abs(delta)))
    flux_relative = float(np.linalg.norm(difference)/max(np.linalg.norm(native_flux), np.finfo(float).tiny))
    return dict(head_rmse_m=head_rmse, head_max_m=head_max,
        flux_relative_l2=flux_relative,
        passed=bool(np.isfinite([head_rmse, head_max, flux_relative]).all()
            and head_rmse <= gates['head_rmse_m'] and head_max <= gates['head_max_m']
            and flux_relative <= gates['flux_relative_l2']))


def comparison_summary(*, rows, case_id, preconditioner, repeats):
    """Require every requested timed repeat before calculating any speedup.

    :param rows: Completed measurement records, including failed runs.
    :param case_id: Case identity to summarize.
    :param preconditioner: DarcyWarp preconditioner selection.
    :param repeats: Required sample count per engine and phase.
    """
    selected = [row for row in rows if row['case_id'] == case_id and row['phase'] == 'timed']
    native = [row for row in selected if row['engine'] == 'mf6']
    warp = [row for row in selected if row['engine'] == 'darcywarp' and row['preconditioner'] == preconditioner]
    accepted = bool(len(native) == repeats and len(warp) == repeats
        and all(row['accepted'] for row in native+warp))
    result = dict(case_id=case_id, preconditioner=preconditioner, accepted=accepted,
        mf6_samples=len(native), darcywarp_samples=len(warp),
        mf6_accepted=sum(row['accepted'] for row in native),
        darcywarp_accepted=sum(row['accepted'] for row in warp),
        mf6_process_median_s=None, mf6_end_to_end_median_s=None,
        darcywarp_public_median_s=None, darcywarp_end_to_end_median_s=None,
        process_vs_public_speedup=None, end_to_end_speedup=None)
    if accepted:
        result.update(mf6_process_median_s=float(np.median([row['process_seconds'] for row in native])),
            mf6_end_to_end_median_s=float(np.median([row['end_to_end_seconds'] for row in native])),
            darcywarp_public_median_s=float(np.median([row['public_solve_seconds'] for row in warp])),
            darcywarp_end_to_end_median_s=float(np.median([row['end_to_end_seconds'] for row in warp])))
        result['process_vs_public_speedup'] = result['mf6_process_median_s']/result['darcywarp_public_median_s']
        result['end_to_end_speedup'] = result['mf6_end_to_end_median_s']/result['darcywarp_end_to_end_median_s']
    return result


def save_progress(*, root, result):
    """Checkpoint full measurements and a compact comparison table.

    :param root: New evidence directory for this invocation.
    :param result: Frozen settings and completed measurements.
    """
    from DARCY_WARP_PACKAGE.validation.json_output import atomic_json
    from DARCY_WARP_PACKAGE.validation.head_checks import finite_diagnostics
    result['summary'] = [comparison_summary(rows=result['runs'], case_id=case['case_id'],
        preconditioner=label, repeats=result['repeats'])
        for case in result['cases'] for label in result['preconditioners']]
    atomic_json(path=root/'results.json', value=finite_diagnostics(value=result))
    with (root/'summary.csv').open(mode='w', newline='') as stream:
        writer = csv.DictWriter(f=stream, fieldnames=list(result['summary'][0]))
        writer.writeheader()
        writer.writerows(result['summary'])


def main():
    """Run the benchmark CLI, or print its allocation-free case plan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tier', choices=['smoke', 'standard', 'scale'], default='standard')
    parser.add_argument('--sizes', type=int, nargs='+')
    parser.add_argument('--variants', choices=['confined', 'convertible', 'phreatic'], nargs='+',
        default=['confined', 'convertible', 'phreatic'])
    parser.add_argument('--preconditioners', choices=['kcycle', 'diagonal', 'diagonal_fgmres'],
        nargs='+', default=['kcycle'])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--resident-repeats', type=int, default=3)
    parser.add_argument('--device', default='cuda:0')
    default_mf6 = Path('/bin/modflow/mf6')
    if not default_mf6.is_file():
        default_mf6 = Path(shutil.which('mf6') or '/bin/modflow/mf6')
    parser.add_argument('--mf6', type=Path, default=default_mf6)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--mf6-timeout', type=float, default=1800.)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.repeats < 3 or args.resident_repeats < 0 or not np.isfinite(args.mf6_timeout) or args.mf6_timeout <= 0.:
        parser.error('Use at least three timed repeats, nonnegative resident repeats and a positive timeout')
    sizes = list(dict.fromkeys(args.sizes or dict(smoke=[16, 32], standard=[50, 100, 250, 500], scale=[1000])[args.tier]))
    variants = list(dict.fromkeys(args.variants))
    labels = list(dict.fromkeys(args.preconditioners))
    try:
        cases = suite_settings(sizes=sizes, variants=variants)
    except ValueError as error:
        parser.error(str(error))
    gates = dict(head_rmse_m=1e-3, head_max_m=5e-3, flux_relative_l2=1e-6)
    native_options = dict(outer_maximum=500, inner_maximum=2000, linear_acceleration='BICGSTAB',
        outer_dvclose=1e-6, inner_dvclose=1e-8, rcloserecord=[1e-7, 'STRICT'],
        preconditioner_levels=8, preconditioner_drop_tolerance=1e-4,
        relaxation_factor=0., reordering_method='RCM', number_orthogonalizations=0,
        scaling_method='DIAGONAL', under_relaxation='DBD')
    options = dict(max_outer=80, max_cycles=4000, krylov_restart=48, return_info=True)
    result = dict(schema_version=1, cases=cases, repeats=args.repeats, resident_repeats=args.resident_repeats,
        preconditioners=labels, gates=gates, native_options=native_options,
        newton_options=options, device=args.device, mf6_timeout_seconds=args.mf6_timeout,
        timing='Warmed kernels, identical saved cold head per solve. MF6 fresh process each time. '
            'End-to-end starts from canonical arrays in RAM, includes construction/write, solve and public decode/audit; '
            'excludes input generation, imports/context startup and duplicate benchmark parity checks.', runs=[])
    if args.dry_run:
        print(json.dumps(result, indent=2))
        return 0
    if os.environ.get('DARCY_FLOAT') != 'float64':
        parser.error('Launch with DARCY_FLOAT=float64 before importing the solver')
    if not args.mf6.is_file():
        parser.error(f'MF6 binary does not exist: {args.mf6}')
    import warp as wp
    from DARCY_WARP_PACKAGE.validation.npf662_benchmark_cases import generate_case
    from DARCY_WARP_PACKAGE.validation.npf662_reference import save_npf662_case
    from DARCY_WARP_PACKAGE.validation.npf662_matrix_free import NPF662MatrixFreeReference
    from DARCY_WARP_PACKAGE.validation.npf662_mf6 import run_reference_mf6
    from DARCY_WARP_PACKAGE.validation.json_output import atomic_json
    from DARCY_WARP_PACKAGE.validation.head_checks import finite_diagnostics
    from DARCY_WARP_PACKAGE.solvers.npf662_kcycle import NPF662KCycleControls
    from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D
    wp.init()
    device = wp.get_device(ident=args.device)
    controls = NPF662KCycleControls(use_cuda_graph=device.is_cuda)
    root = args.output or Path('benchmark_results').joinpath(
        'npf662_vs_mf6_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    root.mkdir(parents=True, exist_ok=False)
    source_root = Path(__file__).resolve().parents[1]
    source_paths = [Path(__file__).resolve()]+list(source_root.joinpath('DARCY_WARP_PACKAGE').rglob('*.py'))
    source_hashes = {}
    for source in source_paths:
        relative = source.relative_to(source_root)
        target = root.joinpath('source_snapshot', relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        source_hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    result['metadata'] = dict(revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source_root, text=True).strip(),
        source_sha256=source_hashes, python=platform.python_version(), platform=platform.platform(),
        cpu=platform.processor(), logical_cpus=os.cpu_count(), warp=wp.__version__, numpy=np.__version__,
        device=device.name, precision=os.environ['DARCY_FLOAT'], interpreter=sys.executable,
        mf6=str(args.mf6.resolve()), mf6_sha256=hashlib.sha256(args.mf6.read_bytes()).hexdigest(),
        runtime_environment={key: os.environ.get(key) for key in
            ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'CUDA_VISIBLE_DEVICES']})
    if device.is_cuda and shutil.which('nvidia-smi'):
        hardware = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
            '--format=csv,noheader'], capture_output=True, text=True, check=False)
        result['metadata']['nvidia_smi'] = hardware.stdout.strip()
    result['multigrid_controls'] = asdict(controls)
    atomic_json(path=root/'frozen_settings.json', value=result)
    for specification in cases:
        directory = root/specification['case_id']
        directory.mkdir()
        geology = dict(specification['geology'], layer_k=np.asarray(specification['geology']['layer_k']))
        case = generate_case(settings=geology, convertible=specification['convertible'])
        save_npf662_case(case=case, path=directory/'inputs.npz')
        case_hash = case.digest()
        reference = NPF662MatrixFreeReference(case=case)
        native_head = None
        native_flux = None
        for repeat in range(args.repeats+1):
            phase = 'warmup' if repeat == 0 else 'timed'
            folder = directory/f'mf6_{phase}_{repeat}'
            ims = dict(native_options)
            if specification['convertible']:
                ims.update(under_relaxation='NONE', number_orthogonalizations=8)
            start = perf_counter()
            try:
                head, info, fields = run_reference_mf6(case=case, directory=folder, executable=args.mf6.resolve(),
                    head_tolerance_m=1e-6, native_solver_options=ims,
                    timeout_seconds=args.mf6_timeout, source_package='rch')
                elapsed = perf_counter()-start
                accepted = bool(info['native_success'] and info.get('independent_conservation')
                    and info.get('independent', {}).get('independent_accepted'))
                if accepted and native_head is None:
                    native_head = head.copy()
                    native_flux = fields['flux_native'].copy()
                elif accepted:
                    parity = parity_metrics(head=head, flux=fields['flux_native'], native_head=native_head,
                        native_flux=native_flux, free=reference.free, gates=gates)
                    info['repeat_parity'] = parity
                    accepted = accepted and parity['passed']
                info.get('settings', {}).pop('icelltype', None)
                row = dict(case_id=specification['case_id'], case_sha256=case_hash, engine='mf6',
                    preconditioner='IMS', phase=phase, repeat=repeat, accepted=accepted,
                    process_seconds=info['seconds'], end_to_end_seconds=elapsed, info=info)
            except Exception as error:
                row = dict(case_id=specification['case_id'], engine='mf6', preconditioner='IMS',
                    phase=phase, repeat=repeat, accepted=False, error=f'{type(error).__name__}: {error}')
            result['runs'].append(row)
            save_progress(root=root, result=result)
            print(f"{specification['case_id']} MF6 {phase} {repeat}: accepted={row['accepted']}", flush=True)
        if native_head is None:
            continue
        top_free = reference.free[0]
        result.setdefault('regimes', {})[specification['case_id']] = dict(
            free_top_cells=int(np.count_nonzero(top_free)),
            above_top_fraction=float(np.mean(native_head[0][top_free] >= case.tops[0][top_free])),
            below_bottom_fraction=float(np.mean(native_head[0][top_free] < case.bottoms[0][top_free])))
        for label in labels:
            model = None
            phases = [('warmup', 0)]+[('timed', i) for i in range(1, args.repeats+1)]
            if args.resident_repeats:
                phases += [('resident_warmup', 0)]+[('resident', i) for i in range(1, args.resident_repeats+1)]
            for phase, repeat in phases:
                resident = phase in ('resident_warmup', 'resident')
                try:
                    wp.synchronize_device(device=device)
                    pool_before = wp.get_mempool_used_mem_current(device=device) if device.is_cuda else None
                    start = perf_counter()
                    if model is None:
                        nz, ny, nx = case.active.shape
                        model = WarpDarcySolver3D(nx=nx, ny=ny, nz=nz, dx=case.dx, dy=case.dy,
                            device=args.device, solver='kcycle', implementation='fast', formulation='mf6_6_6_2_npf')
                        model.build_from_layers(z_interfaces=case.z_interfaces, kx_field=case.kx, ky_field=case.ky,
                            kz_field=case.kz, active=case.active, bc_mask=case.fixed, bc_values=case.values,
                            rhs=case.source, initial_head=case.initial)
                    model.update_initial_head(initial_head=case.initial)
                    wp.synchronize_device(device=device)
                    built = perf_counter()
                    solve_options = dict(options, unconfined=bool(specification['convertible']),
                        npf_cell_type=case.cell_types, npf_preconditioner='diagonal' if label == 'diagonal_fgmres' else label)
                    if label == 'kcycle':
                        solve_options['npf_kcycle_controls'] = controls
                    elif label == 'diagonal_fgmres':
                        solve_options['npf_krylov_method'] = 'fgmres'
                    head, info = model.solve(**solve_options)
                    wp.synchronize_device(device=device)
                    stopped = perf_counter()
                    info.get('effective_options', {}).pop('cell_type', None)
                    checks, fields = reference.evaluate(head=head)
                    parity = parity_metrics(head=head, flux=fields['flux'], native_head=native_head,
                        native_flux=native_flux, free=reference.free, gates=gates)
                    expected_krylov = 'GMRES' if label == 'diagonal' else 'FGMRES'
                    method_passed = bool(info.get('npf_preconditioner') == solve_options['npf_preconditioner']
                        and info.get('krylov_method') == expected_krylov)
                    if phase == 'resident':
                        method_passed = method_passed and bool(info.get('workspace_reused'))
                        if label == 'kcycle':
                            method_passed = method_passed and bool(info.get('multigrid', {}).get('hierarchy_reused'))
                    row = dict(case_id=specification['case_id'], case_sha256=case_hash, engine='darcywarp',
                        preconditioner=label, phase=phase, repeat=repeat,
                        accepted=bool(info['converged'] and checks['independent_accepted'] and parity['passed'] and method_passed),
                        requested_method_passed=method_passed, construction_reset_seconds=built-start,
                        public_solve_seconds=stopped-built, backend_seconds=info['seconds'],
                        end_to_end_seconds=stopped-start, comparison=parity, physical=checks, info=info,
                        pool_current_before_bytes=pool_before,
                        pool_current_after_bytes=wp.get_mempool_used_mem_current(device=device) if device.is_cuda else None,
                        pool_process_cumulative_high_bytes=wp.get_mempool_used_mem_high(device=device) if device.is_cuda else None)
                    np.savez_compressed(file=directory/f'{label}_{phase}_{repeat}_head.npz', head=head)
                except Exception as error:
                    row = dict(case_id=specification['case_id'], engine='darcywarp', preconditioner=label,
                        phase=phase, repeat=repeat, accepted=False, error=f'{type(error).__name__}: {error}')
                    if model is not None:
                        model.close()
                        model = None
                finally:
                    if model is not None and (not resident or phase == 'resident' and repeat == args.resident_repeats):
                        model.close()
                        model = None
                result['runs'].append(row)
                save_progress(root=root, result=result)
                print(f"{specification['case_id']} {label} {phase} {repeat}: accepted={row['accepted']}", flush=True)
        if case.digest() != case_hash:
            raise AssertionError('Canonical case mutated during benchmark')
    save_progress(root=root, result=result)
    result['passed'] = bool(all(row['accepted'] for row in result['runs'])
        and all(row['accepted'] for row in result['summary']))
    save_progress(root=root, result=result)
    print(f"Evidence: {root.resolve()}; all gates passed={result['passed']}", flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

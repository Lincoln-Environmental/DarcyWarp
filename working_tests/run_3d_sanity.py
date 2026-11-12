"""Portable Cartesian/NPF convergence and MF6 sanity checks with cached references."""
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import subprocess
import threading

import numpy as np

from sanity_3d_controls import read_profiles


def save_json(*, path, value):
    from DARCY_WARP_PACKAGE.validation.head_checks import finite_diagnostics
    path.write_text(data=json.dumps(obj=finite_diagnostics(value=value), indent=2)+'\n')


def parse_arguments(*, defaults):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', nargs='+', type=int, default=defaults['sizes'])
    parser.add_argument('--families', nargs='+', choices=['cartesian', 'npf'], default=defaults['families'])
    parser.add_argument('--modes', nargs='+', choices=['steady', 'transient'], default=defaults['modes'])
    parser.add_argument('--variants', nargs='+', choices=['confined', 'convertible', 'phreatic'], default=defaults['variants'])
    parser.add_argument('--device', default=defaults['device'])
    parser.add_argument('--repeats', type=int, default=defaults['repeats'])
    parser.add_argument('--periods', type=int, default=defaults['periods'])
    parser.add_argument('--output', type=Path, default=defaults['output'])
    parser.add_argument('--cache', type=Path, default=defaults['cache'])
    parser.add_argument('--mf6', type=Path, default=defaults['mf6'])
    parser.add_argument('--cached-only', action='store_true')
    parser.add_argument('--compare', action='store_true', help='Compare classic/fast and diagonal/K-cycle; otherwise use the recommended path.')
    parser.add_argument('--ghb', action='store_true', help='Add native fixed-conductance GHB boundaries and audit MF6 GHB flows.')
    parser.add_argument('--tight', action='store_true', help='Use explicit tighter FP64 qualification controls; defaults remain the 2D sanity profiles.')
    parser.add_argument('--gpu-tight', action='store_true', help='Tighten GPU controls while retaining the exact 2D MF6 sanity settings for timing comparisons.')
    parser.add_argument('--cartesian-fgmres', action='store_true', help='Use FP64 FGMRES with the shared line K-cycle for confined Cartesian solves.')
    parser.add_argument('--npf-preconditioner', choices=['auto', 'diagonal', 'kcycle'], default=defaults['preconditioner'])
    parser.add_argument('--npf-relative-tolerance', type=float, default=None,
        help='Explicit NPF cellwise imbalance limit; omitted retains the 1e-8 default.')
    parser.add_argument('--npf-device-native', action='store_true',
        help='Opt-in transient device acceptance; independent verification remains outside the solve timer.')
    parser.add_argument('--npf-controls', type=Path,
        help='JSON containing solve, newton and multigrid controls for a reproducible NPF experiment.')
    parser.add_argument('--npf-history', action='store_true',
        help='Time the complete device history API for transient NPF, including source updates and output download.')
    parser.add_argument('--legacy-max-cycles', type=int, default=None, help='Explicit override; default uses the 2D sanity budget.')
    args = parser.parse_args()
    if min(args.sizes) < 2 or args.repeats < 1 or args.periods < 1:
        parser.error('Sizes must be >=2; repeats and periods must be positive.')
    return args


def sanity_reference_valid(*, info, mode, gates):
    if mode == 'steady':
        checks = info.get('independent', {})
        return bool(info['native_success'] and info.get('ghb_law_valid',True) and checks.get('input_valid')
            and checks.get('boundary_condition_valid') and checks.get('physical_metrics_valid')
            and checks['relative_cell_imbalance'] <= gates['budget_percent']/100.)
    return bool(info.get('native_success',True) and info['periods']) and all(row.get('ghb_law_valid',True) and row['source_allocation_valid'] and row['physical']['input_valid']
        and row['physical']['boundary_condition_valid'] and row['physical']['physical_metrics_valid']
        and row['native_relative_cell_imbalance'] <= gates['budget_percent']/100. for row in info['periods'])


def native_elapsed_seconds(*, text):
    line = next((line for line in text.splitlines() if 'Elapsed run time:' in line), '')
    units = dict(Day=86400., Hour=3600., Minute=60., Second=1.)
    values = re.findall(pattern=r'([0-9.]+)\s+(Day|Hour|Minute|Second)s?', string=line)
    return sum(float(value)*units[unit] for value, unit in values) if values else None


def reference_progress(*, stop, interval, description):
    started = perf_counter()
    while not stop.wait(timeout=interval):
        print(f'  {description}: MF6/reference processing active, {perf_counter()-started:.0f}s elapsed', flush=True)


def prepare_reference_with_progress(*, description, interval, **options):
    stop = threading.Event()
    worker = threading.Thread(target=reference_progress,
        kwargs=dict(stop=stop, interval=interval, description=description), daemon=True)
    worker.start()
    try:
        return prepare_reference(**options)
    finally:
        stop.set()
        worker.join()


def prepare_reference(*, case, family, mode, cache, executable, cached_only, native_options, factors, dt, ss, sy, gates):
    from DARCY_WARP_PACKAGE.validation.npf662_mf6 import run_reference_mf6
    from DARCY_WARP_PACKAGE.validation.npf662_transient_mf6 import run_transient_mf6
    from DARCY_WARP_PACKAGE.validation.npf662_reference import save_npf662_case
    payload = dict(case_sha256=case.digest(), mode=mode, native_options=native_options,
                   factors=factors.tolist(), dt=dt, ss=ss, sy=sy, family=family,
                   acceptance='NPF fixed physical contract' if family == 'npf' else gates)
    fingerprint = hashlib.sha256(json.dumps(obj=payload, sort_keys=True).encode()).hexdigest()
    folder = cache/fingerprint
    manifest_path = folder/'manifest.json'
    if manifest_path.is_file():
        manifest = json.loads(s=manifest_path.read_text())
        if manifest['fingerprint'] != fingerprint:
            raise ValueError('Reference fingerprint mismatch')
        for name, digest in manifest['sha256'].items():
            if hashlib.sha256((folder/name).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Corrupt reference cache: {folder/name}')
        with np.load(file=folder/'reference.npz', allow_pickle=False) as saved:
            arrays = {name: saved[name].copy() for name in saved.files}
        accepted = sanity_reference_valid(info=manifest['native'], mode=mode, gates=gates)
        seconds = manifest.get('native_single_run_seconds', manifest['native'].get('seconds'))
        if seconds is None:
            seconds = native_elapsed_seconds(text=(folder/'native/stdout.log').read_text())
        return arrays, dict(manifest, reused=True, accepted=accepted, native_single_run_seconds=seconds,
            acceptance_criteria='2D sanity validity and 1% cellwise imbalance; strict NPF audit retained separately',
            strict_native_npf_audit=manifest['native'].get('passed', manifest['native'].get('independent_conservation')))
    if cached_only:
        raise FileNotFoundError(f'No validated cached reference: {folder}. Run once without --cached-only.')
    if not executable.is_file():
        raise FileNotFoundError('MF6 is required on cache miss; supply --mf6 /path/to/mf6 (version 6.6.2).')
    if folder.exists():
        raise ValueError(f'Incomplete cache {folder}; select a new --cache directory.')
    folder.mkdir(parents=True)
    save_npf662_case(case=case, path=folder/'inputs.npz')
    started = perf_counter()
    if mode == 'steady':
        head, info, fields = run_reference_mf6(case=case, directory=folder/'native', executable=executable,
            native_solver_options=native_options, replace_native_solver_options=True,
            head_tolerance_m=native_options['outer_dvclose'], source_package='rch')
        if head is None:
            raise RuntimeError(f'MF6 failed: {folder}/native')
        arrays = dict(heads=head[None], fluxes=fields['flux_native'][None],ghb_inflows=fields['ghb_native_inflow'][None])
        accepted = bool(info['native_success'] and info['independent_conservation'])
        if family == 'cartesian':
            checks = info['independent']
            accepted = bool(info['native_success'] and checks['boundary_condition_valid']
                and checks['physical_metrics_valid'] and checks['relative_cell_imbalance'] <= gates['budget_percent']/100.)
    else:
        checksum = hashlib.sha256(executable.read_bytes()).hexdigest()
        arrays, info = run_transient_mf6(case=case, directory=folder/'native', executable=executable,
            source_factors=factors, dt=dt, ss=ss, sy=sy, executable_sha256=checksum,
            source_package='rch', native_solver_options=native_options)
        accepted = bool(info['passed'])
        if family == 'cartesian':
            accepted = all(row['source_allocation_valid'] and row['physical']['boundary_condition_valid']
                and row['physical']['physical_metrics_valid']
                and row['native_relative_cell_imbalance'] <= gates['budget_percent']/100. for row in info['periods'])
    elapsed = perf_counter()-started
    strict_native_npf_audit = accepted
    accepted = sanity_reference_valid(info=info, mode=mode, gates=gates)
    native_seconds = info.get('seconds')
    if mode == 'transient':
        native_seconds = info.get('engine_seconds',native_elapsed_seconds(text=(folder/'native/stdout.log').read_text()))
    np.savez_compressed(file=folder/'reference.npz', **arrays)
    manifest = dict(fingerprint=fingerprint, inputs=payload, accepted=accepted,
        strict_native_npf_audit=strict_native_npf_audit,
        acceptance_criteria='2D sanity validity and 1% cellwise imbalance; strict NPF audit retained separately',
        native=info, reference_generation_end_to_end_s=elapsed, reused=False,
        native_single_run_seconds=native_seconds,
        sha256={name: hashlib.sha256((folder/name).read_bytes()).hexdigest()
                for name in ('inputs.npz', 'reference.npz')})
    save_json(path=manifest_path, value=manifest)
    return arrays, manifest


def measure(*, case, family, mode, label, reference, native_accepted, device, profiles,
            factors, dt, ss, sy, npf_options, kcycle_controls, gates, progress_every, npf_history=False):
    import warp as wp
    from DARCY_WARP_PACKAGE import WarpDarcySolver3D
    from DARCY_WARP_PACKAGE.validation.npf662_matrix_free import NPF662MatrixFreeReference
    from DARCY_WARP_PACKAGE.validation.npf662_transient_reference import NPF662TransientReference
    nz, ny, nx = case.active.shape
    free = (case.active != 0)&(case.fixed == 0)
    previous = case.initial.copy()
    implementation = label if family == 'cartesian' else 'fast'
    kwargs = dict(profiles['transient'] if mode == 'transient' else profiles['steady'][implementation]) if family == 'cartesian' else dict(npf_options, npf_preconditioner=label)
    if mode == 'steady':
        kwargs.pop('npf_device_native',None)
    if family == 'npf' and isinstance(kwargs.get('npf_newton_controls'), dict):
        from DARCY_WARP_PACKAGE.solvers.npf662_newton import NPF662NewtonControls
        kwargs['npf_newton_controls'] = NPF662NewtonControls(**kwargs['npf_newton_controls'])
    if family == 'npf' and label == 'kcycle':
        kwargs['npf_kcycle_controls'] = kcycle_controls
    if family == 'cartesian' and isinstance(kwargs.get('linear_kcycle_controls'),dict):
        from DARCY_WARP_PACKAGE.solvers.npf662_kcycle import NPF662KCycleControls
        kwargs['linear_kcycle_controls']=NPF662KCycleControls(**kwargs['linear_kcycle_controls'])
    wp.synchronize_device(device=device)
    started = perf_counter()
    history, heads = [], []
    with WarpDarcySolver3D(nx=nx, ny=ny, nz=nz, dx=case.dx, dy=case.dy,
            dz=float(case.thicknesses[0, 0, 0]), device=device, implementation=implementation,
            formulation='legacy' if family == 'cartesian' else 'mf6_6_6_2_npf') as model:
        fields = dict(kx_field=case.kx, ky_field=case.ky, kz_field=case.kz, active=case.active,
            bc_mask=case.fixed, bc_values=case.values, rhs=case.source, initial_head=case.initial,
            ghb_conductance=case.ghb_conductance,ghb_head=case.ghb_head)
        if family == 'cartesian':
            model.build_from_K_fields(**fields)
        else:
            model.build_from_layers(z_interfaces=case.z_interfaces, **fields)
        construction_seconds = perf_counter()-started
        schedule = factors if mode == 'transient' else np.ones(shape=1)
        driver_seconds = None
        driver_info = None
        use_history = npf_history and family == 'npf' and mode == 'transient'
        if use_history:
            driver_options = dict(kwargs, unconfined=bool(np.any(case.cell_types)),
                npf_cell_type=case.cell_types, npf_device_native=True)
            solve_started = perf_counter()
            driver_heads, driver_info = model.solve_transient_npf(source_factors=schedule, dt=dt,
                head_prev=previous, ss=ss, sy=sy, **driver_options)
            wp.synchronize_device(device=device)
            driver_seconds = perf_counter()-solve_started
        returned_schedule = schedule[:len(driver_heads)] if use_history else schedule
        for period, factor in enumerate(returned_schedule):
            step = replace(case, source=case.source*factor)
            old = previous.copy()
            if use_history:
                head, info = driver_heads[period], driver_info['periods'][period]
                public_solve_seconds = info['public_step_seconds']
            else:
                model.update_rhs(rhs=step.source)
                model.update_initial_head(initial_head=previous)
                options = dict(kwargs)
                if family == 'npf':
                    options.update(unconfined=bool(np.any(case.cell_types)), npf_cell_type=case.cell_types)
                if mode == 'transient':
                    options.update(transient=True, dt=dt, head_prev=previous)
                    options.update(dict(storage_coeff=ss) if family == 'cartesian' else dict(ss=ss, sy=sy))
                solve_started = perf_counter()
                head, info = model.solve(**options)
                wp.synchronize_device(device=device)
                public_solve_seconds = perf_counter()-solve_started
            audit_started = perf_counter()
            np.testing.assert_array_equal(actual=previous, desired=old)
            auditor = NPF662TransientReference(case=step, previous=previous, dt=dt, ss=ss,
                sy=sy if family == 'npf' else 0.) if mode == 'transient' else NPF662MatrixFreeReference(case=step)
            tolerances = dict(relative_tolerance=kwargs['npf_final_relative_tolerance']) if family == 'npf' and 'npf_final_relative_tolerance' in kwargs else {}
            if family == 'npf' and kwargs.get('npf_roundoff_acceptance',False):
                tolerances['roundoff_acceptance']=True
            if family == 'cartesian':
                tolerances = dict(physical_tolerance=kwargs['abs_tol_min'],
                relative_tolerance=kwargs.get('final_budget_relative_tol',gates['budget_percent']/100.),
                head_tolerance_m=kwargs['dh_rms_tol'] if kwargs['dh_rms_tol'] is not None else gates.get('head_residual_rms',1e-6))
            checks, physical = auditor.evaluate(head=head, **tolerances)
            delta = head[free]-reference['heads'][period][free]
            rmse, maximum = float(np.sqrt(np.mean(delta**2))), float(np.max(np.abs(delta)))
            relative_flux = float(np.linalg.norm(physical['flux']-reference['fluxes'][period])/
                max(np.linalg.norm(reference['fluxes'][period]), np.finfo(float).tiny))
            ghb_error = 0.
            ghb_valid = True
            if case.ghb_conductance is not None:
                actual = -physical['ghb_outflow']
                native = reference['ghb_inflows'][period]
                ghb_error = float(np.max(np.abs(actual-native)))
                # Propagate the measured head difference through the known
                # linear boundary law; allow native C*H-C*h arithmetic roundoff.
                ghb_bound = case.ghb_conductance*np.abs(head-reference['heads'][period])
                ghb_bound += 32.*np.finfo(float).eps*case.ghb_conductance*(np.abs(head)+np.abs(case.ghb_head))
                ghb_valid = bool(np.all(np.isfinite(actual)) and np.all(np.abs(actual-native)<=np.maximum(ghb_bound,1e-14)))
            accepted = bool(info['converged'] and checks['independent_accepted'] and native_accepted
                            and ghb_valid and rmse <= gates['rmse'] and maximum <= gates['maximum'])
            history.append(dict(period=period+1, accepted=accepted, solver_converged=bool(info['converged']),
                head_rmse_m=rmse, head_max_m=maximum, flux_relative_l2=relative_flux,
                ghb_flux_max_error=ghb_error,ghb_law_parity=ghb_valid,
                head_error_in_scaled_eps=maximum/(np.finfo(float).eps*max(1.,float(np.max(np.abs(reference['heads'][period][free]))))),
                physical=checks, cycles=info.get('n_cycles_used'), krylov=info.get('krylov_iterations'),
                outer_iterations=info.get('outer_iterations'), residual_rms=info.get('r_rms_end'),
                device_native=info.get('device_native', False),
                krylov_scalar_readbacks=info.get('krylov_scalar_readbacks'),
                krylov_synchronization_seconds=info.get('krylov_synchronization_seconds'),
                independent_audit_performed=info.get('independent_audit_performed', True),
                state_reused=info.get('reference_reused', False),
                reference_setup_seconds=info.get('reference_setup_seconds'),
                workspace_setup_seconds=info.get('setup_seconds'),
                hierarchy_setup_seconds=info.get('multigrid', {}).get('construction_seconds'),
                hierarchy_reused=info.get('multigrid', {}).get('hierarchy_reused'),
                coefficients_refreshed=info.get('multigrid', {}).get('coefficients_refreshed'),
                outer_history=info.get('outer_history'), final_thresholds=info.get('final_acceptance_thresholds')))
            history[-1].update(public_solve_seconds=public_solve_seconds,
                independent_audit_seconds=perf_counter()-audit_started, backend_seconds=info.get('seconds'))
            heads.append(head.copy())
            if not (info['converged'] and checks['independent_accepted']):
                break
            previous = head.copy()
            if (period+1)%progress_every == 0 or period+1 == len(schedule):
                print(f'    {family} {label}: step {period+1}/{len(schedule)} accepted', flush=True)
        wp.synchronize_device(device=device)
        elapsed = perf_counter()-started
    return dict(seconds=elapsed, construction_seconds=construction_seconds,
        public_solve_seconds=driver_seconds if use_history else sum(row['public_solve_seconds'] for row in history),
        execution='device_transient_history' if use_history else 'individual_solve_calls',
        history_driver_overhead_seconds=driver_seconds-sum(row['public_solve_seconds'] for row in history) if use_history else 0.,
        history_readbacks={key:driver_info[key] for key in ('solver_head_readbacks', 'output_history_readbacks')} if use_history else None,
        independent_audit_seconds=sum(row['independent_audit_seconds'] for row in history),
        accepted=len(history) == len(schedule) and all(row['accepted'] for row in history), periods=history), np.stack(arrays=heads)


def run(*, args, root, settings):
    import warp as wp
    from DARCY_WARP_PACKAGE.solvers.npf662_kcycle import NPF662KCycleControls
    from DARCY_WARP_PACKAGE.validation.npf662_benchmark_cases import generate_case
    from benchmark_npf662_vs_mf6 import suite_settings
    wp.init()
    device = ('cuda:0' if wp.is_cuda_available() else 'cpu') if args.device == 'auto' else args.device
    profiles = read_profiles(root=root)
    overrides = {}
    if args.npf_history:
        if args.compare or args.npf_preconditioner == 'diagonal':
            raise ValueError('--npf-history requires the kcycle selection without --compare')
        overrides['npf_history'] = True
    settings = dict(settings, npf_options=dict(settings['npf_options']))
    if args.npf_controls is not None:
        payload = json.loads(args.npf_controls.read_text())
        if set(payload) != {'solve', 'newton', 'multigrid'}:
            raise ValueError('NPF control file requires exactly solve, newton and multigrid sections')
        if args.npf_preconditioner == 'diagonal' or args.compare:
            raise ValueError('An explicit NPF multigrid control file requires the kcycle selection without --compare')
        args.npf_preconditioner = 'kcycle'
        settings['npf_options'].update(payload['solve'], npf_newton_controls=payload['newton'])
        settings['kcycle'] = payload['multigrid']
        overrides['npf_controls'] = payload
    if args.npf_device_native:
        settings['npf_options']['npf_device_native'] = True
        overrides['npf_device_native'] = True
    if args.npf_relative_tolerance is not None:
        if not np.isfinite(args.npf_relative_tolerance) or args.npf_relative_tolerance < 0.:
            raise ValueError('--npf-relative-tolerance must be finite and nonnegative')
        settings['npf_options']['npf_final_relative_tolerance'] = args.npf_relative_tolerance
        overrides['npf_final_relative_tolerance'] = args.npf_relative_tolerance
    if args.legacy_max_cycles is not None:
        if args.legacy_max_cycles < 1:
            raise ValueError('--legacy-max-cycles must be positive')
        overrides['legacy_max_cycles'] = args.legacy_max_cycles
        for controls in profiles['steady'].values():
            controls['max_cycles'] = args.legacy_max_cycles
        profiles['transient']['max_cycles'] = args.legacy_max_cycles
    if args.tight or args.gpu_tight:
        for profile in (*profiles['steady'].values(),profiles['transient']):
            profile.update(settings['tight_cartesian'])
        settings['npf_options']['npf_final_relative_tolerance']=settings['tight_npf_relative']
        settings['npf_options']['npf_roundoff_acceptance']=True
        if args.tight:
            for profile in profiles['mf6'].values():
                profile.update(settings['tight_mf6'])
            settings['gates']=dict(settings['gates'],**settings['tight_gates'])
        overrides['tight']=dict(cartesian=settings['tight_cartesian'],
            mf6=settings['tight_mf6'] if args.tight else 'unchanged 2D sanity controls',
            npf_relative=settings['tight_npf_relative'],gates=settings['gates'])
    if args.cartesian_fgmres:
        if args.compare:
            raise ValueError('--cartesian-fgmres requires the fast-only selection')
        profiles['steady']['fast'].update(settings['cartesian_linear_options'])
        profiles['transient'].update(settings['cartesian_linear_options'])
        overrides['cartesian_linear_options']=settings['cartesian_linear_options']
    overrides['ghb']=bool(args.ghb)
    args.output.mkdir(parents=True, exist_ok=False)
    args.cache.mkdir(parents=True, exist_ok=True)
    factors = np.resize(a=np.repeat(a=settings['source_factors'], repeats=settings['block_periods']), new_shape=args.periods)
    controls = NPF662KCycleControls(**settings['kcycle'])
    result = dict(device=device, warp=wp.__version__, profiles=profiles, settings=settings,
        npf_source_sha256={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((root/'DARCY_WARP_PACKAGE/solvers').glob('npf662*.py'))},
        python=platform.python_version(), overrides=overrides,
        revision=subprocess.check_output(args=['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
        timing='Primary: synchronized public solver calls. With --npf-history, the entire transient NPF history call includes source updates and output download. Model build, cache loading, external audit and file saving excluded. Secondary: full verification runtime.',
        cases=[])
    sources=set((root/'DARCY_WARP_PACKAGE/solvers').glob('npf662*.py'))
    sources.update((root/'DARCY_WARP_PACKAGE/validation').glob('npf662*.py'))
    sources.update(root/name for name in (
        'DARCY_WARP_PACKAGE/solvers/fast_3d.py','DARCY_WARP_PACKAGE/solvers/linear_fgmres_3d.py',
        'DARCY_WARP_PACKAGE/solvers/session_3d.py','DARCY_WARP_PACKAGE/solvers/face_kernels_3d_f64.py',
        'DARCY_WARP_PACKAGE/solvers_3d.py','DARCY_WARP_PACKAGE/warped_darcy_3d.py',
        'DARCY_WARP_PACKAGE/physics/ghb_3d.py','DARCY_WARP_PACKAGE/physics/budget_3d.py',
        'working_tests/run_3d_sanity.py','working_tests/sanity_3d_controls.py'))
    result['solver_source_sha256']={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(sources)}
    for family in args.families:
        variants = args.variants if family == 'npf' else ['confined']
        for definition in suite_settings(sizes=args.sizes, variants=variants):
            geology = dict(definition['geology'], layer_k=np.asarray(definition['geology']['layer_k']))
            case = generate_case(settings=geology, convertible=definition['convertible'])
            if family == 'cartesian':
                interfaces = np.broadcast_to(array=(settings['cartesian_top']-settings['cartesian_dz']*np.arange(case.active.shape[0]+1))[:, None, None], shape=case.z_interfaces.shape).copy()
                source = case.source*(settings['cartesian_dx']**2)/(case.dx*case.dy)
                case = replace(case, z_interfaces=interfaces, dx=settings['cartesian_dx'],
                               dy=settings['cartesian_dx'], source=source)
            if args.ghb:
                from DARCY_WARP_PACKAGE.validation.npf662_benchmark_cases import add_ghb_boundaries
                case=add_ghb_boundaries(case=case,**settings['ghb'])
            for mode in args.modes:
                identity = f'{family}_{definition["case_id"]}_{mode}'+('_ghb' if args.ghb else '')
                row = dict(case=identity, case_sha256=case.digest(), shape=list(case.active.shape), runs=[], accepted=False)
                result['cases'].append(row)
                print(f'Starting {identity}; checking saved MF6 reference.', flush=True)
                try:
                    native_profile = profiles['mf6']['transient'] if mode == 'transient' else profiles['mf6']['steady_unconfined' if np.any(case.cell_types) else 'steady_confined']
                    reference, manifest = prepare_reference_with_progress(description=identity,
                        interval=settings['progress_seconds'], case=case, family=family, mode=mode, cache=args.cache,
                        executable=args.mf6, cached_only=args.cached_only, native_options=native_profile,
                        factors=factors, dt=settings['dt'], ss=settings['ss'], sy=settings['sy'] if family == 'npf' else 0., gates=settings['gates'])
                    row['reference'] = {key: value for key, value in manifest.items() if key not in ('native', 'sha256')}
                    row['reference']['artifact_sha256']=manifest['sha256']
                    row['reference']['native_identity']={key:manifest['native'][key] for key in
                        ('mf6_sha256','mf6_version','mf6_build_identity','native_success','executable_sha256','version')
                        if key in manifest['native']}
                    if args.compare:
                        labels = ('classic', 'fast') if family == 'cartesian' else ('diagonal', 'kcycle')
                    else:
                        selection = args.npf_preconditioner
                        if args.npf_history and family == 'npf' and mode == 'transient':selection = 'kcycle'
                        if selection == 'auto':
                            selection = 'kcycle' if case.active.size >= settings['kcycle_threshold_cells'] else 'diagonal'
                        labels = ('fast',) if family == 'cartesian' else (selection,)
                    for label in labels:
                        for repeat in range(args.repeats+1):
                            print(f'  {identity} {label}: {"warmup" if repeat == 0 else "repeat "+str(repeat)}', flush=True)
                            measured, heads = measure(case=case, family=family, mode=mode, label=label,
                                reference=reference, native_accepted=manifest['accepted'], device=device,
                                profiles=profiles, factors=factors, dt=settings['dt'], ss=settings['ss'], sy=settings['sy'],
                                npf_options=settings['npf_options'], kcycle_controls=controls, gates=settings['gates'],
                                progress_every=settings['progress_every'], npf_history=args.npf_history)
                            measured.update(implementation=label, repeat=repeat, warmup=repeat == 0)
                            row['runs'].append(measured)
                            np.savez_compressed(file=args.output/f'{identity}_{label}_{repeat}.npz', heads=heads)
                            save_json(path=args.output/'results.json', value=result)
                        timed = [item for item in row['runs'] if item['implementation'] == label and not item['warmup']]
                        passed = all(item['accepted'] for item in timed)
                        seconds = float(np.median(a=[item['public_solve_seconds'] for item in timed]))
                        total = float(np.median(a=[item['seconds'] for item in timed]))
                        print(f'{identity} {label}: {"PASS" if passed else "FAIL"}; median solve-only {seconds:.6f}s; verified total {total:.6f}s; reference cache={manifest["reused"]}', flush=True)
                    row['accepted'] = all(item['accepted'] for item in row['runs'])
                except Exception as error:
                    row['error'] = f'{type(error).__name__}: {error}'
                    print(f'{identity}: FAIL {row["error"]}', flush=True)
                save_json(path=args.output/'results.json', value=result)
    result['passed'] = all(row['accepted'] for row in result['cases'])
    summary = []
    for case in result['cases']:
        for implementation in sorted({row['implementation'] for row in case['runs']}):
            timed = [row for row in case['runs'] if row['implementation'] == implementation and not row['warmup']]
            periods = [period for row in timed for period in row['periods']]
            summary.append(dict(case=case['case'], implementation=implementation,
                passed=bool(len(timed) == args.repeats and all(row['accepted'] for row in timed)),
                median_public_solve_seconds=float(np.median(a=[row['public_solve_seconds'] for row in timed])),
                mf6_single_reference_run_seconds=case['reference'].get('native_single_run_seconds'),
                median_model_construction_seconds=float(np.median(a=[row['construction_seconds'] for row in timed])),
                median_extra_audit_seconds=float(np.median(a=[row['independent_audit_seconds'] for row in timed])),
                median_verified_total_seconds=float(np.median(a=[row['seconds'] for row in timed])),
                worst_head_rmse_m=max(period['head_rmse_m'] for period in periods),
                worst_head_max_m=max(period['head_max_m'] for period in periods),
                worst_ghb_flux_max_error=max(period['ghb_flux_max_error'] for period in periods),
                worst_head_error_in_scaled_eps=max(period['head_error_in_scaled_eps'] for period in periods),
                reference_reused=case['reference']['reused']))
    result['summary'] = summary
    if summary:
        with (args.output/'summary.csv').open(mode='w', newline='') as stream:
            writer = csv.DictWriter(f=stream, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    save_json(path=args.output/'results.json', value=result)
    print(f'Overall: {"PASS" if result["passed"] else "FAIL"}; evidence: {args.output.resolve()}', flush=True)
    return 0 if result['passed'] else 1


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    defaults = dict(sizes=[16, 32], families=['cartesian', 'npf'], modes=['steady', 'transient'],
        variants=['confined', 'convertible', 'phreatic'], device='auto', repeats=3, periods=20,
        output=Path('benchmark_results')/datetime.now(tz=timezone.utc).strftime('3d_sanity_%Y%m%dT%H%M%S'),
        cache=Path('benchmark_results/mf6_3d_cache'),
        mf6=Path(shutil.which('mf6') or '/bin/modflow/mf6'), preconditioner='auto')
    settings = dict(source_factors=[1., 1.5, .25, 1.], block_periods=5, dt=7., ss=1e-5, sy=.15,
        progress_seconds=30., progress_every=5,
        cartesian_top=8., cartesian_dz=10., cartesian_dx=25., kcycle_threshold_cells=50000,
        kcycle=dict(levels=5, pre_sweeps=2, post_sweeps=2, coarse_sweeps=20, cycles=1, omega=.8),
        npf_options=dict(max_outer=80, max_cycles=4000, krylov_restart=48),
        gates=dict(rmse=1e-3, maximum=1e-2, budget_percent=1.),
        ghb=dict(conductance_factor=1.,draining_offset=-.5,injecting_offset=1.),
        cartesian_linear_options=dict(linear_acceleration='fgmres',linear_krylov_restart=8,
            linear_kcycle_controls=dict(levels=3,pre_sweeps=2,post_sweeps=2,coarse_sweeps=2,
                cached_line_factors=True,preconditioner_precision='float32',device_hierarchy_build=True)),
        tight_cartesian=dict(max_cycles=4000,rel_tol=1e-14,abs_tol_min=1e-12,dh_rms_tol=None,dh_max_tol=None,final_budget_relative_tol=1e-14),
        tight_mf6=dict(outer_dvclose=1e-12,inner_dvclose=1e-12,rcloserecord=[1e-9,'STRICT'],
            under_relaxation='NONE',preconditioner_levels=8,preconditioner_drop_tolerance=1e-4,
            relaxation_factor=0.,reordering_method='RCM',number_orthogonalizations=8),
        tight_npf_relative=1e-12,tight_gates=dict(rmse=1e-10,maximum=1e-9,head_residual_rms=1e-12))
    os.environ['DARCY_FLOAT'] = 'float64'
    sys.path.insert(0, str(root))
    arguments = parse_arguments(defaults=defaults)
    raise SystemExit(run(args=arguments, root=root, settings=settings))

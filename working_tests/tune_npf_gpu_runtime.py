"""Explore NPF runtime controls; finalists require --audit and repeated timing."""
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
import argparse
import cProfile
import hashlib
import json
import statistics
import subprocess

import numpy as np
import warp as wp

from benchmark_npf662_vs_mf6 import suite_settings
from benchmark_npf_calibration_runtime import make_model
from sanity_3d_controls import read_profiles
from run_3d_sanity import native_elapsed_seconds, sanity_reference_valid
from DARCY_WARP_PACKAGE.solvers.npf662_newton import NPF662NewtonControls
from DARCY_WARP_PACKAGE.solvers.npf662_kcycle import NPF662KCycleControls
from DARCY_WARP_PACKAGE.validation.npf662_benchmark_cases import generate_case
from DARCY_WARP_PACKAGE.validation.npf662_transient_reference import NPF662TransientReference


def load_reference(*, folder, case, factors, options, native_options):
    paths = [folder/'manifest.json'] if (folder/'manifest.json').is_file() else sorted(folder.glob('*/manifest.json'))
    expected = dict(case_sha256=case.digest(), factors=factors.tolist(), native_options=native_options,
        mode='transient', family='npf', **{key:options[key] for key in ('dt', 'ss', 'sy')})
    matches = []
    for path in paths:
        manifest = json.loads(path.read_text())
        if all(manifest['inputs'].get(key) == value for key, value in expected.items()):
            matches.append((path.parent, manifest))
    if len(matches) != 1:
        raise ValueError('Exactly one matching cached MF6 history required; generate it with run_3d_sanity.py first')
    folder, manifest = matches[0]
    # Older cache manifests retain the strict NPF 1e-8 audit result. The
    # portable sanity runner independently applies the shared 2D validity gate.
    if not sanity_reference_valid(info=manifest['native'], mode='transient', gates=dict(budget_percent=1.)):
        raise ValueError('Cached MF6 history failed the shared sanity validity gates')
    if 'Normal termination of simulation' not in (folder/'native/stdout.log').read_text():
        raise ValueError('Cached MF6 history lacks normal termination')
    for filename, checksum in manifest['sha256'].items():
        if hashlib.sha256((folder/filename).read_bytes()).hexdigest() != checksum:
            raise ValueError('Cached MF6 reference hash mismatch')
    path = folder/'reference.npz'
    with np.load(file=path) as data:
        heads = data['heads'].copy()
    if heads.shape != (len(factors), *case.active.shape) or not np.isfinite(heads).all():
        raise ValueError('Cached MF6 heads are invalid or have the wrong shape')
    return heads, manifest, folder


def save_results(*, output, result):
    temporary = output/'results.json.tmp'
    temporary.write_text(json.dumps(result, indent=2)+'\n')
    temporary.replace(target=output/'results.json')


def run_history(*, case, factors, options, controls, device, reference, audit, head_gate, profile_path):
    history_start = perf_counter()
    previous = case.initial.copy()
    steps = []
    audit_seconds = 0.
    profiler = cProfile.Profile() if profile_path else None
    construction_start = perf_counter()
    model = make_model(case=case, device=device)
    wp.synchronize_device(device=device)
    construction_seconds = perf_counter()-construction_start
    with model:
        for period, factor in enumerate(factors):
            step = replace(case, source=case.source*factor)
            model.update_rhs(rhs=step.source)
            model.update_initial_head(initial_head=previous)
            kwargs = dict(options, **controls, head_prev=previous, npf_cell_type=case.cell_types)
            wp.synchronize_device(device=device)
            if profiler:profiler.enable()
            start = perf_counter()
            head, info = model.solve(**kwargs)
            wp.synchronize_device(device=device)
            elapsed = perf_counter()-start
            if profiler:profiler.disable()
            error = float(np.max(np.abs(head-reference[period])))
            accepted = bool(info['converged'] and error <= head_gate)
            checks = info['device_final_metrics']
            if audit:
                audit_start = perf_counter()
                checks, unused = NPF662TransientReference(case=step, previous=previous,
                    dt=options['dt'], ss=options['ss'], sy=options['sy']).evaluate(
                        head=head, relative_tolerance=kwargs['npf_final_relative_tolerance'])
                accepted = accepted and checks['independent_accepted']
                audit_seconds += perf_counter()-audit_start
            steps.append(dict(period=period+1, accepted=bool(accepted), seconds=elapsed,
                numerical_seconds=info['seconds'], setup_seconds=info['setup_seconds'],
                reference_setup_seconds=info['reference_setup_seconds'],
                multigrid_device_seconds=info['multigrid']['device_apply_seconds'],
                krylov_synchronization_seconds=info['krylov_synchronization_seconds'],
                krylov=info['krylov_iterations'], outer=info['outer_iterations'],
                residual_evaluations=info['nonlinear_residual_evaluations'],
                timestep_predictor_used=info.get('timestep_predictor_used', False),
                head_max_error_m=error, physical=checks))
            if not accepted:break
            previous = head
    if profiler:profiler.dump_stats(file=str(profile_path))
    return dict(passed=len(steps) == len(factors) and all(s['accepted'] for s in steps),
        construction_seconds=construction_seconds, extra_audit_seconds=audit_seconds,
        history_wall_seconds=perf_counter()-history_start,
        seconds=sum(s['seconds'] for s in steps), numerical_seconds=sum(s['numerical_seconds'] for s in steps),
        krylov=sum(s['krylov'] for s in steps), outer=sum(s['outer'] for s in steps), steps=steps)


def run_sweep(*, root, cache, output, size, case_variant, device, factors, options, variants, repeats, audit, head_gate, profile):
    wp.init()
    output.mkdir(parents=True, exist_ok=False)
    definition = suite_settings(sizes=[size], variants=[case_variant])[0]
    geology = dict(definition['geology'], layer_k=np.asarray(definition['geology']['layer_k']))
    case = generate_case(settings=geology, convertible=definition['convertible'])
    options = dict(options, unconfined=bool(np.any(case.cell_types)))
    profiles = read_profiles(root=root)
    reference, manifest, reference_folder = load_reference(folder=cache, case=case, factors=factors, options=options,
        native_options=profiles['mf6']['transient'])
    source_paths = sorted((root/'DARCY_WARP_PACKAGE/solvers').glob('npf662*.py'))
    sources = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    result = dict(revision=subprocess.check_output(args=['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256=sources, case_sha256=case.digest(), reference=manifest['fingerprint'],
        mf6_single_engine_seconds=native_elapsed_seconds(text=(reference_folder/'native/stdout.log').read_text()),
        mf6_common_physical_metrics=[period['physical'] for period in manifest['native']['periods']],
        options=options, head_comparison_gate_m=head_gate, audit=audit, repeats=repeats, profiled=profile,
        timing='Synchronized public model.solve calls only, same exclusion boundaries as run_3d_sanity.py.', variants=[])
    for specification in variants:
        controls = dict(npf_newton_controls=NPF662NewtonControls(inner_forcing=specification['forcing'], **specification.get('newton', {})),
            npf_kcycle_controls=NPF662KCycleControls(**specification['mg']))
        serialized = {k:asdict(v) for k,v in controls.items()}
        controls['npf_final_relative_tolerance'] = specification.get('relative_limit', options['npf_final_relative_tolerance'])
        serialized['npf_final_relative_tolerance'] = controls['npf_final_relative_tolerance']
        if 'restart' in specification:
            controls['krylov_restart'] = specification['restart']
            serialized['krylov_restart'] = controls['krylov_restart']
        row = dict(specification=specification, controls=serialized, runs=[])
        result['variants'].append(row)
        for repeat in range(repeats+1):
            print(f"{specification['name']}: {'warmup' if repeat == 0 else 'repeat '+str(repeat)}", flush=True)
            measured = run_history(case=case, factors=factors, options=options, controls=controls,
                device=device, reference=reference, audit=audit, head_gate=head_gate,
                profile_path=output/f"{specification['name']}_{repeat}.prof" if profile else None)
            measured.update(warmup=repeat == 0)
            row['runs'].append(measured)
            print(f"  passed={measured['passed']} public={measured['seconds']:.4f}s numerical={measured['numerical_seconds']:.4f}s Krylov={measured['krylov']} outer={measured['outer']}", flush=True)
            save_results(output=output, result=result)
            if not measured['passed']:break
        timed = [r for r in row['runs'] if not r['warmup']]
        row['passed'] = len(timed) == repeats and all(r['passed'] for r in row['runs'])
        row['median_seconds'] = statistics.median(r['seconds'] for r in timed) if timed else None
        save_results(output=output, result=result)
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--variants', nargs='+', default=['baseline'])
    parser.add_argument('--cache', type=Path, default=Path('benchmark_results/mf6_3d_cache'))
    parser.add_argument('--size', type=int, default=500)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--case', choices=['confined', 'convertible'], default='convertible')
    parser.add_argument('--head-gate', type=float, default=1e-6)
    arguments = parser.parse_args()
    size = arguments.size
    device = arguments.device
    cache = arguments.cache
    factors = np.repeat([1., 1.5, .25, 1.], 5)
    options = dict(unconfined=True, transient=True, dt=7., ss=1e-5, sy=.15,
        npf_preconditioner='kcycle', npf_device_native=True, npf_final_relative_tolerance=1e-7,
        max_outer=80, max_cycles=4000, krylov_restart=48)
    head_gate = arguments.head_gate
    baseline_mg = dict(levels=5, pre_sweeps=2, post_sweeps=2, coarse_sweeps=20)
    variants = [dict(name='baseline', forcing=.01, mg=baseline_mg)]
    forcing_values = [.003, .03, .1, .3]
    coarse_values = [2, 5, 10]
    level_values = [2, 3, 4]
    for forcing in forcing_values:
        variants.append(dict(name=f'forcing_{forcing}', forcing=forcing, mg=baseline_mg))
    for coarse in coarse_values:
        variants.append(dict(name=f'coarse_{coarse}', forcing=.01, mg=dict(baseline_mg, coarse_sweeps=coarse)))
    for levels in level_values:
        variants.append(dict(name=f'levels_{levels}', forcing=.01, mg=dict(baseline_mg, levels=levels)))
    variants.append(dict(name='light', forcing=.01, mg=dict(levels=3, pre_sweeps=1, post_sweeps=1, coarse_sweeps=5)))
    combined_forcing = [.003, .01, .03, .1]
    relative_values = [1.5e-7, 3e-7, 1e-6]
    for forcing in combined_forcing:
        variants.append(dict(name=f'combined_{forcing}', forcing=forcing,
            mg=dict(baseline_mg, levels=3, coarse_sweeps=2)))
        variants.append(dict(name=f'vcycle_{forcing}', forcing=forcing,
            mg=dict(baseline_mg, levels=5, coarse_sweeps=2, cycle_kind='v')))
    for tolerance in relative_values:
        variants.append(dict(name=f'relative_{tolerance}', forcing=.01, mg=baseline_mg, relative_limit=tolerance))
    factor_levels = [1, 2, 3, 4, 5]
    factor_sweeps = [1, 2]
    for levels in factor_levels:
        for sweeps in factor_sweeps:
            variants.append(dict(name=f'factored_{levels}_{sweeps}', forcing=.03,
                mg=dict(levels=levels, pre_sweeps=sweeps, post_sweeps=sweeps,
                    coarse_sweeps=2, cached_line_factors=True)))
    for tolerance in relative_values:
        variants.append(dict(name=f'combined_relative_{tolerance}', forcing=.03, relative_limit=tolerance,
            mg=dict(levels=3, pre_sweeps=2, post_sweeps=2, coarse_sweeps=2, cached_line_factors=True)))
    adaptive_values = [.003, .01, .03, .1, .3]
    for forcing in adaptive_values:
        for adaptive in (False, True):
            variants.append(dict(name=f'adaptive_{forcing}_{adaptive}', forcing=forcing,
                newton=dict(adaptive_forcing=adaptive),
                mg=dict(levels=3, pre_sweeps=2, post_sweeps=2, coarse_sweeps=2, cached_line_factors=True)))
    for forcing in (.01, .03, .1):
        for adaptive in (False, True):
            variants.append(dict(name=f'predictor_{forcing}_{adaptive}', forcing=forcing,
                newton=dict(adaptive_forcing=adaptive, timestep_predictor=True),
                mg=dict(levels=3, pre_sweeps=2, post_sweeps=2, coarse_sweeps=2, cached_line_factors=True)))
    for forcing in (.003, .01, .03, .1):
        for predictor in (False, True):
            variants.append(dict(name=f'mixed_{forcing}_{predictor}', forcing=forcing,
                newton=dict(timestep_predictor=predictor),
                mg=dict(levels=3, pre_sweeps=2, post_sweeps=2, coarse_sweeps=2,
                    cached_line_factors=True, preconditioner_precision='float32')))
    for levels in (2, 3, 4):
        for sweeps in (2, 3, 4):
            variants.append(dict(name=f'mixed_grid_{levels}_{sweeps}', forcing=.03, restart=8,
                newton=dict(timestep_predictor=True),
                mg=dict(levels=levels, pre_sweeps=sweeps, post_sweeps=sweeps, coarse_sweeps=2,
                    cached_line_factors=True, preconditioner_precision='float32')))
    for forcing in (.003, .01, .03, .1):
        variants.append(dict(name=f'quadratic_{forcing}', forcing=forcing, restart=8,
            newton=dict(timestep_predictor=True, timestep_predictor_order=2),
            mg=dict(levels=3, pre_sweeps=2, post_sweeps=2, coarse_sweeps=2,
                cached_line_factors=True, preconditioner_precision='float32')))
    for forcing in (.003, .01, .03, .1):
        for predictor in (False, True):
            variants.append(dict(name=f'current_{forcing}_{predictor}', forcing=forcing, restart=8,
                newton=dict(timestep_predictor=predictor),
                mg=dict(levels=3, pre_sweeps=2, post_sweeps=2, coarse_sweeps=2,
                    cached_line_factors=True, preconditioner_precision='float32', coefficient_state='current')))
    for tolerance in relative_values:
        variants.append(dict(name=f'current_relative_{tolerance}', forcing=.01, restart=8, relative_limit=tolerance,
            newton=dict(timestep_predictor=True),
            mg=dict(levels=3, pre_sweeps=2, post_sweeps=2, coarse_sweeps=2,
                cached_line_factors=True, preconditioner_precision='float32', coefficient_state='current')))
    for forcing in (.001, .01, .1):
        for restart in (24, 48):
            variants.append(dict(name=f'confined_{forcing}_{restart}', forcing=forcing, restart=restart,
                mg=dict(levels=5, pre_sweeps=2, post_sweeps=2, coarse_sweeps=2,
                    cached_line_factors=True, preconditioner_precision='float32')))
    selected = variants if arguments.variants == ['all'] else [v for v in variants if v['name'] in arguments.variants]
    unknown = set(arguments.variants)-{v['name'] for v in variants}-{'all'}
    if not selected or unknown or arguments.repeats < 1 or arguments.size < 2 or not np.isfinite(head_gate) or head_gate < 0.:
        raise ValueError('Select known variants and at least one repeat')
    run_sweep(root=root, cache=cache, output=arguments.output, size=size, case_variant=arguments.case, device=device, factors=factors,
        options=options, variants=selected, repeats=arguments.repeats, audit=arguments.audit,
        head_gate=head_gate, profile=arguments.profile)

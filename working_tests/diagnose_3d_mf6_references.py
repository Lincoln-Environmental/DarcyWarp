"""Recover missing MF6 references with a separately recorded solver strategy.

Original baseline artifacts and convergence tolerances are preserved. This
script verifies baseline result hashes and re-audits the saved GPU histories;
it does not rerun or retime the GPU solver.
"""
from pathlib import Path
from datetime import datetime, timezone
from copy import deepcopy
import argparse
import hashlib
import json
import os
import sys


def file_digest(*, path):
    with path.open(mode='rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(*, path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(data=json.dumps(obj=value, indent=2, allow_nan=False) + '\n')
    temporary.replace(target=path)


def diagnostic_options(*, original, overrides, tolerance_keys):
    result = deepcopy(original)
    result.update(deepcopy(overrides))
    for key in tolerance_keys:
        if result[key] != original[key]:
            raise ValueError(f'Diagnostic must preserve baseline convergence tolerance: {key}')
    return result


def baseline_inputs(*, row, attempt, result):
    import numpy as np
    from DARCY_WARP_PACKAGE.validation.npf662_reference import load_npf662_case
    settings = result['settings']
    case_row = result['cases'][0]
    reference = case_row['reference']
    original = reference['native_options'] if 'native_options' in reference else reference['inputs']['native_options']
    factors = np.resize(a=np.repeat(a=settings['source_factors'], repeats=settings['block_periods']),
                        new_shape=row['periods'])
    sy = settings['sy'] if row['family'] == 'npf' else 0.
    payload = dict(case_sha256=case_row['case_sha256'], mode=row['mode'], native_options=original,
                   factors=factors.tolist(), dt=settings['dt'], ss=settings['ss'], sy=sy,
                   family=row['family'], acceptance='NPF fixed physical contract' if row['family'] == 'npf' else settings['gates'])
    fingerprint = hashlib.sha256(json.dumps(obj=payload, sort_keys=True).encode()).hexdigest()
    command = attempt['command']
    cache = Path(command[command.index('--cache') + 1])
    case_path = cache / fingerprint / 'inputs.npz'
    case = load_npf662_case(path=case_path)
    if case.digest() != case_row['case_sha256']:
        raise ValueError('Saved physical inputs do not match the measured GPU case')
    return case, original, factors, sy, case_path


def compare_history(*, heads, reference, case, expected_periods, roundoff_ulps, flux_floor,
                    expected_times=None):
    import numpy as np
    expected_shape = (expected_periods, *case.active.shape)
    if heads.shape != expected_shape or reference['heads'].shape != expected_shape:
        raise ValueError('Complete matching time histories are required')
    if reference['ghb_inflows'].shape != expected_shape:
        raise ValueError('Native GHB history does not match the requested time grid')
    if expected_times is not None and not np.array_equal(reference.get('times'), expected_times):
        raise ValueError('Native timestamps differ from the measured GPU endpoints')
    free = (case.active != 0) & (case.fixed == 0)
    maximum = rmse = ghb_maximum = 0.
    ghb_valid = True
    periods = []
    for index in range(expected_periods):
        head, native = heads[index], reference['heads'][index]
        if not np.isfinite(head[free]).all() or not np.isfinite(native[free]).all():
            raise ValueError('Nonfinite comparison heads')
        delta = head[free] - native[free]
        period_max = float(np.max(np.abs(delta)))
        period_rmse = float(np.sqrt(np.mean(delta * delta)))
        maximum, rmse = max(maximum, period_max), max(rmse, period_rmse)
        valid, ghb_error = True, 0.
        if case.ghb_conductance is not None:
            actual = case.ghb_conductance * (case.ghb_head - head)
            error = np.abs(actual - reference['ghb_inflows'][index])
            bound = case.ghb_conductance * np.abs(head - native)
            bound += roundoff_ulps * np.finfo(float).eps * case.ghb_conductance * (np.abs(head) + np.abs(case.ghb_head))
            valid = bool(np.all(np.isfinite(error)) and np.all(error <= np.maximum(bound, flux_floor)))
            ghb_error = float(np.max(error))
            ghb_maximum = max(ghb_maximum, ghb_error)
            ghb_valid = ghb_valid and valid
        periods.append(dict(period=index + 1, head_max_m=period_max, head_rmse_m=period_rmse,
                            ghb_flux_max_error=ghb_error, ghb_law_valid=valid))
    return dict(head_max_m=maximum, head_rmse_m=rmse, ghb_flux_max_error=ghb_maximum,
                ghb_law_valid=ghb_valid, periods=periods)


def audit_saved_history(*, heads, case, factors, mode, dt, ss, sy, original_periods,
                       roundoff_acceptance):
    from dataclasses import replace
    from DARCY_WARP_PACKAGE.validation.npf662_matrix_free import NPF662MatrixFreeReference
    from DARCY_WARP_PACKAGE.validation.npf662_transient_reference import NPF662TransientReference
    if len(heads) != len(original_periods):
        raise ValueError('Saved heads no longer match the measured GPU history')
    old = case.initial.copy()
    audits = []
    for index, head in enumerate(heads):
        step = replace(case, source=case.source * factors[index]) if mode == 'transient' else case
        auditor = NPF662TransientReference(case=step, previous=old, dt=dt, ss=ss, sy=sy) if mode == 'transient' else NPF662MatrixFreeReference(case=step)
        thresholds = original_periods[index]['physical']['final_acceptance_thresholds']
        checks, _ = auditor.evaluate(head=head,
            physical_tolerance=thresholds['physical_rms_m3_day'],
            relative_tolerance=thresholds['relative_cell_imbalance'],
            head_tolerance_m=thresholds['head_equivalent_rms_m'],
            boundary_tolerance_m=thresholds['prescribed_head_error_m'],
            roundoff_acceptance=roundoff_acceptance)
        audits.append(checks)
        old = head.copy()
    return audits


def diagnose_case(*, row, attempt, directory, executable, overrides, tolerance_keys,
                  timeout_seconds, progress_seconds, roundoff_ulps, flux_floor):
    import numpy as np
    from working_tests.run_3d_sanity import prepare_reference_with_progress
    baseline_path = Path(row['output']) / 'results.json'
    baseline_hash = file_digest(path=baseline_path)
    if baseline_hash != attempt['results_sha256']:
        raise ValueError('Baseline GPU results changed; refusing comparison')
    result = json.loads(s=baseline_path.read_text())
    if len(result['cases']) != 1 or len(result['summary']) != 1:
        raise ValueError('Exactly one measured case is required')
    case, original, factors, sy, case_path = baseline_inputs(row=row, attempt=attempt, result=result)
    options = diagnostic_options(original=original, overrides=overrides, tolerance_keys=tolerance_keys)
    evidence = dict(job={key: row[key] for key in ('size', 'ratio', 'ghb', 'family', 'variant', 'mode', 'periods')},
        baseline_output=row['output'], baseline_results_sha256=baseline_hash,
        physical_inputs_sha256=file_digest(path=case_path), physical_case_digest=case.digest(),
        original_native_options=original, diagnostic_native_options=options,
        unchanged_tolerances={key: options[key] for key in tolerance_keys},
        baseline_reference_error=row.get('error'), gpu_seconds=row['gpu_seconds'],
        baseline_gpu_physical_accepted=row['gpu_physical_accepted'],
        gpu_physical_accepted=False, complete=False, passed=False,
        preferred_head_target_met=False, mf6_seconds=None, speedup=None, runs=[])
    write_json(path=directory / 'result.json', value=evidence)
    try:
        reference, manifest = prepare_reference_with_progress(description=f'diagnostic {row["fingerprint"][:12]}',
            interval=progress_seconds, case=case, family=row['family'], mode=row['mode'],
            cache=directory / 'native_cache', executable=executable, cached_only=False,
            native_options=options, factors=factors, dt=result['settings']['dt'], ss=result['settings']['ss'],
            sy=sy, gates=result['settings']['gates'], timeout_seconds=timeout_seconds)
        evidence.update(native_reference_accepted=manifest['accepted'],
                        reference_artifact_sha256=manifest['sha256'],
                        mf6_seconds=manifest['native_single_run_seconds'],
                        strict_native_npf_audit=manifest.get('strict_native_npf_audit'),
                        native_acceptance_criteria=manifest['acceptance_criteria'],
                        native_identity={key: manifest['native'][key] for key in
                            ('mf6_sha256', 'mf6_version', 'mf6_build_identity', 'executable_sha256', 'version')
                            if key in manifest['native']})
        measured = result['cases'][0]
        expected_periods = row['periods'] if row['mode'] == 'transient' else 1
        expected_times = result['settings']['dt'] * np.arange(1, expected_periods + 1) if row['mode'] == 'transient' else None
        evidence['times_days'] = expected_times.tolist() if expected_times is not None else None
        for run in measured['runs']:
            path = Path(row['output']) / f'{measured["case"]}_{run["implementation"]}_{run["repeat"]}.npz'
            checksum = file_digest(path=path)
            with np.load(file=path, allow_pickle=False) as saved:
                heads = saved['heads']
                metrics = compare_history(heads=heads, reference=reference, case=case,
                    expected_periods=expected_periods, roundoff_ulps=roundoff_ulps, flux_floor=flux_floor,
                    expected_times=expected_times)
                audits = audit_saved_history(heads=heads, case=case, factors=factors, mode=row['mode'],
                    dt=result['settings']['dt'], ss=result['settings']['ss'], sy=sy,
                    original_periods=run['periods'],
                    roundoff_acceptance=bool(row['family'] == 'npf' and result['settings']['npf_options'].get('npf_roundoff_acceptance', False)))
            evidence['runs'].append(dict(metrics, repeat=run['repeat'], warmup=run['warmup'],
                                         gpu_heads_sha256=checksum, independent_physical=audits))
        timed = [run for run in evidence['runs'] if not run['warmup']]
        if len(timed) != row['repeats']:
            raise ValueError('Missing timed GPU histories')
        evidence.update(head_max_m=max(run['head_max_m'] for run in evidence['runs']),
                        head_rmse_m=max(run['head_rmse_m'] for run in evidence['runs']))
        gates = result['settings']['gates']
        evidence['gpu_physical_accepted'] = bool(row['gpu_physical_accepted'] and all(
            checks['independent_accepted'] for run in evidence['runs'] for checks in run['independent_physical']))
        passed = bool(manifest['accepted'] and evidence['gpu_physical_accepted']
            and all(run['ghb_law_valid'] for run in evidence['runs'])
            and evidence['head_max_m'] <= gates['maximum'] and evidence['head_rmse_m'] <= gates['rmse'])
        evidence.update(passed=passed, preferred_head_target_m=row['preferred_head_target'],
            preferred_head_target_met=bool(passed and evidence['head_max_m'] <= row['preferred_head_target']))
        if passed:
            evidence['speedup'] = evidence['mf6_seconds'] / evidence['gpu_seconds']
    except Exception as error:
        evidence['error'] = f'{type(error).__name__}: {error}'
    evidence['complete'] = True
    write_json(path=directory / 'result.json', value=evidence)
    return evidence


def run_diagnostics(*, args, overrides, tolerance_keys, roundoff_ulps, flux_floor):
    matrix = json.loads(s=args.matrix.read_text())
    if not matrix.get('complete') or len(matrix['selected']) != args.expected_cases:
        raise ValueError('Selected baseline cases must be complete before running diagnostics')
    if file_digest(path=args.mf6) != matrix['identity']['mf6_sha256']:
        raise ValueError('Diagnostic MF6 executable differs from the pinned baseline executable')
    selected = [row for row in matrix['selected'] if not row['native_reference_accepted'] and row['gpu_physical_accepted']]
    attempts = {entry['fingerprint']: entry for entry in matrix['attempts'] if entry.get('complete')}
    identity = dict(baseline_source_identity=matrix['identity'],
                    diagnostic_sources={name: file_digest(path=args.root / name)
                                        for name in matrix['identity']['sources']},
                    script_sha256=file_digest(path=Path(__file__).resolve()), overrides=overrides,
                    tolerance_keys=list(tolerance_keys), timeout_seconds=args.timeout, python=sys.version,
                    thread_limits={key: os.environ.get(key) for key in
                                   ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / 'diagnostics.json'
    previous = json.loads(s=report_path.read_text()) if report_path.exists() else None
    if previous is not None and previous['identity'] != identity:
        raise ValueError('Diagnostic identity changed; choose a new output directory')
    report = previous or dict(identity=identity, started_utc=datetime.now(tz=timezone.utc).isoformat(),
        scope='separate MF6 strategy diagnostic with unchanged baseline convergence tolerances',
        timing='Saved baseline GPU medians versus new diagnostic MF6 native process times; original benchmarks preserved.',
        attempts=[], selected=[], baseline_matrix_inputs=[], complete=False)
    report['baseline_matrix_inputs'].append(dict(path=str(args.matrix.resolve()), sha256=file_digest(path=args.matrix)))
    report['selected'] = []
    report['complete'] = False
    for row in selected:
        earlier = [record for record in report['attempts'] if record['fingerprint'] == row['fingerprint']]
        if earlier and earlier[-1]['result']['complete'] and not (args.retry_failed and not earlier[-1]['result']['passed']):
            record = earlier[-1]
            if record['result']['baseline_results_sha256'] != attempts[row['fingerprint']]['results_sha256']:
                raise ValueError('Baseline result changed since the previous diagnostic')
        else:
            directory = args.output / f'{row["fingerprint"][:12]}_attempt{len(earlier) + 1}'
            directory.mkdir()
            record = dict(fingerprint=row['fingerprint'], output=str(directory),
                          result=dict(complete=False, passed=False))
            report['attempts'].append(record)
            write_json(path=report_path, value=report)
            print(f'Starting {directory.name}: {row["family"]} {row["variant"]} {row["mode"]}, Kh/Kv={row["ratio"]}, GHB={row["ghb"]}', flush=True)
            result = diagnose_case(row=row, attempt=attempts[row['fingerprint']], directory=directory,
                executable=args.mf6, overrides=overrides, tolerance_keys=tolerance_keys,
                timeout_seconds=args.timeout, progress_seconds=args.progress_seconds,
                roundoff_ulps=roundoff_ulps, flux_floor=flux_floor)
            record['result'] = result
        report['selected'].append(record)
        write_json(path=report_path, value=report)
        print(f'Finished {row["fingerprint"][:12]}: passed={record["result"]["passed"]}', flush=True)
    report.update(complete=True, passed=all(row['result']['passed'] for row in report['selected']))
    write_json(path=report_path, value=report)
    return 0 if report['passed'] else 1


def parse_arguments(*, defaults):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=defaults['root'])
    parser.add_argument('--matrix', type=Path, default=defaults['matrix'])
    parser.add_argument('--output', type=Path, default=defaults['output'])
    parser.add_argument('--mf6', type=Path, default=defaults['mf6'])
    parser.add_argument('--timeout', type=float, default=defaults['timeout'])
    parser.add_argument('--expected-cases', type=int, default=defaults['expected_cases'])
    parser.add_argument('--progress-seconds', type=float, default=defaults['progress_seconds'])
    parser.add_argument('--retry-failed', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    root = Path.cwd()
    defaults = dict(root=root, matrix=Path('benchmark_results/ugly_t_3d/large_500_sanity/matrix.json'),
        output=Path('benchmark_results/ugly_t_3d/mf6_strategy_diagnostics'),
        mf6=Path('/bin/modflow/mf6'), timeout=1800., expected_cases=36, progress_seconds=30.)
    overrides = dict(under_relaxation='NONE', backtracking_number=0,
        linear_acceleration='BICGSTAB', preconditioner_levels=8,
        preconditioner_drop_tolerance=1e-4,
        number_orthogonalizations=8, relaxation_factor=0.)
    tolerance_keys = ('outer_dvclose', 'inner_dvclose', 'rcloserecord')
    roundoff_ulps = 32.
    flux_floor = 1e-14
    thread_settings = dict(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                           DARCY_FLOAT='float64', MPLBACKEND='Agg')
    os.environ.update(thread_settings)
    arguments = parse_arguments(defaults=defaults)
    arguments.root = arguments.root.resolve()
    arguments.matrix = (arguments.root / arguments.matrix).resolve()
    arguments.output = (arguments.root / arguments.output).resolve()
    sys.path.insert(0, str(arguments.root))
    sys.path.insert(0, str(arguments.root / 'working_tests'))
    raise SystemExit(run_diagnostics(args=arguments, overrides=overrides,
        tolerance_keys=tolerance_keys, roundoff_ulps=roundoff_ulps, flux_floor=flux_floor))

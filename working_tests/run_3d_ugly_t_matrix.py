"""Resumable, serial GPU/MF6 ugly-T tests with GHB off/on and Kh/Kv sweeps."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys

import numpy as np


def parse_arguments(*, defaults):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', nargs='+', type=int, default=defaults['sizes'])
    parser.add_argument('--vertical-anisotropy', nargs='+', type=float, default=defaults['ratios'])
    parser.add_argument('--ghb-modes', nargs='+', choices=['off', 'on'], default=defaults['ghb_modes'])
    parser.add_argument('--families', nargs='+', choices=['cartesian', 'npf'], default=defaults['families'])
    parser.add_argument('--variants', nargs='+', choices=['confined', 'convertible'], default=defaults['variants'])
    parser.add_argument('--modes', nargs='+', choices=['steady', 'transient'], default=defaults['modes'])
    parser.add_argument('--periods', type=int, default=defaults['periods'])
    parser.add_argument('--repeats', type=int, default=defaults['repeats'])
    parser.add_argument('--seed', type=int, default=defaults['seed'])
    parser.add_argument('--reference-thickness', type=float, default=defaults['reference_thickness'])
    parser.add_argument('--preferred-head-target', type=float, default=defaults['preferred_head_target'])
    parser.add_argument('--mf6-timeout', type=float, default=defaults['mf6_timeout'])
    parser.add_argument('--device', default=defaults['device'])
    parser.add_argument('--mf6', type=Path, default=defaults['mf6'])
    parser.add_argument('--output', type=Path, default=defaults['output'])
    parser.add_argument('--cache', type=Path, default=defaults['cache'])
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    if min(args.sizes) < 2 or min(args.periods, args.repeats) < 1:
        parser.error('Sizes >=2 and positive periods/repeats are required')
    if any(not np.isfinite(value) or value <= 0. for value in
           (*args.vertical_anisotropy, args.reference_thickness, args.preferred_head_target, args.mf6_timeout)):
        parser.error('Positive finite anisotropy, thickness, target and timeout are required')
    return args


def digest_file(*, path):
    with path.open(mode='rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(*, path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(data=json.dumps(obj=value, indent=2, allow_nan=False) + '\n')
    temporary.replace(target=path)


def case_jobs(*, args):
    jobs = []
    for size in args.sizes:
        for ratio in args.vertical_anisotropy:
            for ghb in args.ghb_modes:
                for family in args.families:
                    variants = ['confined'] if family == 'cartesian' else args.variants
                    for variant in variants:
                        for mode in args.modes:
                            jobs.append(dict(size=size, ratio=ratio, ghb=ghb, family=family, variant=variant, mode=mode,
                                periods=args.periods, repeats=args.repeats, seed=args.seed,
                                reference_thickness=args.reference_thickness, preferred_head_target=args.preferred_head_target,
                                mf6_timeout=args.mf6_timeout, device=args.device))
    return jobs


def job_command(*, job, root, directory, cache, mf6):
    command = [sys.executable, str(root / 'working_tests/run_3d_sanity.py'),
        '--sizes', str(job['size']), '--families', job['family'], '--variants', job['variant'],
        '--modes', job['mode'], '--periods', str(job['periods']), '--repeats', str(job['repeats']),
        '--device', job['device'], '--output', str(directory), '--cache', str(cache), '--mf6', str(mf6),
        '--k-field', 'ugly_t', '--vertical-anisotropy', str(job['ratio']), '--ugly-t-seed', str(job['seed']),
        '--ugly-t-reference-thickness', str(job['reference_thickness']),
        '--preferred-head-target', str(job['preferred_head_target']), '--mf6-timeout', str(job['mf6_timeout']),
        '--gpu-tight']
    if job['ghb'] == 'on':
        command.append('--ghb')
    if job['family'] == 'cartesian':
        command.append('--cartesian-fgmres')
    else:
        profile = 'npf_runtime_controls.json' if job['variant'] == 'convertible' and job['mode'] == 'transient' else 'npf_ghb_five_level_controls.json'
        command.extend(['--npf-controls', str(root / 'working_tests' / profile)])
        if job['mode'] == 'transient':
            command.append('--npf-history')
    return command


def source_identity(*, root):
    tracked = subprocess.check_output(args=['git', '-C', str(root), 'ls-files', 'DARCY_WARP_PACKAGE', 'working_tests'], text=True)
    paths = [root / name for name in tracked.splitlines() if name.endswith('.py')]
    paths.extend([root / 'working_tests/npf_runtime_controls.json', root / 'working_tests/npf_ghb_five_level_controls.json'])
    # Include the runner itself before its first git commit as well.
    paths.extend([Path(__file__).resolve(), root / 'working_tests/run_3d_sanity.py'])
    return {str(path.relative_to(root)): digest_file(path=path) for path in sorted(set(paths))}


def job_summary(*, record):
    result_path = Path(record['output']) / 'results.json'
    summary = dict(record['job'], fingerprint=record['fingerprint'], returncode=record['returncode'],
        sanity_passed=False, preferred_head_target_met=False, gpu_seconds=None, mf6_seconds=None,
        speedup=None, max_head_error_m=None, max_physical_head_rms_m=None,
        native_reference_accepted=None, error=record.get('error'), output=record['output'])
    if not result_path.is_file():
        return summary
    result = json.loads(s=result_path.read_text())
    rows = result.get('summary', [])
    cases = result.get('cases', [])
    if len(rows) == 1:
        row = rows[0]
        summary.update(sanity_passed=bool(record['returncode'] == 0 and result.get('passed')),
            preferred_head_target_met=row['preferred_head_target_met'],
            gpu_seconds=row['median_public_solve_seconds'], mf6_seconds=row['mf6_single_reference_run_seconds'],
            max_head_error_m=row['worst_head_max_m'])
        if summary['sanity_passed']:
            summary['speedup'] = summary['mf6_seconds'] / summary['gpu_seconds']
    if len(cases) == 1:
        case = cases[0]
        summary['native_reference_accepted'] = case.get('reference', {}).get('accepted')
        summary['error'] = case.get('error', summary['error'])
        residuals = [period['physical']['head_residual_rms'] for run in case['runs'] for period in run['periods']]
        if residuals:
            summary['max_physical_head_rms_m'] = max(residuals)
        summary['conductivity'] = case.get('conductivity')
    return summary


def run_matrix(*, args, root, environment):
    args.output = args.output.resolve()
    args.cache = args.cache.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / 'matrix.json'
    previous = json.loads(s=report_path.read_text()) if report_path.exists() else dict(attempts=[])
    sources = source_identity(root=root)
    identity = dict(sources=sources, mf6_sha256=digest_file(path=args.mf6), python=sys.version,
        thread_limits={name: environment[name] for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')})
    report = dict(schema=1, started_utc=datetime.now(tz=timezone.utc).isoformat(),
        revision=subprocess.check_output(args=['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
        identity=identity, complete=False, attempts=previous['attempts'], selected=[])
    for job in case_jobs(args=args):
        fingerprint = hashlib.sha256(json.dumps(obj=dict(job=job, identity=identity), sort_keys=True).encode()).hexdigest()
        earlier = [record for record in report['attempts'] if record['fingerprint'] == fingerprint]
        if earlier and earlier[-1].get('complete') and not (args.retry_failed and earlier[-1]['returncode'] != 0):
            record = earlier[-1]
            if record.get('results_sha256') and digest_file(path=Path(record['output']) / 'results.json') != record['results_sha256']:
                raise ValueError('Saved job results changed; refusing reuse')
            print(f'Reusing {Path(record["output"]).name}', flush=True)
        else:
            name = f'{job["size"]}_va{job["ratio"]:g}_{job["ghb"]}_{job["family"]}_{job["variant"]}_{job["mode"]}'
            name += f'_{fingerprint[:12]}_attempt{len(earlier) + 1}'
            directory = args.output / name
            record = dict(job=job, fingerprint=fingerprint, output=str(directory), complete=False)
            report['attempts'].append(record)
            save_json(path=report_path, value=report)
            command = job_command(job=job, root=root, directory=directory, cache=args.cache, mf6=args.mf6)
            record['command'] = command
            print(f'{datetime.now(tz=timezone.utc).isoformat()} START {name}', flush=True)
            with (args.output / (name + '.log')).open(mode='w') as stream:
                process = subprocess.run(args=command, cwd=root, env=environment, stdout=stream,
                    stderr=subprocess.STDOUT, check=False)
            record.update(complete=True, returncode=process.returncode)
            result_path = directory / 'results.json'
            if result_path.is_file():
                record['results_sha256'] = digest_file(path=result_path)
            else:
                record['error'] = 'No results.json; inspect retained job log'
            print(f'FINISH {name}: returncode={process.returncode}', flush=True)
        report['selected'].append(job_summary(record=record))
        save_json(path=report_path, value=report)
    report.update(complete=True, sanity_passed=all(row['sanity_passed'] for row in report['selected']),
        preferred_head_target_met=all(row['preferred_head_target_met'] for row in report['selected']))
    save_json(path=report_path, value=report)
    rows = [{key: value for key, value in row.items() if key != 'conductivity'} for row in report['selected']]
    with (args.output / 'summary.csv').open(mode='w', newline='') as stream:
        writer = csv.DictWriter(f=stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return 0 if report['sanity_passed'] else 1


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    defaults = dict(sizes=[500], ratios=[1., 10., 100.], ghb_modes=['off', 'on'],
        families=['cartesian', 'npf'], variants=['confined', 'convertible'], modes=['steady', 'transient'],
        periods=20, repeats=3, seed=42, reference_thickness=100., preferred_head_target=1e-6,
        mf6_timeout=1800., device='cuda:0', mf6=Path('/bin/modflow/mf6'),
        output=Path('benchmark_results/ugly_t_3d/matrix'), cache=Path('benchmark_results/ugly_t_3d/mf6_cache'))
    environment = dict(os.environ, DARCY_FLOAT='float64', OPENBLAS_NUM_THREADS='1',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MPLBACKEND='Agg', PYTHONPATH=str(root) + ':' + str(root / 'working_tests'))
    arguments = parse_arguments(defaults=defaults)
    raise SystemExit(run_matrix(args=arguments, root=root, environment=environment))

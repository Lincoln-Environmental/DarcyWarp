"""Diagnose one missing confined reference with CG and ILU(0).

Original MF6 convergence tolerances and saved GPU measurements are retained.
This command is separate from the original benchmark profile.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import os
import shutil
import sys


def select_confined_case(*, baseline, selection):
    if not baseline['complete']:
        raise ValueError('Baseline must be complete')
    selected = [row for row in baseline['selected']
                if all(row[key] == value for key, value in selection.items())]
    if len(selected) != 1:
        raise ValueError('Select exactly one baseline case')
    row = selected[0]
    if row['variant'] != 'confined':
        raise ValueError('CG diagnostic requires a confined problem')
    if not row['gpu_physical_accepted']:
        raise ValueError('GPU physical checks must pass before a reference diagnostic')
    if row['native_reference_accepted']:
        raise ValueError('This command diagnoses missing or invalid native references only')
    return selected


def save_snapshot(*, source, destination):
    if destination.exists():
        if destination.read_bytes() != source.read_bytes():
            raise ValueError('Diagnostic source changed; choose a new output directory')
    else:
        shutil.copyfile(src=source, dst=destination)


def run_confined_diagnostic(*, args, selection, overrides, tolerance_keys,
                           roundoff_ulps, flux_floor):
    from working_tests import diagnose_3d_mf6_references as diagnostic
    baseline = json.loads(s=args.matrix.read_text())
    selected = select_confined_case(baseline=baseline, selection=selection)
    args.output.mkdir(parents=True, exist_ok=True)
    subset_path = args.output / 'baseline_subset.json'
    subset = dict(baseline, selected=selected,
        subset_of_sha256=diagnostic.file_digest(path=args.matrix),
        scope='One missing confined reference; original baseline retained')
    save_snapshot(source=Path(diagnostic.__file__), destination=args.output / 'driver_used.py')
    save_snapshot(source=Path(__file__), destination=args.output / 'launcher_used.py')
    diagnostic.write_json(path=subset_path, value=subset)
    forwarded = SimpleNamespace(root=args.root, matrix=subset_path, output=args.output,
        mf6=args.mf6, timeout=args.timeout, expected_cases=1,
        progress_seconds=args.progress_seconds, retry_failed=False)
    return diagnostic.run_diagnostics(args=forwarded, overrides=overrides,
        tolerance_keys=tolerance_keys, roundoff_ulps=roundoff_ulps, flux_floor=flux_floor)


def parse_arguments(*, defaults):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=defaults['root'])
    parser.add_argument('--matrix', type=Path, default=defaults['matrix'])
    parser.add_argument('--output', type=Path, default=defaults['output'])
    parser.add_argument('--mf6', type=Path, default=defaults['mf6'])
    parser.add_argument('--size', type=int, default=defaults['size'])
    parser.add_argument('--ratio', type=float, default=defaults['ratio'])
    parser.add_argument('--ghb', choices=['off', 'on'], default=defaults['ghb'])
    parser.add_argument('--family', choices=['cartesian', 'npf'], default=defaults['family'])
    parser.add_argument('--mode', choices=['steady', 'transient'], default=defaults['mode'])
    parser.add_argument('--timeout', type=float, default=defaults['timeout'])
    parser.add_argument('--progress-seconds', type=float, default=defaults['progress_seconds'])
    return parser.parse_args()


if __name__ == "__main__":
    root = Path.cwd()
    defaults = dict(root=root, matrix=Path('benchmark_results/ugly_t_3d/large_500_sanity/matrix.json'),
        output=Path('benchmark_results/ugly_t_3d/mf6_confined_cg_diagnostic'),
        mf6=Path('/bin/modflow/mf6'), size=500, ratio=1., ghb='off', family='cartesian',
        mode='transient', timeout=1200., progress_seconds=30.)
    overrides = dict(complexity='SIMPLE', under_relaxation='NONE', backtracking_number=0,
        linear_acceleration='CG', preconditioner_levels=0,
        preconditioner_drop_tolerance=0., number_orthogonalizations=0, relaxation_factor=0.)
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
    selection = dict(size=arguments.size, ratio=arguments.ratio, ghb=arguments.ghb,
                     family=arguments.family, mode=arguments.mode, variant='confined')
    raise SystemExit(run_confined_diagnostic(args=arguments, selection=selection,
        overrides=overrides, tolerance_keys=tolerance_keys, roundoff_ulps=roundoff_ulps,
        flux_floor=flux_floor))

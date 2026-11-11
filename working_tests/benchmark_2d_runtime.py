"""Compare 2D transient runtime controls against validated saved MF6 histories.

Run with PYTHONPATH=.:working_tests and DARCY_FLOAT=float64. A JSON controls
file maps variant names to solve-control overrides, or to a record containing
solve_controls and supports_ghb. Each variant receives a
separate warmup, then repetitions run in alternating order on fresh models.
The solver's public history timer includes its internal setup and diagnostic
output; external model construction and the independent audit are separate.
"""
from pathlib import Path
from time import perf_counter
import argparse
import hashlib
import json
import statistics
import subprocess

import numpy as np
import warp as wp

from working_tests.transient_artifacts import (
    validate_transient_artifact, spatial_fields_from_artifact,
    select_artifact_warm_start,
)
from working_tests.transient_replay_support import run_warp_transient_replay
from working_tests.transient_replay_settings import HEAD_ACCURACY_CRITERIA
from working_tests.transient_replay_metrics import compare_transient
from working_tests.transient_replay_mass_balance import (
    compute_replay_mass_balance, annotate_mass_balance_classification,
)
from DARCY_WARP_PACKAGE.solvers.transient_config import default_solve_controls
from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver


def timed_solver_factory(*, timing, formulation, device):
    """Time the entire public API, including validation and session creation."""
    class TimedSolver(WarpDarcySolver):
        def solve_transient_2d_unconfined(self, **kwargs):
            wp.synchronize_device(device=device)
            started = perf_counter()
            result = super().solve_transient_2d_unconfined(**kwargs)
            wp.synchronize_device(device=device)
            timing['seconds'] += perf_counter() - started
            return result

        def solve(self, **kwargs):
            if formulation != 'confined':
                return super().solve(**kwargs)
            wp.synchronize_device(device=device)
            started = perf_counter()
            result = super().solve(**kwargs)
            wp.synchronize_device(device=device)
            timing['seconds'] += perf_counter() - started
            return result

    return TimedSolver


def evaluate_heads(*, comparison, finite, gates):
    rows = comparison['per_period']
    final = comparison['final']
    return bool(finite
        and final['rmse'] <= gates['final_rmse_max']
        and final['max_abs_diff'] <= gates['final_max_abs_diff_max']
        and max(row['rmse'] for row in rows) <= gates['worst_period_rmse_max']
        and max(row['max_abs_diff'] for row in rows) <= gates['worst_period_max_abs_diff_max']
        and min(row['percent_within_0_01m'] for row in rows)
            >= gates['all_period_percent_within_0_01m_min'])


def measure(*, artifact, spatial, initial, mode, controls, device, audit, implementation):
    started = perf_counter()
    formulation = str(artifact['formulation'])
    timing = {'seconds': 0.}

    def solver_factory():
        return timed_solver_factory(timing=timing, formulation=formulation, device=device)

    result = run_warp_transient_replay(
        spatial=spatial, recharge_rates=artifact['recharge_rates'],
        sy=float(artifact['sy']), ss=float(artifact['ss']), dt=float(artifact['dt_days']),
        device=device, warm_start_mode=mode, warm_start_head=initial,
        solve_controls=controls, formulation=formulation,
        solver_backend='confined_kcycle' if formulation == 'confined' else 'unconfined_picard_kcycle',
        implementation=implementation,
        _warp_solver_class_fn=solver_factory,
    )
    wp.synchronize_device(device=device)
    wrapper_seconds = perf_counter() - started
    infos = result['period_infos']
    active = np.asarray(spatial['active']) != 0
    heads = result['heads_per_period']
    finite = bool(np.isfinite(heads[:, active]).all())
    comparison = compare_transient(warp_result=result,
        mf6_heads_per_period=artifact['heads_per_period'],
        mf6_heads_final=artifact['heads_final'], active=spatial['active'])
    strict = all(bool(row.get('strict_picard_convergence_passed', row.get('converged', False))) for row in infos)
    head_passed = evaluate_heads(comparison=comparison, finite=finite, gates=HEAD_ACCURACY_CRITERIA)
    budget = None
    audit_seconds = 0.
    if audit:
        started = perf_counter()
        budget = compute_replay_mass_balance(spatial=spatial,
            recharge_rates=artifact['recharge_rates'], sy=float(artifact['sy']),
            ss=float(artifact['ss']), dt=float(artifact['dt_days']),
            formulation=formulation, unconfined_storage_mode=result['unconfined_storage_mode'],
            warp_result=result, min_sat=controls['min_saturated_thickness'])
        annotate_mass_balance_classification(mass_balance=budget)
        audit_seconds = perf_counter() - started
    row = dict(public_seconds=timing['seconds'], reported_history_seconds=result['total_time'], wrapper_seconds=wrapper_seconds,
        audit_seconds=audit_seconds, strict=strict, head_passed=head_passed,
        passed=bool(strict and head_passed and (budget is None or budget['mass_balance_passed'])),
        outer=sum(info.get('outer_iterations', 0) for info in infos),
        cycles=sum(info.get('total_inner_kcycles', info.get('n_cycles_used', 0)) for info in infos),
        startup_cycles=sum(info.get('startup_inner_kcycles', 0) for info in infos),
        fallback_count=sum(info.get('legacy_dh_fallback_count', 0) for info in infos),
        retries=sum(info.get('adaptive_dt_retry_count', 0) for info in infos),
        comparison=comparison,
        max_budget_percent=None if budget is None else budget['max_abs_percent_discrepancy'],
        budget_passed=None if budget is None else budget['mass_balance_passed'],
        period_seconds=np.asarray(result['period_times']).tolist(),
        final_residuals=[info.get('final_head_residual_rms') for info in infos],
        max_head_changes=[info.get('final_max_abs_head_change') for info in infos])
    return row, heads


def write_report(*, path, report):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(target=path)


def variant_controls(*, defaults, variants, use_ghb):
    controls = {}
    for name, specification in variants.items():
        if use_ghb and not specification.get('supports_ghb', True):
            raise ValueError(f'{name} is not qualified for GHB; use production controls')
        overrides = specification.get('solve_controls', specification)
        controls[name] = dict(defaults, **overrides)
    return controls


def benchmark(*, artifact_path, output, variants, repeats, device, audit, implementation, baseline_heads, default_overrides):
    artifact = validate_transient_artifact(path=artifact_path)
    spatial = spatial_fields_from_artifact(artifact=artifact)
    initial, mode = select_artifact_warm_start(artifact=artifact, spatial=spatial,
        warm_start_mode=str(artifact['warm_start_mode']))
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    defaults = default_solve_controls()
    defaults.update(default_overrides)
    controls = variant_controls(defaults=defaults, variants=variants,
        use_ghb=bool(np.any(spatial.get('ghb_mask', False))))
    report = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        artifact=str(artifact_path), case_fingerprint=str(artifact['case_fingerprint']),
        artifact_sha256=hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        source_sha256={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((root/'DARCY_WARP_PACKAGE').rglob('*.py'))},
        device=device, warp_version=wp.__version__, warm_start=mode,
        mf6_engine_seconds=float(artifact.get('engine_time', np.nan)),
        controls=controls, variant_specifications=variants, head_gates=HEAD_ACCURACY_CRITERIA,
        audit_enabled=audit, implementation=implementation,
        timing='Synchronized entire public API call, including input validation and session creation. External model construction and independent audit excluded; original solver-reported timer retained separately.',
        rows=[], summary={})
    reference = None
    if baseline_heads is not None:
        reference = np.load(file=baseline_heads)['heads']
    print(f'Validated {artifact_path}; warm start {mode}', flush=True)
    for repeat in range(repeats+1):
        names = list(controls)
        if repeat % 2:
            names.reverse()
        for name in names:
            row, heads = measure(artifact=artifact, spatial=spatial, initial=initial, mode=mode,
                controls=controls[name], device=device, audit=audit, implementation=implementation)
            row.update(variant=name, repeat=repeat, warmup=repeat == 0)
            if reference is not None:
                if heads.shape != reference.shape:
                    raise ValueError('Baseline head shape differs')
                row['baseline_max_head_difference'] = float(np.max(np.abs(heads-reference)))
            report['rows'].append(row)
            np.savez(file=output/f'{name}_latest_heads.npz', heads=heads)
            print(json.dumps({key:value for key,value in row.items()
                if key not in ('comparison', 'period_seconds', 'final_residuals', 'max_head_changes')}), flush=True)
            write_report(path=output/'results.json', report=report)
    for name in controls:
        rows = [row for row in report['rows'] if row['variant'] == name and not row['warmup']]
        report['summary'][name] = dict(
            median_public_seconds=statistics.median(row['public_seconds'] for row in rows),
            passed=all(row['passed'] for row in report['rows'] if row['variant'] == name),
            worst_head_error=max(max(period['max_abs_diff'] for period in row['comparison']['per_period']) for row in rows))
    write_report(path=output/'results.json', report=report)
    print(json.dumps(report['summary']), flush=True)
    return all(row['passed'] for row in report['rows'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--controls', type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--implementation', choices=('fast', 'classic'), default='fast')
    parser.add_argument('--baseline-heads', type=Path)
    args = parser.parse_args()
    default_overrides = dict(save_transient_diagnostics=True, transient_mixed_precision_enabled=True)
    variants = {'production': {}} if args.controls is None else json.loads(args.controls.read_text())
    if args.repeats < 1 or not variants:
        raise ValueError('At least one repeat and variant are required')
    wp.init()
    passed = benchmark(artifact_path=args.artifact, output=args.output, variants=variants,
        repeats=args.repeats, device=args.device, audit=args.audit,
        implementation=args.implementation, baseline_heads=args.baseline_heads,
        default_overrides=default_overrides)
    raise SystemExit(0 if passed else 1)

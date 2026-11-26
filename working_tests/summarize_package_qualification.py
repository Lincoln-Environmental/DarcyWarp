"""Assemble matched physical comparisons from owned accepted output files."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np


def compare_2d_heads(*, numerical, native, transient, target):
    """Report endpoint differences without changing either solver's tolerances.

    :param numerical: Owned accepted numerical output directory.
    :param native: Exact-input pinned native output directory.
    :param transient: Compare every transient endpoint rather than two steady endpoints.
    :param target: Preferred head parity target, independently specified in metres.
    :return: Measured errors and explicit preferred-target status.
    """
    with np.load(numerical / 'heads.npz') as computed, np.load(native / 'reference_outputs.npz') as reference:
        native_endpoints = len(reference['heads'])
        if transient or 'heads' in computed:
            pairs = [(index, head) for index, head in enumerate(computed['heads'])]
        else:
            pairs = [(0, computed['cold']), (len(reference['heads']) - 1, computed['calibration'])]
        rows = []
        for index, head in pairs:
            difference = head - reference['heads'][index, 0]
            rows.append(dict(index=index, maximum_m=float(np.max(np.abs(difference))),
                rms_m=float(np.sqrt(np.mean(difference * difference)))))
    result = dict(periods=rows, compared_endpoints=len(rows), native_endpoints=native_endpoints,
        preferred_head_target_m=target,
        preferred_target_met=all(row['maximum_m'] <= target for row in rows))
    (numerical / 'mf6_exact_head_comparison.json').write_text(json.dumps(result, indent=2))
    return result


def compare_confined_storage(*, numerical, native, previous, coefficient, dt, fixed_columns):
    """Independently calculate physical Ss accumulation from accepted heads.

    :param numerical: Owned accepted resident-history output directory.
    :param native: Exact-input pinned confined history output directory.
    :param previous: Actual initial physical head field.
    :param coefficient: Specific storage times the physical cell volume.
    :param dt: Actual matched timestep duration.
    :param fixed_columns: Prescribed-head columns with no physical STO equation.
    :return: Per-cell storage-rate and integrated accumulation differences.
    """
    rows = []
    with np.load(numerical / 'accepted_heads.npz') as computed, np.load(native / 'reference_outputs.npz') as reference:
        for index in range(len(reference['heads'])):
            head = computed[f'head_{index}']
            accumulation = coefficient * (head - previous) / dt
            accumulation[:, :, fixed_columns] = 0.
            native_rate = -reference[f'STO_SS_{index}'].reshape(head.shape)
            error = accumulation - native_rate
            rows.append(dict(index=index, maximum_rate_error_m3_day=float(np.max(np.abs(error))),
                net_volume_error_m3=float(np.sum(error) * dt),
                accepted_accumulation_m3=float(np.sum(accumulation) * dt),
                native_accumulation_m3=float(np.sum(native_rate) * dt)))
            previous = head
    result = dict(periods=rows, maximum_rate_error_m3_day=max(row['maximum_rate_error_m3_day'] for row in rows),
        total_accumulation_error_m3=sum(row['net_volume_error_m3'] for row in rows))
    (numerical / 'mf6_exact_independent_storage.json').write_text(json.dumps(result, indent=2))
    return result


def collect_comparisons(*, evidence, native_3d, native_2d, controls, shapes, storage_coefficient, fixed_columns):
    """Check available exact references and explicitly list outstanding outputs.

    :param evidence: Numerical qualification and compact evidence root.
    :param native_3d: Owned exact-input native 3D workspace root.
    :param native_2d: Owned exact-input native 2D workspace root.
    :param controls: Independently specified physical and comparison controls.
    :param shapes: Exact numerical grid shapes keyed by size label.
    :param storage_coefficient: Physical Ss times cell volume.
    :param fixed_columns: Prescribed-head columns.
    :return: Completed comparisons and explicitly outstanding references.
    :raises AssertionError: If strict native or accepted source-volume comparisons fail.
    """
    from compare_package_qualification import compare_outputs
    from compare_package_source_volumes import compare_sources
    result = dict(comparisons={}, pending=[])
    for label, shape in shapes.items():
        for mode in ('steady', 'history'):
            native = native_3d / 'strict' / label / mode
            if not (native / 'reference.json').exists():
                result['pending'].append(str(native))
                continue
            profiles = ('history',) if mode == 'history' else ('saturated_fp64', 'current_fp64', 'qualified_current_fp32')
            for profile in profiles:
                numerical = evidence / 'production_3d' / label
                numerical = numerical / 'history' if mode == 'history' else numerical / 'packages' / profile
                key = f'3d/{label}/{mode}/{profile}'
                result['comparisons'][key] = compare_outputs(gpu_directory=numerical, native_directory=native,
                    output=numerical / 'mf6_exact_comparison.json', head_gate=controls['head_gate_m'],
                    rate_gate=controls['entry_rate_gate_m3_day'], history=mode == 'history')
                compare_sources(numerical=numerical / 'result.json', native=native / 'reference.json',
                    output=numerical / 'mf6_exact_source_comparison.json', transient=mode == 'history',
                    dimensionality=3, summation_ulps=controls['source_summation_ulps'])
                if mode == 'history':
                    result['comparisons'][key]['storage'] = compare_confined_storage(numerical=numerical,
                        native=native, previous=np.full(shape, controls['initial_head']),
                        coefficient=storage_coefficient, dt=controls['history_dt_days'], fixed_columns=fixed_columns)
        for mode in ('steady', 'mixed', 'transient', 'picard'):
            native = native_2d / label / mode
            if not (native / 'reference.json').exists():
                result['pending'].append(str(native))
                continue
            numerical = evidence / 'production_2d' / label / mode
            if mode in ('steady', 'mixed'):
                numerical = numerical / 'wel'
            transient = mode in ('transient', 'picard')
            key = f'2d/{label}/{mode}'
            result['comparisons'][key] = compare_2d_heads(numerical=numerical, native=native,
                transient=transient, target=controls['head_gate_m'])
            compare_sources(numerical=numerical / 'result.json', native=native / 'reference.json',
                output=numerical / 'mf6_exact_source_comparison.json', transient=transient,
                dimensionality=2, summation_ulps=controls['source_summation_ulps'])
    for origin, category in ((native_3d, 'native_3d_exact'), (native_2d, 'native_2d_exact')):
        for path in origin.rglob('reference.json'):
            destination = evidence / category / path.relative_to(origin)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src=path, dst=destination)
    (evidence / 'exact_reference_summary.json').write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-3d', type=Path, default=Path('/tmp/darcywarp-package-mf6-exact'))
    parser.add_argument('--native-2d', type=Path, default=Path('/tmp/darcywarp-package-2d-mf6-exact'))
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    evidence = root / 'benchmark_results/package_audit_repair'
    controls = json.loads((root / 'working_tests/wel_regional_qualification_controls.json').read_text())
    controls['source_summation_ulps'] = 64.
    shapes = {'100k': (10, 112, 90), '1M': (5, 450, 450)}
    storage_coefficient = controls['specific_storage_per_m'] * 40. * 40. * 10.
    fixed_columns = [0, -1]
    result = collect_comparisons(evidence=evidence, native_3d=arguments.native_3d,
        native_2d=arguments.native_2d, controls=controls, shapes=shapes,
        storage_coefficient=storage_coefficient, fixed_columns=fixed_columns)
    print(json.dumps(dict(completed=len(result['comparisons']), pending=result['pending']), indent=2))

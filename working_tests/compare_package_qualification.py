"""Compare accepted numerical outputs with independently executed pinned MF6."""
import argparse
import json
from math import fsum
from pathlib import Path

import numpy as np


def compare_accepted_boundary_volumes(*, accepted_totals, native_periods, entry_counts, rate_gate):
    """Compare the driver's accepted volume ledger, independently of rate errors.

    :param accepted_totals: Actual integrated DRN/RIV ledger returned by the driver.
    :param native_periods: Matched MF6 per-period CBC volume ledgers.
    :param entry_counts: Independently counted entries of each package per period.
    :param rate_gate: Prespecified absolute per-entry rate error gate.
    :return: Actual accepted volume errors and conservative integration bounds.
    :raises AssertionError: If an accepted integrated volume fails its bound.
    """
    result = {}
    for package in ('DRN', 'RIV'):
        if package not in entry_counts:
            continue
        expected = {name: fsum(period['volumes'][package][name] for period in native_periods)
                    for name in ('inflow', 'outflow', 'net')}
        actual = accepted_totals[package.lower()]
        errors = {name: float(actual[name] - expected[name]) for name in expected}
        # Sign splitting is Lipschitz: N entry error bounds integrated over
        # the physical time grid bound gross inflow, gross outflow and net.
        counts = entry_counts[package]
        if len(counts) != len(native_periods):
            raise ValueError(f'{package} entry-count history does not match the period history')
        bound = fsum(count * period['dt'] * rate_gate
                     for count, period in zip(counts, native_periods, strict=True))
        assert all(abs(value) <= bound for value in errors.values()), (package, errors, bound)
        result[package] = dict(accepted=actual, native=expected, errors_m3=errors,
                               entry_integrated_bound_m3=bound)
    return result


def _compare_complete_outputs(*, gpu_directory, native_directory, output, head_gate, rate_gate, history):
    """Check every matched accepted endpoint and individual switching entry.

    :param gpu_directory: Accepted GPU arrays and actual execution diagnostics.
    :param native_directory: Pinned native arrays and convergence evidence.
    :param output: Compact comparison destination.
    :param head_gate: Independently selected head difference limit in metres.
    :param rate_gate: Independent absolute entry-rate difference limit.
    :param history: Compare every accepted transient endpoint.
    :return: Head, entry rate, gross budget and volume comparison evidence.
    :raises AssertionError: If the prescribed numerical comparison fails.
    """
    native_info = json.loads((native_directory / 'reference.json').read_text())
    gpu_info = json.loads((gpu_directory / 'result.json').read_text())
    periods = gpu_info['warm'][-1]['periods'] if history else [gpu_info['cold'], *gpu_info['warm_start_calibration']]
    comparisons = []
    with np.load(gpu_directory / 'accepted_heads.npz') as numerical, np.load(native_directory / 'reference_outputs.npz') as native:
        assert len(periods) == len(native['heads']) == len(native_info['periods'])
        assert all(period.get('converged') for period in periods)
        np.testing.assert_allclose(native['times'], np.cumsum([
            row['dt'] for row in native_info['periods']]), rtol=0., atol=1.e-12)
        if history:
            numerical_dt = [period.get('dt', gpu_info['controls'].get('history_dt_days')) for period in periods]
            assert all(value is not None for value in numerical_dt)
            np.testing.assert_allclose(numerical_dt,
                [row['dt'] for row in native_info['periods']], rtol=0., atol=1.e-12)
        for index, period in enumerate(periods):
            assert period['converged'], (index, period.get('stop_reason'))
            numerical_head = numerical[f'head_{index}']
            native_head = native['heads'][index]
            assert native_head.ndim == 3, 'native head fields must match the three-dimensional domain'
            assert numerical_head.shape == native_head.shape, (
                f'head_{index} shape {numerical_head.shape} does not match '
                f'physical domain {native_head.shape}')
            difference = numerical_head - native_head
            row = dict(index=index, head_max_m=float(np.max(np.abs(difference))),
                head_rms_m=float(np.sqrt(np.mean(difference * difference))), entry_rates={}, budgets={})
            expected_packages = set(native_info['packages'])
            assert set(native_info['periods'][index]['packages']) == expected_packages
            package_counts = {}
            for package in expected_packages:
                native_key = f'{package}_{index}'
                assert native_key in native, f'missing native {native_key} array'
                native_rates = native[native_key]
                assert native_rates.ndim == 1, f'native {native_key} entry rates must be one-dimensional'
                package_counts[package] = int(native_rates.shape[0])
            for package in ('DRN', 'RIV'):
                if package not in expected_packages:
                    continue
                key = f'{package.lower()}_{index}'
                assert key in numerical, f'missing numerical {key} entry array'
                physical = numerical[key]
                assert physical.ndim == 1, f'numerical {key} entry rates must be one-dimensional'
                assert physical.shape[0] == package_counts[package]
                error = physical + native[f'{package}_{index}']
                row['entry_rates'][package] = dict(max_absolute_error_m3_day=float(np.max(np.abs(error), initial=0.)),
                    count=int(error.size), gross_inflow=float(np.sum(np.maximum(-physical, 0.))),
                    gross_outflow=float(np.sum(np.maximum(physical, 0.))))
            source = period.get('source_budgets') or {}
            acceptance = period.get('physical_acceptance') or {}
            raw_chd = period.get('chd_budget') or acceptance
            chd = ({name: raw_chd.get('chd_' + name, raw_chd.get(name))
                for name in ('inflow', 'outflow')} if raw_chd else None)
            if chd is not None and all(chd.get(name) is not None for name in ('inflow', 'outflow')):
                chd = dict(inflow=chd['inflow'], outflow=chd['outflow'],
                    net=chd['inflow'] - chd['outflow'])
            for package, numerical_budget in [('DRN', period.get('drn_budget')), ('RIV', period.get('riv_budget')),
                ('GHB', period.get('ghb_budget')), ('WEL', source.get('wel')),
                ('RCH', source.get('recharge')), ('CHD', chd)]:
                if package not in expected_packages:
                    continue
                assert numerical_budget is not None, f'missing numerical {package} budget'
                if package == 'GHB':
                    numerical_budget = {name: numerical_budget['ghb_' + name] for name in ('inflow', 'outflow', 'net')}
                reference = native_info['periods'][index]['packages'][package]
                errors = {name: float(numerical_budget[name] - reference[name]) for name in ('inflow', 'outflow', 'net')}
                bound = max(1, package_counts[package]) * rate_gate
                row['budgets'][package] = dict(rate_errors_m3_day=errors,
                    volume_errors_m3={name: value * native_info['periods'][index]['dt'] for name, value in errors.items()},
                    rate_bound_m3_day=bound,
                    passed=all(abs(value) <= bound for value in errors.values()))
            assert set(row['budgets']) == expected_packages
            row['passed'] = row['head_max_m'] <= head_gate and all(
                item['max_absolute_error_m3_day'] <= rate_gate for item in row['entry_rates'].values()) and all(
                item['passed'] for item in row['budgets'].values())
            comparisons.append(row)
    result = dict(status='complete', native_version=native_info['version'], native_sha256=native_info['sha256'],
        gates=dict(head_m=head_gate, entry_rate_m3_day=rate_gate), periods=comparisons,
        passed=all(row['passed'] for row in comparisons))
    if history:
        result['accepted_boundary_volume_totals'] = compare_accepted_boundary_volumes(
            accepted_totals=gpu_info['warm'][-1]['accepted_exchange_volume_totals'],
            native_periods=native_info['periods'],
            entry_counts={name: [row['entry_rates'][name]['count'] for row in comparisons]
                          for name in comparisons[0]['entry_rates']},
            rate_gate=rate_gate)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    assert result['passed'], result
    return result


def compare_outputs(*, gpu_directory, native_directory, output, head_gate, rate_gate, history):
    """Write an explicit complete, incomplete or failed comparison verdict.

    Missing arrays/files are incomplete evidence. Numerical disagreement or
    inconsistent accepted metadata is failed evidence. Both reject the caller
    after leaving a machine-readable receipt at ``output``.
    """
    try:
        return _compare_complete_outputs(gpu_directory=gpu_directory,
            native_directory=native_directory, output=output, head_gate=head_gate,
            rate_gate=rate_gate, history=history)
    except (AssertionError, KeyError, OSError, ValueError) as error:
        message = str(error)
        incomplete = isinstance(error, (FileNotFoundError, KeyError)) or 'missing' in message.lower()
        if output.exists():
            try:
                result = json.loads(output.read_text())
            except (OSError, json.JSONDecodeError):
                result = {}
        else:
            result = {}
        result.update(status='incomplete' if incomplete else 'failed', passed=False,
            error_type=type(error).__name__, error=message)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=Path, required=True)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--history', action='store_true')
    parser.add_argument('--head-gate', type=float, default=1.e-6)
    parser.add_argument('--rate-gate', type=float, default=1.e-5)
    arguments = parser.parse_args()
    compare_outputs(gpu_directory=arguments.gpu, native_directory=arguments.native, output=arguments.output,
        head_gate=arguments.head_gate, rate_gate=arguments.rate_gate, history=arguments.history)

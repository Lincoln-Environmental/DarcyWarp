"""Compare accepted numerical outputs with independently executed pinned MF6."""
import argparse
import json
from pathlib import Path

import numpy as np


def compare_outputs(*, gpu_directory, native_directory, output, head_gate, rate_gate, history):
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
        assert len(periods) == len(native['heads'])
        for index, period in enumerate(periods):
            difference = numerical[f'head_{index}'] - native['heads'][index]
            row = dict(index=index, head_max_m=float(np.max(np.abs(difference))),
                head_rms_m=float(np.sqrt(np.mean(difference * difference))), entry_rates={}, budgets={})
            for package in ('DRN', 'RIV'):
                key = f'{package.lower()}_{index}'
                if key not in numerical:
                    continue
                physical = numerical[key]
                error = physical + native[f'{package}_{index}']
                row['entry_rates'][package] = dict(max_absolute_error_m3_day=float(np.max(np.abs(error), initial=0.)),
                    count=int(error.size), gross_inflow=float(np.sum(np.maximum(-physical, 0.))),
                    gross_outflow=float(np.sum(np.maximum(physical, 0.))))
            source = period.get('source_budgets') or {}
            for package, numerical_budget in [('DRN', period.get('drn_budget')), ('RIV', period.get('riv_budget')),
                ('GHB', period.get('ghb_budget')), ('WEL', source.get('wel')), ('RCH', source.get('recharge'))]:
                if numerical_budget is None or package not in native_info['periods'][index]['packages']:
                    continue
                if package == 'GHB':
                    numerical_budget = {name: numerical_budget['ghb_' + name] for name in ('inflow', 'outflow', 'net')}
                reference = native_info['periods'][index]['packages'][package]
                errors = {name: float(numerical_budget[name] - reference[name]) for name in ('inflow', 'outflow', 'net')}
                row['budgets'][package] = dict(rate_errors_m3_day=errors,
                    volume_errors_m3={name: value * native_info['periods'][index]['dt'] for name, value in errors.items()})
            row['passed'] = row['head_max_m'] <= head_gate and all(
                item['max_absolute_error_m3_day'] <= rate_gate for item in row['entry_rates'].values())
            comparisons.append(row)
    result = dict(native_version=native_info['version'], native_sha256=native_info['sha256'],
        gates=dict(head_m=head_gate, entry_rate_m3_day=rate_gate), periods=comparisons,
        passed=all(row['passed'] for row in comparisons))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    assert result['passed'], result
    return result


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

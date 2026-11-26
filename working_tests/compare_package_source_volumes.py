"""Check accepted gross WEL/recharge rates and volumes against native CBC ledgers."""
import argparse
import json
from pathlib import Path

import numpy as np


def compare_sources(*, numerical, native, output, transient, dimensionality, summation_ulps):
    """Use accepted metadata, not restored current model forcing, for every period.

    :param numerical: Saved accepted numerical result JSON.
    :param native: Completed pinned native reference JSON.
    :param output: Compact comparison destination.
    :param transient: Compare accepted integrated physical volumes.
    :param dimensionality: Select the maintained 2D or 3D result schema.
    :param summation_ulps: Independent FP64 budget summation allowance.
    :return: Per-period gross source errors and accepted volume errors.
    :raises AssertionError: If sources or volumes do not match at summation precision.
    """
    accepted = json.loads(numerical.read_text())
    reference = json.loads(native.read_text())
    if transient:
        periods = accepted['warm'][-1]['periods']
    else:
        periods = [accepted['cold'], *accepted['calibration' if dimensionality == 2 else 'warm_start_calibration']]
    assert len(periods) == len(reference['periods'])
    rows, total_native, total_numerical = [], {}, {}
    for index, (period, target) in enumerate(zip(periods, reference['periods'])):
        assert period['converged']
        row = dict(index=index, rates={}, volumes={})
        for package, native_name in (('wel', 'WEL'), ('recharge', 'RCH')):
            actual = period['source_budgets'][package]
            expected = target['packages'][native_name]
            throughput = expected['inflow'] + expected['outflow']
            bound = summation_ulps * np.finfo(np.float64).eps * throughput
            errors = {name: float(actual[name] - expected[name]) for name in ('inflow', 'outflow', 'net')}
            row['rates'][package] = dict(errors_m3_day=errors, roundoff_bound_m3_day=bound)
            assert all(abs(value) <= bound for value in errors.values()), row
            if transient:
                expected_volume = target['volumes'][native_name]
                for name in ('inflow', 'outflow', 'net'):
                    total_native[(package, name)] = total_native.get((package, name), 0.) + expected_volume[name]
                if dimensionality == 2:
                    actual_volume = period['source_volume_budgets'][package]
                    volume_errors = {name: float(actual_volume[name] - expected_volume[name]) for name in ('inflow', 'outflow', 'net')}
                    assert all(abs(value) <= bound * target['dt'] for value in volume_errors.values()), row
                    row['volumes'][package] = volume_errors
                else:
                    # The resident benchmark retains the production driver's
                    # independently accumulated accepted volume ledger below.
                    for name in ('inflow', 'outflow', 'net'):
                        total_numerical[(package, name)] = accepted['warm'][-1]['accepted_exchange_volume_totals']['sources'][package][name]
        rows.append(row)
    totals = {}
    if transient and dimensionality == 3:
        for package in ('wel', 'recharge'):
            errors = {name: total_numerical[(package, name)] - total_native[(package, name)] for name in ('inflow', 'outflow', 'net')}
            bound = summation_ulps * np.finfo(np.float64).eps * (
                total_native[(package, 'inflow')] + total_native[(package, 'outflow')])
            assert all(abs(value) <= bound for value in errors.values()), (errors, bound)
            totals[package] = dict(errors_m3=errors, roundoff_bound_m3=bound)
    result = dict(passed=True, native_version=reference['version'], native_sha256=reference['sha256'],
        summation_ulps=summation_ulps, periods=rows, resident_volume_totals=totals)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--numerical', type=Path, required=True)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--transient', action='store_true')
    parser.add_argument('--dimensionality', type=int, choices=(2, 3), required=True)
    parser.add_argument('--summation-ulps', type=float, default=64.)
    arguments = parser.parse_args()
    compare_sources(numerical=arguments.numerical, native=arguments.native, output=arguments.output,
        transient=arguments.transient, dimensionality=arguments.dimensionality, summation_ulps=arguments.summation_ulps)

"""Head and rate agreement cannot substitute for accepted volume agreement."""
import json

import numpy as np
import pytest

from working_tests.compare_package_qualification import (
    compare_accepted_boundary_volumes,
    compare_outputs,
)


def write_comparison_fixture(*, root, include_entry_array, drain_outflow):
    """Write one accepted endpoint and its independent package evidence."""
    gpu = root / 'gpu'
    native = root / 'native'
    gpu.mkdir()
    native.mkdir()
    period = dict(converged=True,
        drn_budget=dict(inflow=0., outflow=drain_outflow, net=-drain_outflow),
        chd_budget=dict(inflow=0., outflow=0., net=0.))
    (gpu / 'result.json').write_text(json.dumps(dict(
        cold=period, warm_start_calibration=[], warm=[], controls={})))
    numerical = dict(head_0=np.zeros((1, 1, 1)))
    if include_entry_array:
        numerical['drn_0'] = np.array([1.])
    np.savez(gpu / 'accepted_heads.npz', **numerical)
    package_rates = dict(
        DRN=dict(inflow=0., outflow=1., net=-1.),
        CHD=dict(inflow=0., outflow=0., net=0.))
    (native / 'reference.json').write_text(json.dumps(dict(
        version='mf6-test', sha256='test', packages=['DRN', 'CHD'],
        periods=[dict(dt=1., packages=package_rates,
            volumes={name: values for name, values in package_rates.items()})])))
    np.savez(native / 'reference_outputs.npz', heads=np.zeros((1, 1, 1, 1)),
        times=np.array([1.]), DRN_0=np.array([-1.]), CHD_0=np.array([0.]))
    return gpu, native, root / 'comparison.json'


@pytest.mark.parametrize('accepted_outflow,accepted', [(6., True), (3., False)])
def test_volume_comparison_uses_driver_ledger(accepted_outflow, accepted):
    """A three-unit rate over two days is six units, even with correct heads."""
    inputs = dict(accepted_totals={'drn': dict(inflow=0., outflow=accepted_outflow, net=-accepted_outflow)},
        native_periods=[dict(dt=2., volumes={'DRN': dict(inflow=0., outflow=6., net=-6.)})],
        entry_counts={'DRN': 1}, rate_gate=1.e-8)
    if accepted:
        result = compare_accepted_boundary_volumes(**inputs)
        assert result['DRN']['errors_m3'] == dict(inflow=0., outflow=0., net=0.)
    else:
        with pytest.raises(AssertionError):
            compare_accepted_boundary_volumes(**inputs)


def test_missing_entry_array_is_recorded_as_incomplete(tmp_path):
    """Claimed DRN evidence cannot pass when its numerical entry array is absent."""
    gpu, native, output = write_comparison_fixture(
        root=tmp_path, include_entry_array=False, drain_outflow=1.)
    with pytest.raises(AssertionError, match='missing numerical drn_0'):
        compare_outputs(gpu_directory=gpu, native_directory=native, output=output,
            head_gate=1.e-6, rate_gate=1.e-5, history=False)
    assert json.loads(output.read_text())['status'] == 'incomplete'


def test_claimed_gross_budget_error_is_recorded_as_failed(tmp_path):
    """A billion-unit DRN budget error fails despite matching heads and entries."""
    gpu, native, output = write_comparison_fixture(
        root=tmp_path, include_entry_array=True, drain_outflow=1.e9)
    with pytest.raises(AssertionError):
        compare_outputs(gpu_directory=gpu, native_directory=native, output=output,
            head_gate=1.e-6, rate_gate=1.e-5, history=False)
    result = json.loads(output.read_text())
    assert result['status'] == 'failed'
    assert not result['passed']
    assert 'CHD' in result['periods'][0]['budgets']

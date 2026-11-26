"""A qualification receipt must detect errors even when a head plot looks right."""

import json

import numpy as np
import pytest

from working_tests.compare_package_qualification import compare_outputs


def drain_budget(outflow):
    return dict(inflow=0., outflow=outflow, net=-outflow)


def write_history(directory, *, numerical_heads=None, endpoint_dts=(1., 9.),
                  omit_second_rates=False, second_budget=None, accepted_volume=12.):
    """CBC inflow sign is negative for a drain. Three entries become one after day 1."""
    model = directory / 'model'
    native = directory / 'mf6'
    model.mkdir()
    native.mkdir()
    heads = np.full((2, 1, 1, 3), 4.)
    if numerical_heads is None:
        numerical_heads = heads
    numerical_arrays = dict(head_0=numerical_heads[0], head_1=numerical_heads[1], drn_0=np.ones(3))
    if not omit_second_rates:
        numerical_arrays['drn_1'] = np.ones(1)
    np.savez(model / 'accepted_heads.npz', **numerical_arrays)
    np.savez(native / 'reference_outputs.npz', heads=heads, times=[1., 10.],
        DRN_0=-np.ones(3), DRN_1=-np.ones(1))
    if second_budget is None:
        second_budget = drain_budget(1.)
    periods = [
        dict(converged=True, dt=endpoint_dts[0], drn_budget=drain_budget(3.)),
        dict(converged=True, dt=endpoint_dts[1], drn_budget=second_budget),
    ]
    history = dict(periods=periods, accepted_exchange_volume_totals=dict(drn=drain_budget(accepted_volume)))
    (model / 'result.json').write_text(json.dumps(dict(controls={}, warm=[history])))
    periods = [
        dict(dt=1., packages=dict(DRN=drain_budget(3.)), volumes=dict(DRN=drain_budget(3.))),
        dict(dt=9., packages=dict(DRN=drain_budget(1.)), volumes=dict(DRN=drain_budget(9.))),
    ]
    (native / 'reference.json').write_text(json.dumps(dict(
        version='6.6.2', sha256='fixture', packages=['DRN'], periods=periods)))
    return dict(gpu_directory=model, native_directory=native, output=directory / 'verdict.json',
                head_gate=1e-6, rate_gate=.01, history=True)


def test_entry_signs_and_unequal_period_lengths_agree_with_the_driver_ledger(tmp_path):
    result = compare_outputs(**write_history(tmp_path))
    assert result['passed']
    assert result['periods'][0]['entry_rates']['DRN']['gross_outflow'] == 3.
    assert result['periods'][1]['entry_rates']['DRN']['count'] == 1
    assert result['accepted_boundary_volume_totals']['DRN']['native'] == drain_budget(12.)


def test_missing_entry_output_leaves_an_incomplete_receipt(tmp_path):
    inputs = write_history(tmp_path, omit_second_rates=True)
    with pytest.raises((AssertionError, KeyError)):
        compare_outputs(**inputs)
    receipt = json.loads(inputs['output'].read_text())
    assert receipt['status'] == 'incomplete'
    assert receipt['passed'] is False
    assert 'drn_1' in receipt['error']


def test_a_single_head_column_cannot_qualify_a_three_column_domain(tmp_path):
    # NumPy can broadcast the one returned column against every expected column.
    inputs = write_history(tmp_path, numerical_heads=np.full((2, 1, 1, 1), 4.))
    with pytest.raises((AssertionError, ValueError)):
        compare_outputs(**inputs)


def test_wrong_gross_drain_exchange_fails_despite_exact_net_heads_and_entries(tmp_path):
    wrong = dict(inflow=100., outflow=101., net=-1.)
    inputs = write_history(tmp_path, second_budget=wrong)
    with pytest.raises(AssertionError):
        compare_outputs(**inputs)
    row = json.loads(inputs['output'].read_text())['periods'][1]
    assert row['head_max_m'] == 0.
    assert row['budgets']['DRN']['rate_errors_m3_day'] == dict(inflow=100., outflow=100., net=0.)
    assert row['passed'] is False


def test_different_accepted_timestep_grid_fails_despite_equal_final_time(tmp_path):
    inputs = write_history(tmp_path, endpoint_dts=(5., 5.))
    with pytest.raises((AssertionError, ValueError)):
        compare_outputs(**inputs)
    assert json.loads(inputs['output'].read_text())['status'] == 'failed'


def test_integrated_volume_gate_tracks_the_entry_count_in_each_period(tmp_path):
    # Allowed volume error is 0.01 * (3 entries * 1 day + 1 entry * 9 days)
    # = 0.12 m³. Reusing the first count throughout permits an erroneous 0.30.
    inputs = write_history(tmp_path, accepted_volume=12.2)
    with pytest.raises(AssertionError):
        compare_outputs(**inputs)

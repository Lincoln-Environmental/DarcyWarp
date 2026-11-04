"""Benchmark regressions for physical inputs and refusal of invalid speedups."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from DARCY_WARP_PACKAGE.validation.npf662_benchmark_cases import generate_case
from DARCY_WARP_PACKAGE.validation.npf662_reference import NPF662Reference, load_npf662_case, save_npf662_case
from working_tests.benchmark_npf662_vs_mf6 import suite_settings, parity_metrics, comparison_summary


def test_case_units_types_and_exact_cold_head_roundtrip(tmp_path):
    specifications = suite_settings(sizes=[8], variants=['confined', 'convertible', 'phreatic'])
    generated = []
    for specification in specifications:
        settings = dict(specification['geology'], layer_k=np.asarray(specification['geology']['layer_k']))
        case = generate_case(settings=settings, convertible=specification['convertible'])
        again = generate_case(settings=settings, convertible=specification['convertible'])
        assert case.digest() == again.digest()
        assert case.active.shape == (5, 8, 8)
        assert case.dx*8 == 25000. and case.dy*8 == 25000.
        assert np.all(case.thicknesses > 0.)
        assert not np.any(case.source[1:])
        assert not np.any(case.source[case.fixed != 0])
        assert not np.any(case.cell_types[1:])
        assert np.all(case.cell_types[0] == (specification['variant'] != 'confined'))
        y, x = np.meshgrid(np.linspace(0., 1., 8), np.linspace(0., 1., 8), indexing='ij')
        expected = settings['recharge_annual_mm']/1000./365.*case.dx*case.dy*(
            1.+.2*np.sin(2.*np.pi*y[:, 1:])*np.sin(np.pi*x[:, 1:]))
        np.testing.assert_allclose(case.source[0, :, 1:], expected, rtol=1e-15)
        np.testing.assert_array_equal(case.initial[case.fixed != 0], case.values[case.fixed != 0])
        path = tmp_path.joinpath(specification['case_id']+'.npz')
        save_npf662_case(case=case, path=path)
        assert load_npf662_case(path=path).digest() == case.digest()
        generated.append(case)
    np.testing.assert_array_equal(generated[0].kx, generated[1].kx)
    np.testing.assert_allclose(generated[1].source, 30.*generated[2].source, rtol=1e-15)


def test_baseline_physics_at_250_and_resolution_ladder():
    cases = suite_settings(sizes=[50, 100, 250, 500, 1000], variants=['convertible'])
    baseline = cases[2]
    assert baseline['geology']['shape'] == [5, 250, 250]
    assert baseline['geology']['dx'] == 100.
    assert baseline['geology']['seed'] == 6622505
    assert baseline['geology']['recharge_annual_mm'] == 150.
    assert all(case['geology']['dx']*case['geology']['shape'][1] == 25000. for case in cases)


@pytest.mark.parametrize('variant', ['confined', 'phreatic'])
def test_generated_case_has_an_independently_conservative_small_root(variant):
    specification = suite_settings(sizes=[8], variants=[variant])[0]
    case = generate_case(settings=dict(specification['geology'],
        layer_k=np.asarray(specification['geology']['layer_k'])), convertible=specification['convertible'])
    reference = NPF662Reference(case=case)
    head, info, fields = reference.solve() if variant == 'confined' else reference.solve_newton(max_outer=100)
    assert info['independent_accepted'], info
    if variant == 'phreatic':
        assert np.mean(head[0, :, 1:] < case.tops[0, :, 1:]) > .9
    else:
        assert np.max(head) > 1.


def test_parity_rejects_nonfinite_heads_flux_sign_and_head_error():
    head = np.arange(8., dtype=float).reshape(2, 2, 2)
    flux = np.array([1., -2., 3.])
    free = np.ones_like(head, dtype=bool)
    gates = dict(head_rmse_m=1e-3, head_max_m=5e-3, flux_relative_l2=1e-6)
    assert parity_metrics(head=head, flux=flux, native_head=head, native_flux=flux, free=free, gates=gates)['passed']
    bad_head = head.copy()
    bad_head[0, 0, 0] += .01
    assert not parity_metrics(head=bad_head, flux=flux, native_head=head, native_flux=flux, free=free, gates=gates)['passed']
    assert not parity_metrics(head=head, flux=-flux, native_head=head, native_flux=flux, free=free, gates=gates)['passed']
    bad_head[0, 0, 0] = np.nan
    assert not parity_metrics(head=bad_head, flux=flux, native_head=head, native_flux=flux, free=free, gates=gates)['passed']


def test_medians_require_every_timed_repeat_and_exclude_warmup_and_resident():
    rows = []
    for index in range(3):
        rows.append(dict(case_id='case', engine='mf6', preconditioner='IMS', phase='timed', accepted=True,
            process_seconds=12.+index, end_to_end_seconds=20.+index))
        rows.append(dict(case_id='case', engine='darcywarp', preconditioner='kcycle', phase='timed', accepted=True,
            public_solve_seconds=2.+index, end_to_end_seconds=3.+index))
    for phase in ['warmup', 'resident']:
        rows.append(dict(case_id='case', engine='darcywarp', preconditioner='kcycle', phase=phase, accepted=False))
    summary = comparison_summary(rows=rows, case_id='case', preconditioner='kcycle', repeats=3)
    assert summary['accepted']
    assert summary['mf6_process_median_s'] == 13.
    assert summary['darcywarp_public_median_s'] == 3.
    assert summary['end_to_end_speedup'] == 21./4.
    rows[1]['accepted'] = False
    failed = comparison_summary(rows=rows, case_id='case', preconditioner='kcycle', repeats=3)
    assert not failed['accepted'] and failed['end_to_end_speedup'] is None
    rows[1]['accepted'] = True
    missing = comparison_summary(rows=rows[2:], case_id='case', preconditioner='kcycle', repeats=3)
    assert not missing['accepted'] and missing['process_vs_public_speedup'] is None
    alternate = comparison_summary(rows=rows, case_id='case', preconditioner='diagonal', repeats=3)
    assert not alternate['accepted']


def test_dry_run_emits_exact_plan_without_solver_environment():
    root = Path(__file__).resolve().parents[1]
    run = subprocess.run([sys.executable, str(root/'working_tests/benchmark_npf662_vs_mf6.py'),
        '--sizes', '16', '16', '--variants', 'confined', '--dry-run'], capture_output=True, text=True, check=True)
    plan = json.loads(run.stdout)
    assert len(plan['cases']) == 1
    assert plan['cases'][0]['geology']['shape'] == [5, 16, 16]
    assert plan['repeats'] == 3


def test_cpu_driver_against_small_independent_oracle(tmp_path, monkeypatch):
    """Exercise real CPU solves and reuse; this deliberately does not run MF6."""
    pytest.importorskip('warp')
    monkeypatch.setenv('DARCY_FLOAT', 'float64')
    from DARCY_WARP_PACKAGE.validation import npf662_mf6
    from working_tests import benchmark_npf662_vs_mf6

    def independent_oracle(**kwargs):
        reference = NPF662Reference(case=kwargs['case'])
        head, checks, fields = reference.solve()
        return head, dict(native_success=True, independent_conservation=checks['independent_accepted'],
            independent=checks, seconds=1., settings={}), dict(flux_native=fields['flux'])

    monkeypatch.setattr(npf662_mf6, 'run_reference_mf6', independent_oracle)
    output = tmp_path/'cpu_evidence'
    monkeypatch.setattr(sys, 'argv', ['benchmark', '--sizes', '4', '--variants', 'confined', 'phreatic',
        '--device', 'cpu', '--mf6', sys.executable, '--output', str(output), '--resident-repeats', '1'])
    assert benchmark_npf662_vs_mf6.main() == 0
    result = json.loads(output.joinpath('results.json').read_text())
    assert result['passed'] and len(result['summary']) == 2
    assert all(row['accepted'] for row in result['runs'])
    resident = [row for row in result['runs'] if row['phase'] == 'resident']
    assert len(resident) == 2
    assert all(row['info']['multigrid']['hierarchy_reused'] for row in resident)
    assert all(row['mf6_samples'] == row['darcywarp_samples'] == 3 for row in result['summary'])

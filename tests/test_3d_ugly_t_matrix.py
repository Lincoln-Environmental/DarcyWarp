"""Exact 2D field reuse, anisotropy isolation and complete benchmark reporting."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from DARCY_WARP_PACKAGE.model_builder import make_ugly_T_field
from DARCY_WARP_PACKAGE.validation.npf662_benchmark_cases import (
    add_ghb_boundaries, generate_case, with_ugly_conductivity,
)
from DARCY_WARP_PACKAGE.validation.npf662_reference import NPF662Reference, load_npf662_case, save_npf662_case
from working_tests.benchmark_npf662_vs_mf6 import suite_settings
from working_tests.run_3d_ugly_t_matrix import case_jobs, job_command, job_summary


def base_case():
    specification = suite_settings(sizes=[16], variants=['confined'])[0]
    geology = dict(specification['geology'], layer_k=np.asarray(specification['geology']['layer_k']))
    case = generate_case(settings=geology, convertible=specification['convertible'])
    return case, geology['layer_k'] / geology['layer_k'][0]


@pytest.mark.parametrize('ratio', [1., 10., 100.])
def test_exact_2d_lift_and_vertical_flux_ratio(tmp_path, ratio):
    base, multipliers = base_case()
    before = base.digest()
    case = with_ugly_conductivity(case=base, seed=42, reference_thickness=100.,
        layer_multipliers=multipliers, vertical_anisotropy=ratio)
    expected = make_ugly_T_field(nx=16, ny=16, seed=42).astype(np.float64)[None] / 100. * multipliers[:, None, None]
    np.testing.assert_array_equal(actual=case.kx, desired=expected)
    np.testing.assert_array_equal(actual=case.ky, desired=expected)
    np.testing.assert_array_equal(actual=case.kz, desired=expected / ratio)
    assert base.digest() == before
    for name in ('source', 'initial', 'z_interfaces', 'active', 'fixed', 'values'):
        np.testing.assert_array_equal(actual=getattr(case, name), desired=getattr(base, name))
    isotropic = replace(case, kz=case.kx.copy())
    reference, iso_reference = NPF662Reference(case=case), NPF662Reference(case=isotropic)
    expected_conductance = iso_reference.c0.copy()
    expected_conductance[reference.axes == 0] /= ratio
    np.testing.assert_allclose(actual=reference.c0, desired=expected_conductance, rtol=1e-15, atol=0.)
    path = tmp_path / 'case.npz'
    save_npf662_case(case=case, path=path)
    assert load_npf662_case(path=path).digest() == case.digest()


def test_anisotropy_changes_cache_identity_but_not_ghb():
    base, multipliers = base_case()
    cases = [add_ghb_boundaries(case=with_ugly_conductivity(case=base, seed=42,
        reference_thickness=100., layer_multipliers=multipliers, vertical_anisotropy=ratio),
        conductance_factor=1., draining_offset=-.5, injecting_offset=1.) for ratio in (1., 10., 100.)]
    assert len({case.digest() for case in cases}) == 3
    for case in cases[1:]:
        np.testing.assert_array_equal(actual=case.ghb_conductance, desired=cases[0].ghb_conductance)
        np.testing.assert_array_equal(actual=case.ghb_head, desired=cases[0].ghb_head)
    with pytest.raises(ValueError, match='before adding GHB'):
        with_ugly_conductivity(case=cases[0], seed=42, reference_thickness=100.,
            layer_multipliers=multipliers, vertical_anisotropy=10.)


@pytest.mark.parametrize('ratio', [0., -1., np.nan, np.inf])
def test_invalid_ratio_rejected(ratio):
    base, multipliers = base_case()
    with pytest.raises(ValueError, match='positive finite'):
        with_ugly_conductivity(case=base, seed=42, reference_thickness=100.,
            layer_multipliers=multipliers, vertical_anisotropy=ratio)


def test_matrix_covers_all_requested_physics_and_preserves_mf6_settings(tmp_path):
    args = SimpleNamespace(sizes=[500], vertical_anisotropy=[1., 10., 100.], ghb_modes=['off', 'on'],
        families=['cartesian', 'npf'], variants=['confined', 'convertible'], modes=['steady', 'transient'],
        periods=20, repeats=3, seed=42, reference_thickness=100., preferred_head_target=1e-6,
        mf6_timeout=1800., device='cuda:0')
    jobs = case_jobs(args=args)
    assert len(jobs) == 36
    for job in jobs:
        command = job_command(job=job, root=tmp_path, directory=tmp_path / 'run', cache=tmp_path / 'cache', mf6=tmp_path / 'mf6')
        assert '--tight' not in command
        assert '--gpu-tight' in command
        assert ('--ghb' in command) == (job['ghb'] == 'on')
        if job['family'] == 'npf' and job['mode'] == 'transient':
            assert '--npf-history' in command
        if job['family'] == 'npf' and job['variant'] == 'convertible' and job['mode'] == 'transient':
            assert str(tmp_path / 'working_tests/npf_runtime_controls.json') in command


def test_failed_process_never_advertises_speedup(tmp_path):
    row = dict(preferred_head_target_met=False, median_public_solve_seconds=1.,
        mf6_single_reference_run_seconds=10., worst_head_max_m=1e-5)
    (tmp_path / 'results.json').write_text(data=json.dumps(obj=dict(passed=False, summary=[row], cases=[])))
    result = job_summary(record=dict(job={}, output=str(tmp_path), returncode=1, fingerprint='failed'))
    assert not result['sanity_passed']
    assert result['speedup'] is None

"""An explicit final tolerance must reach both gates without changing defaults."""
import numpy as np
import pytest
import warp as wp

from DARCY_WARP_PACKAGE import WarpDarcySolver3D


def build_simple_model(*, device):
    shape = (1, 1, 2)
    one = np.ones(shape)
    model = WarpDarcySolver3D(nx=2, ny=1, nz=1, dx=1., dy=1., device=device,
        formulation='mf6_6_6_2_npf', implementation='fast')
    model.build_from_layers(z_interfaces=np.array([[[3., 3.]], [[0., 0.]]]),
        kx_field=one, ky_field=one, kz_field=one, active=one.astype(int),
        bc_mask=np.array([0, 1]).reshape(shape), bc_values=4.*one,
        rhs=np.array([.4, 0.]).reshape(shape), initial_head=4.*one)
    return model


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_relative_override_controls_both_acceptance_gates_and_resets(device):
    if device.startswith('cuda') and not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    shape = (1, 1, 2)
    one = np.ones(shape)
    values = 4.*one
    initial = values.copy()
    initial[0, 0, 0] += .4/3.+(.8*5e-8)/3.
    with WarpDarcySolver3D(nx=2, ny=1, nz=1, dx=1., dy=1., device=device,
            formulation='mf6_6_6_2_npf', implementation='fast') as model:
        model.build_from_layers(z_interfaces=np.array([[[3., 3.]], [[0., 0.]]]),
            kx_field=one, ky_field=one, kz_field=one, active=one.astype(int),
            bc_mask=np.array([0, 1]).reshape(shape), bc_values=values,
            rhs=np.array([.4, 0.]).reshape(shape), initial_head=initial)
        options = dict(unconfined=True, npf_cell_type=1, max_outer=0, max_cycles=0, krylov_restart=4)
        first, strict = model.solve(**options)
        relaxed, loose = model.solve(npf_final_relative_tolerance=1e-7, **options)
        again, restored = model.solve(**options)
    assert not strict['converged']
    assert loose['converged']
    assert not restored['converged']
    assert loose['final_acceptance_thresholds']['relative_cell_imbalance'] == 1e-7
    assert restored['final_acceptance_thresholds']['relative_cell_imbalance'] == 1e-8
    for name in ('physical_rms_m3_day', 'head_equivalent_rms_m', 'prescribed_head_error_m'):
        assert loose['final_acceptance_thresholds'][name] == strict['final_acceptance_thresholds'][name]
    np.testing.assert_array_equal(actual=first, desired=relaxed)
    np.testing.assert_array_equal(actual=first, desired=again)


@pytest.mark.parametrize('value', [-1., np.nan, np.inf, True])
def test_invalid_relative_override_rejected(value):
    with build_simple_model(device='cpu') as model:
        with pytest.raises(ValueError):
            model.solve(unconfined=True, npf_cell_type=1, npf_final_relative_tolerance=value)


def test_conflicting_relative_controls_rejected():
    with build_simple_model(device='cpu') as model:
        with pytest.raises(ValueError, match='Choose'):
            model.solve(npf_final_relative_tolerance=1e-7, final_budget_relative_tol=1e-6)

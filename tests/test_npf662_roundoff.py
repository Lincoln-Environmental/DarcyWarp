"""Roundoff acceptance is explicit, cellwise, and cannot bypass conservation."""
import numpy as np
import pytest
import warp as wp

from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D
from DARCY_WARP_PACKAGE.solvers.npf662_newton import NPF662NewtonControls
from DARCY_WARP_PACKAGE.validation.npf662_metrics import componentwise_roundoff


@pytest.fixture(params=['cpu', 'cuda:0'])
def device(request):
    if request.param.startswith('cuda') and not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    return request.param


def build_cell(*, model, source, conductance, boundary, initial):
    shape = (1, 1, 1)
    model.build_from_K_fields(kx_field=np.ones(shape=shape), ky_field=np.ones(shape=shape),
        kz_field=np.ones(shape=shape), active=np.ones(shape=shape, dtype=np.int32),
        bc_mask=np.zeros(shape=shape, dtype=np.int32), bc_values=np.zeros(shape=shape),
        rhs=np.full(shape=shape, fill_value=source), initial_head=np.full(shape=shape, fill_value=initial),
        ghb_conductance=conductance, ghb_head=boundary)


@pytest.mark.parametrize('transient,native', [(False, False), (True, False), (True, True)])
def test_explicit_roundoff_mode_reports_requested_thresholds_separately(device, transient, native):
    with WarpDarcySolver3D(nx=1, ny=1, nz=1, dx=1., device=device,
            implementation='fast', formulation='mf6_6_6_2_npf') as model:
        build_cell(model=model, source=.1, conductance=3., boundary=1., initial=.25)
        options = dict(max_outer=2, max_cycles=20, npf_preconditioner='kcycle',
            abs_tol_min=0., npf_final_relative_tolerance=0.,
            npf_newton_controls=NPF662NewtonControls(tiled_reductions=True, reuse_trial_state=True))
        if transient:
            options.update(transient=True, dt=2., head_prev=np.full(shape=(1,1,1), fill_value=.25),
                ss=.3, sy=0., npf_device_native=native)
        strict_head, strict = model.solve(**options)
        assert not strict['converged']
        head, info = model.solve(npf_roundoff_acceptance=True, **options)
        assert info['converged'], info
        assert info['roundoff_accepted']
        assert info['roundoff_bound_ratio_max'] <= 1.
        assert not info['requested_thresholds_met']
        assert info['precision_limited']
        slope = .15 if transient else 0.
        expected = (3.1+slope*.25)/(3.+slope)
        np.testing.assert_allclose(actual=head, desired=expected, rtol=0., atol=4*np.finfo(float).eps)
        np.testing.assert_allclose(actual=head, desired=strict_head, rtol=0., atol=4*np.finfo(float).eps)
        # Reuse must not leave the opt-in mode active for subsequent defaults.
        unused, strict_again = model.solve(**options)
        assert not strict_again['converged']


def test_roundoff_mode_rejects_unrepresentable_forcing_with_bad_mass_balance(device):
    with WarpDarcySolver3D(nx=1, ny=1, nz=1, dx=1., device=device,
            implementation='fast', formulation='mf6_6_6_2_npf') as model:
        build_cell(model=model, source=1e-6, conductance=1e12, boundary=1e16, initial=1e16)
        unused, info = model.solve(max_outer=0, npf_roundoff_acceptance=True)
        assert not info['converged']
        assert info['physical_acceptance']['relative_cell_imbalance'] == pytest.approx(1.)
        assert not info['physical_acceptance']['roundoff_accepted']


@pytest.mark.parametrize('error,accepted', [(0., True), (1e-16, True), (1e-8, False)])
def test_componentwise_bound_does_not_cancel_opposite_cell_errors(error, accepted):
    result = componentwise_roundoff(head=np.ones(shape=2), source=np.zeros(shape=2),
        residual=np.array([error, -error]), free=np.ones(shape=2, dtype=bool),
        left=np.array([0]), right=np.array([1]), conductance=np.array([1.]),
        ghb_conductance=np.zeros(shape=2), ghb_head=np.zeros(shape=2))
    assert result['roundoff_rows_valid'] is accepted


@pytest.mark.parametrize('value', [1, 'yes', None])
def test_roundoff_switch_rejects_non_booleans(value):
    with WarpDarcySolver3D(nx=1, ny=1, nz=1, dx=1., device='cpu',
            implementation='fast', formulation='mf6_6_6_2_npf') as model:
        build_cell(model=model, source=.1, conductance=3., boundary=1., initial=.25)
        with pytest.raises(ValueError, match='npf_roundoff_acceptance must be Boolean'):
            model.solve(npf_roundoff_acceptance=value)


def test_convertible_storage_roundoff_includes_layer_datum(device):
    previous = np.full(shape=(1, 1, 1), fill_value=.004)
    expected = .01
    slope = .15*2500./7./(1.-1e-6)
    source = slope*(expected-.004)+2.*expected
    with WarpDarcySolver3D(nx=1, ny=1, nz=1, dx=50., dy=50., dz=24., device=device,
            implementation='fast', formulation='mf6_6_6_2_npf') as model:
        build_cell(model=model, source=source, conductance=2., boundary=0., initial=.004)
        head, info = model.solve(transient=True, dt=7., head_prev=previous, ss=0., sy=.15,
            unconfined=True, top_elevation=8., npf_roundoff_acceptance=True,
            npf_final_relative_tolerance=1e-12, npf_device_native=True, max_outer=10)
        assert info['converged'], info
        assert info['roundoff_bound_ratio_max'] <= 1.
        np.testing.assert_allclose(actual=head, desired=expected, rtol=0., atol=1e-14)
        from DARCY_WARP_PACKAGE.solvers.npf662_3d import case_from_model
        from DARCY_WARP_PACKAGE.validation.npf662_transient_reference import NPF662TransientReference
        case = case_from_model(model=model, top=8., cell_type=1)
        reference = NPF662TransientReference(case=case, previous=previous, dt=7., ss=0., sy=.15)
        checks, unused = reference.evaluate(head=head, roundoff_acceptance=True)
        assert checks['independent_accepted'], checks

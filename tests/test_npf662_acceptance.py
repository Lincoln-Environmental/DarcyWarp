"""Requested physical thresholds and finite aggregates are independent gates."""
import numpy as np
import pytest

from DARCY_WARP_PACKAGE.validation.npf662_reference import NPF662Case, NPF662Reference
from DARCY_WARP_PACKAGE.solvers.session_3d import ThreeDOperatorSession
from DARCY_WARP_PACKAGE.solvers.npf662_newton import device_accepted
from validate_npf662_confined import make_gpu_model


def line_case(*, conductivity=1., endpoints=(0., 2.)):
    shape = (1, 1, 3)
    one = np.ones(shape)
    values = np.linspace(*endpoints, num=3).reshape(shape)
    return NPF662Case(kx=conductivity*one, ky=conductivity*one, kz=conductivity*one,
        active=one.astype(int), fixed=np.array([1, 0, 1]).reshape(shape), values=values,
        source=np.zeros(shape), initial=values.copy(), dx=1., dy=1., dz=1.)


def synthetic_backend(self, **kwargs):
    head = self.bc_values.copy()
    head[0, 0, 1] += 1e-9
    return head, dict(converged=True)


def test_confined_wrapper_honours_requested_physical_thresholds(monkeypatch):
    case = line_case()
    monkeypatch.setattr(ThreeDOperatorSession, 'solve', synthetic_backend)
    with make_gpu_model(case=case, device='cpu') as model:
        head, info = model.solve(abs_tol_min=1e-14, final_budget_relative_tol=1e-12)
    assert info['backend_converged']
    assert not info['converged']
    assert info['final_acceptance_thresholds']['physical_rms_m3_day'] == 1e-14
    assert info['final_acceptance_thresholds']['relative_cell_imbalance'] == 1e-12


def test_reference_thresholds_and_unchanged_defaults():
    case = line_case()
    head = case.values.copy(); head[0, 0, 1] += 1e-9
    reference = NPF662Reference(case=case)
    defaults, unused = reference.evaluate(head=head)
    strict, unused = reference.evaluate(head=head, physical_tolerance=1e-14, relative_tolerance=1e-12)
    assert defaults['independent_accepted']
    assert not strict['independent_accepted']
    assert strict['physical_metrics_valid']
    assert strict['head_residual_rms'] == pytest.approx(1e-9, rel=1e-6)


@pytest.mark.parametrize('conductivity,endpoints,offset', [
    (1e308, (0., 2.), .001), (1e307, (-10., 10.), 1e-7)])
def test_finite_faces_cannot_hide_overflowed_aggregates(conductivity, endpoints, offset):
    case = line_case(conductivity=conductivity, endpoints=endpoints)
    head = case.values.copy(); head[0, 0, 1] += offset
    with np.errstate(over='ignore', invalid='ignore'):
        checks, fields = NPF662Reference(case=case).evaluate(head=head)
    assert np.isfinite(fields['flux']).all()
    assert not checks['independent_accepted']
    assert not checks['physical_metrics_valid']


def test_stable_large_representable_rms_and_tiny_throughput():
    for conductivity in (1e200, 1e-300):
        case = line_case(conductivity=conductivity)
        head = case.values.copy(); head[0, 0, 1] += .001
        checks, unused = NPF662Reference(case=case).evaluate(head=head)
        assert checks['physical_metrics_valid']
        assert np.isfinite(checks['physical_rms']) and checks['physical_rms'] > 0.
        np.testing.assert_allclose(checks['physical_rms'], .002*conductivity, rtol=1e-12, atol=0.)
        exact,unused=NPF662Reference(case=case).evaluate(head=case.values)
        assert exact['physical_metrics_valid'] and exact['independent_accepted']


def test_subnormal_throughput_uses_the_declared_tiny_denominator():
    case=line_case(endpoints=(0.,2e-310))
    head=case.values.copy();head[0,0,1]+=1e-313
    checks,unused=NPF662Reference(case=case).evaluate(head=head)
    assert 0.<checks['gross_throughput']<np.finfo(float).tiny
    assert checks['physical_metrics_valid'] and checks['independent_accepted']
    assert checks['relative_cell_imbalance']==checks['residual_l1']/np.finfo(float).tiny


@pytest.mark.parametrize('value', [np.nan, np.inf, -1.])
@pytest.mark.parametrize('name', ['physical_tolerance', 'relative_tolerance', 'head_tolerance_m', 'boundary_tolerance_m'])
def test_invalid_evaluator_thresholds_rejected(name, value):
    case = line_case()
    with pytest.raises(ValueError, match='tolerance'):
        NPF662Reference(case=case).evaluate(head=case.values, **{name: value})


@pytest.mark.parametrize('value', [np.nan, np.inf, -1.])
@pytest.mark.parametrize('name', ['abs_tol_min', 'final_budget_relative_tol', 'rel_tol'])
def test_invalid_public_thresholds_rejected(name, value):
    case = line_case()
    with make_gpu_model(case=case, device='cpu') as model:
        with pytest.raises(ValueError, match='tolerance'):
            model.solve(**{name: value})


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_real_strict_public_solve_and_default_policy(device):
    case = line_case()
    with make_gpu_model(case=case, device=device) as model:
        head, info = model.solve(abs_tol_min=1e-14, final_budget_relative_tol=1e-12)
        assert info['converged'], info
        checks, unused = NPF662Reference(case=case).evaluate(head=head,
            physical_tolerance=1e-14, relative_tolerance=1e-12)
        assert checks['independent_accepted']
        head, default = model.solve()
        assert default['converged']
        assert default['final_acceptance_thresholds']['relative_cell_imbalance'] == 1e-8


def test_prescribed_empty_free_and_inactive_markers():
    case = line_case(); case.fixed[:] = 1
    checks, unused = NPF662Reference(case=case).evaluate(head=case.values)
    assert checks['independent_accepted'] and checks['free_cells'] == 0
    case.active[0, 0, 1] = 0; case.fixed[0, 0, 1] = 0
    head = case.values.copy(); head[0, 0, 1] = np.nan
    checks, unused = NPF662Reference(case=case).evaluate(head=head)
    assert checks['independent_accepted'] and checks['inactive_marker_cells'] == 1


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_device_overflow_is_not_promoted_at_final_gate(device):
    case = line_case(conductivity=1e308)
    case.initial[0, 0, 1] += .001
    with np.errstate(over='ignore', invalid='ignore'):
        with make_gpu_model(case=case, device=device) as model:
            head, info = model.solve(unconfined=True, top_elevation=0., max_outer=0, max_cycles=0)
    assert not info['converged']
    assert not info['device_final_metrics']['finite']
    assert not info['independent_acceptance']['physical_metrics_valid']


def test_device_acceptance_checks_actual_metrics_before_serialization():
    metrics = dict(finite=True, head_residual_rms=0., physical_rms=np.inf,
                   relative_cell_imbalance=0., gross_throughput=np.inf)
    assert not device_accepted(metrics=metrics)

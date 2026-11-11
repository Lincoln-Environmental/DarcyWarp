"""Device histories must match explicit timesteps and preserve storage state."""
from dataclasses import replace

import numpy as np
import pytest
import warp as wp

from test_npf662_device_native import make_case, make_model, solve_options
from DARCY_WARP_PACKAGE.solvers.npf662_newton import NPF662NewtonControls
from DARCY_WARP_PACKAGE.solvers.npf662_kcycle import NPF662KCycleControls
from DARCY_WARP_PACKAGE.validation.npf662_transient_reference import NPF662TransientReference


@pytest.fixture(params=['cpu', 'cuda:0'])
def device(request):
    if request.param.startswith('cuda') and not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    return request.param


def history_options(*, case, precision, predictor):
    options = solve_options(case=case, previous=case.initial, dt=7.)
    for key in ('transient', 'dt', 'head_prev', 'ss', 'sy'):
        options.pop(key)
    options.update(npf_newton_controls=NPF662NewtonControls(timestep_predictor=predictor, inner_forcing=.03),
        npf_kcycle_controls=NPF662KCycleControls(levels=3, coarse_sweeps=2,
            cached_line_factors=True, preconditioner_precision=precision, coefficient_state='current'))
    return options


@pytest.mark.parametrize('precision,predictor,tiled', [
    ('float64', False, False), ('float64', True, False), ('float32', True, False),
    ('float64', True, True), ('float32', True, True)])
def test_history_matches_explicit_steps_and_oracle(device, precision, predictor, tiled):
    case = replace(make_case(convertible=True, shape=(2, 7, 9)), dx=50., dy=50.)
    factors = np.array([1., 1., 1.5, .25, 1.])
    times = np.array([7., 7., 7., 3.5, 7.])
    options = history_options(case=case, precision=precision, predictor=predictor)
    options['npf_newton_controls'] = replace(options['npf_newton_controls'], tiled_reductions=tiled)
    expected = []
    previous = case.initial.copy()
    with make_model(case=case, device=device) as model:
        for factor, dt in zip(factors, times):
            model.update_rhs(rhs=case.source*factor)
            model.update_initial_head(initial_head=previous)
            head, info = model.solve(transient=True, dt=dt, head_prev=previous, ss=1e-3, sy=.15, **options)
            assert info['converged']
            expected.append(head)
            previous = head
    with make_model(case=case, device=device) as model:
        options['npf_newton_controls'] = replace(options['npf_newton_controls'], reuse_trial_state=True)
        options['npf_kcycle_controls'] = replace(options['npf_kcycle_controls'], device_hierarchy_build=True)
        heads, info = model.solve_transient_npf(source_factors=factors, dt=times,
            head_prev=case.initial, ss=1e-3, sy=.15, **options)
        assert info['converged'] and info['accepted_steps'] == len(factors)
        assert info['solver_head_readbacks'] == 1 and info['output_history_readbacks'] == 1
        np.testing.assert_allclose(actual=heads, desired=np.stack(expected), rtol=0., atol=1e-9)
        np.testing.assert_array_equal(actual=model._session.rhs, desired=case.source)
        np.testing.assert_array_equal(actual=model._session.initial_head, desired=case.initial)
        state = model._session._npf662_workspace_holder['native_state']
        np.testing.assert_array_equal(actual=state.previous, desired=heads[-2])
        previous = case.initial
        for head, factor, dt in zip(heads, factors, times):
            checks, unused = NPF662TransientReference(case=replace(case, source=case.source*factor),
                previous=previous, dt=dt, ss=1e-3, sy=.15).evaluate(head=head)
            assert checks['independent_accepted']
            previous = head
        model.update_initial_head(initial_head=heads[-1])
        next_head, next_info = model.solve(transient=True, dt=7., head_prev=heads[-1], ss=1e-3, sy=.15, **options)
        assert next_info['converged']
        checks, unused = NPF662TransientReference(case=case, previous=heads[-1], dt=7., ss=1e-3, sy=.15).evaluate(head=next_head)
        assert checks['independent_accepted']


def test_history_changes_k_between_calls_and_matches_fresh(device):
    case = make_case(convertible=True)
    options = history_options(case=case, precision='float32', predictor=True)
    options['npf_kcycle_controls'] = replace(options['npf_kcycle_controls'], device_hierarchy_build=True)
    factors = np.array([1., 1.5, .25])
    with make_model(case=case, device=device) as model:
        heads, info = model.solve_transient_npf(source_factors=factors, dt=7.,
            head_prev=case.initial, ss=1e-3, sy=.15, **options)
        assert info['converged']
        changed = replace(case, kx=case.kx*2., ky=case.ky*.7, kz=case.kz*1.2)
        model.update_K_fields(kx_field=changed.kx, ky_field=changed.ky, kz_field=changed.kz)
        updated, info = model.solve_transient_npf(source_factors=factors, dt=7.,
            head_prev=case.initial, ss=1e-3, sy=.15, **options)
        assert info['converged']
    with make_model(case=changed, device=device) as model:
        fresh, info = model.solve_transient_npf(source_factors=factors, dt=7.,
            head_prev=case.initial, ss=1e-3, sy=.15, **options)
        assert info['converged']
    np.testing.assert_allclose(actual=updated, desired=fresh, rtol=0., atol=1e-9)


def test_history_stops_at_first_failed_step_and_restores_inputs(device):
    case = make_case(convertible=True)
    case.initial[:] = 4.
    options = history_options(case=case, precision='float64', predictor=True)
    options.update(max_outer=0, max_cycles=0)
    with make_model(case=case, device=device) as model:
        heads, info = model.solve_transient_npf(source_factors=[0., 1e6, 1.], dt=7.,
            head_prev=case.initial, ss=1e-3, sy=.15, **options)
        assert not info['converged'] and info['accepted_steps'] == 1
        assert info['returned_steps'] == 2 and heads.shape[0] == 2
        assert info['periods'][0]['converged'] and not info['periods'][1]['converged']
        np.testing.assert_array_equal(actual=model._session.rhs, desired=case.source)
        np.testing.assert_array_equal(actual=model._session.initial_head, desired=case.initial)


def test_history_independent_fallback_uses_each_previous_head_and_source(device):
    case = make_case(convertible=True)
    factors = [1., 1.7, .2]
    options = history_options(case=case, precision='float64', predictor=True)
    options['npf_newton_controls'] = replace(options['npf_newton_controls'], reuse_trial_state=True)
    with make_model(case=case, device=device) as model:
        heads, info = model.solve_transient_npf(source_factors=factors, dt=7.,
            head_prev=case.initial, ss=0., sy=.15, **options)
        assert info['converged']
        assert info['solver_head_readbacks'] == len(factors)
        assert all(row['independent_audit_performed'] for row in info['periods'])
        previous = case.initial
        for head, factor in zip(heads, factors):
            checks, unused = NPF662TransientReference(case=replace(case, source=case.source*factor),
                previous=previous, dt=7., ss=0., sy=.15).evaluate(head=head)
            assert checks['independent_accepted']
            previous = head


def test_history_exception_invalidates_workspace_and_restores_inputs(device, monkeypatch):
    from DARCY_WARP_PACKAGE.solvers import npf662_history

    def reject_step(**unused):
        raise RuntimeError('Injected next-step failure')

    case = make_case(convertible=True)
    options = history_options(case=case, precision='float64', predictor=True)
    with make_model(case=case, device=device) as model:
        monkeypatch.setattr(npf662_history, 'prepare_next_step', reject_step)
        with pytest.raises(RuntimeError, match='Injected'):
            model.solve_transient_npf(source_factors=[1., 2.], dt=7.,
                head_prev=case.initial, ss=1e-3, sy=.15, **options)
        assert 'workspace' not in model._session._npf662_workspace_holder
        assert 'native_state' not in model._session._npf662_workspace_holder
        np.testing.assert_array_equal(actual=model._session.rhs, desired=case.source)
        np.testing.assert_array_equal(actual=model._session.initial_head, desired=case.initial)
        head, info = model.solve(transient=True, dt=7., head_prev=case.initial, ss=1e-3, sy=.15, **options)
        assert info['converged']

"""Shared production period engine, explicit forcing and scratch rollback."""

import numpy as np
import pytest
import warp as wp

from DARCY_WARP_PACKAGE.solvers.transient_config import production_secant_sy_settings
from DARCY_WARP_PACKAGE.solvers.transient_unconfined import TransientUnconfinedSession
from working_tests.run_device_transient_fast_path_smoke import build_small_solver


def make_session(*, device):
    wp.init()
    if device.startswith('cuda') and not wp.get_cuda_devices():
        pytest.skip('CUDA unavailable')
    model, initial, k, bottom, top, active, bc, values = build_small_solver(device=device)
    settings = production_secant_sy_settings()
    parameters = dict(k_field=k, zbot_field=bottom, ztop_field=top, sy=.1, ss=1e-5,
        active=active, bc_mask=bc, bc_values=values, solve_controls=settings['solve_controls'])
    session = TransientUnconfinedSession(model=model, **parameters)
    recharge = np.full(initial.shape, .001)
    recharge[bc != 0] = 0.
    return model, session, parameters, initial, recharge


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_uniform_period_driver_and_spatial_api_match(device):
    model, session, parameters, initial, recharge = make_session(device=device)
    spatial, info = session.advance(head_prev=initial, recharge_rate_m_per_day=recharge, dt=1.)
    replay_model, _, _, _, _ = make_session(device=device)
    heads, replay = replay_model.solve_transient_2d_unconfined(initial_head=initial,
        recharge_rates=np.array([.001, .002]), dt=1., return_info=True, **parameters)
    np.testing.assert_allclose(spatial, heads[0], rtol=0., atol=1e-10)
    first = replay['period_infos'][0]
    for name in ('strict_picard_convergence_passed', 'practical_picard_acceptance_passed',
                 'total_inner_kcycles', 'outer_iterations', 'adaptive_dt_substep_count',
                 'adaptive_dt_substep_dts', 'adaptive_dt_retry_count'):
        assert info.get(name) == first.get(name), name


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_workspace_persists_and_provisional_state_is_not_accepted(device, monkeypatch):
    model, session, parameters, initial, recharge = make_session(device=device)
    session.advance(head_prev=initial, recharge_rate_m_per_day=recharge, dt=1.)
    pointers = {name: value.ptr for name, value in session.workspace['buffers'].items()}
    hierarchy = id(model.mg_levels)
    graphs = {key: id(value) for key, value in session.workspace['kcycle_graphs'].items()}

    def forbidden_rebuild(**arguments):
        raise AssertionError('ordinary interval rebuilt hierarchy')

    monkeypatch.setattr(model, 'build_hierarchy', forbidden_rebuild)
    alternate = recharge.copy()
    alternate[:, 1:3] *= 3.
    repeated, info = session.advance(head_prev=initial, recharge_rate_m_per_day=alternate, dt=.5)
    fresh_model, _, _, _, _ = make_session(device=device)
    fresh = TransientUnconfinedSession(model=fresh_model, **parameters)
    oracle, oracle_info = fresh.advance(head_prev=initial, recharge_rate_m_per_day=alternate, dt=.5)
    np.testing.assert_allclose(repeated, oracle, rtol=0., atol=1e-10)
    assert info['total_inner_kcycles'] == oracle_info['total_inner_kcycles']
    assert session.setup_count == 1
    assert pointers == {name: value.ptr for name, value in session.workspace['buffers'].items()}
    assert id(model.mg_levels) == hierarchy
    assert all(id(session.workspace['kcycle_graphs'][key]) == value for key, value in graphs.items())


@pytest.mark.parametrize('invalid', ['shape', 'nan', 'inf', 'chd', 'inactive'])
def test_invalid_spatial_forcing_rejected_before_numerical_state(invalid):
    model, session, parameters, initial, recharge = make_session(device='cpu')
    if invalid == 'shape':
        recharge = recharge[:-1]
    elif invalid in {'nan', 'inf'}:
        recharge[0, 1] = np.nan if invalid == 'nan' else np.inf
    elif invalid == 'inactive':
        session._parameters['active'][1, 1] = 0
    else:
        recharge[:, 0] = .1
    with pytest.raises(ValueError):
        session.advance(head_prev=initial, recharge_rate_m_per_day=recharge, dt=1.)
    assert session.setup_count == 0


def test_production_replay_imports_same_fresh_configuration():
    from working_tests import transient_replay_settings as replay
    from DARCY_WARP_PACKAGE.solvers import transient_config as production
    assert replay.default_solve_controls is production.default_solve_controls
    assert replay.production_secant_sy_settings is production.production_secant_sy_settings
    first = production.production_secant_sy_settings()
    first['solve_controls']['max_levels'] = 2
    second = production.production_secant_sy_settings()
    assert second['solve_controls']['max_levels'] == 4
    assert second['solve_controls']['min_coarse_cells'] == 500
    assert second['solve_controls']['use_device_transient_fast_path']
    assert second['solve_controls']['adaptive_dt_enabled']
    assert second['solve_controls']['unconfined_startup_mode'] == 'confined_pre_solve'


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_public_interval_bridge_never_calls_multi_period_driver(device, monkeypatch):
    model, _, parameters, initial, recharge = make_session(device=device)
    from DARCY_WARP_PACKAGE.solvers import transient_unconfined as periods

    def forbidden_driver(**arguments):
        raise AssertionError('interval API called the whole multi-period driver')

    monkeypatch.setattr(periods, 'solve_transient_unconfined_backend', forbidden_driver)
    controls = parameters['solve_controls']
    try:
        heads = []
        for duration in (1., .5):
            model.update_R_in_place(recharge)
            head, info = model.solve(formulation='unconfined', solver='unconfined_picard_kcycle',
                transient=True, initial_head=initial, head_prev=initial,
                K_field=parameters['k_field'], zbot_field=parameters['zbot_field'],
                ztop_field=parameters['ztop_field'], sy=parameters['sy'], ss=parameters['ss'],
                dt=duration, return_info=True, **controls)
            assert info['device_side_picard_fast_path_active']
            assert info['strict_picard_convergence_passed']
            assert 'budget_summary' in info
            assert model._production_transient_session.setup_count == 1
            heads.append(head)
        assert not np.array_equal(heads[0], heads[1])
    finally:
        model.close()
    assert model._production_transient_session is None

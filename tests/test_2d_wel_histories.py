"""Accepted physical WEL/recharge schedules on maintained 2D period drivers."""
from copy import deepcopy

import numpy as np
import pytest

from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('backend,fast', [('unconfined_picard_kcycle', False),
    ('unconfined_picard_kcycle', True), ('unconfined_semismooth_newton_kcycle', False)])
@pytest.mark.parametrize('scheduled', [False, True])
def test_uniform_source_history_matches_exact_storage(device, backend, fast, scheduled):
    """Uniform heads give the exact storage balance with independently varied packages."""
    shape = (6, 6)
    active = np.ones(shape, dtype=np.int32)
    initial = np.full(shape, 5.)
    wells = [((row, column), -.5) for row in range(6) for column in range(6)]
    changed = [((row, column), -1.5) for row in range(6) for column in range(6)]
    schedule = [wells, changed] if scheduled else None
    controls = dict(unconfined_head_tol=1e-10, strict_head_residual_tol=1e-10,
        unconfined_relax=1., omega=1., omega_max=1.)
    if fast:
        controls.update(use_device_transient_fast_path=True, transient_face_operator_enabled=True,
            transient_face_graphs_enabled=True)
    with WarpDarcySolver(nx=6, ny=6, dx=10., device=device) as model:
        model.build_from_fields(T_field=np.full(shape, 5.), R_field=0., active=active,
            bc_mask=np.zeros(shape, dtype=np.int32), bc_values=initial, wel_entries=wells)
        heads, info = model.solve_transient_2d_unconfined(solver=backend, initial_head=initial,
            recharge_rates=np.array([.01, .02]), wel_rates=schedule, k_field=np.ones(shape),
            zbot_field=np.zeros(shape), ztop_field=np.full(shape, 10.), sy=.2, ss=0., dt=1.,
            active=active, bc_mask=np.zeros(shape, dtype=np.int32), bc_values=initial,
            solve_controls=controls)
        expected = [5.025, 5.05 if scheduled else 5.1]
        np.testing.assert_allclose(heads[:, 0, 0], expected, atol=1e-8, rtol=0.)
        assert info['solver_backend'] == backend
        for index, period in enumerate(info['period_infos']):
            assert period['converged']
            assert period['solver_backend'] == backend
            rates = model.accepted_wel_rates(head=heads[index], info=period)
            np.testing.assert_array_equal(rates, np.full(36, -1.5 if scheduled and index else -.5))
            assert period['source_budgets']['recharge']['inflow'] == pytest.approx(36. * (index + 1))
            assert period['source_volume_budgets']['wel']['outflow'] == pytest.approx(54. if scheduled and index else 18.)
            if fast:
                assert period['device_side_picard_fast_path_active']
        previous = info['period_infos'][0]
        modified = deepcopy(previous)
        modified['wel_entry_rates'][0] -= 1.
        with pytest.raises(ValueError):
            model.accepted_wel_rates(head=heads[0], info=modified)
        model.update_wel_in_place(wel_entries=None)
        np.testing.assert_array_equal(model.accepted_wel_rates(head=heads[0], info=previous), np.full(36, -.5))

"""GHB parity under full Picard updates, clipped starts, and device refreshes."""
from pathlib import Path
import json

import numpy as np
import pytest

from DARCY_WARP_PACKAGE.solvers.transient_config import default_solve_controls
from DARCY_WARP_PACKAGE.warped_darcy import _adaptive_dt_should_early_shrink
from working_tests.transient_replay_support import run_warp_transient_replay
from working_tests.transient_replay_mass_balance import compute_replay_mass_balance


def make_case(*, nx, ny, dx, initial_head, ghb_head):
    shape = (ny, nx)
    active = np.ones(shape, dtype=np.int32)
    bc = np.zeros(shape, dtype=np.int32)
    bc[:, [0, -1]] = 1
    initial = np.full(shape, initial_head)
    ghb = np.zeros(shape, dtype=np.int32)
    ghb[ny // 2, 1:-1] = 1
    rng = np.random.default_rng(seed=42)
    return dict(nx=nx, ny=ny, dx=dx, active=active, bc_mask=bc,
        bc_values=initial * bc, initial_head=initial,
        top=np.full(shape, 300.), bottom=np.zeros(shape),
        k=np.exp(rng.uniform(low=np.log(5.), high=np.log(100.), size=shape)),
        ghb_mask=ghb, ghb_head=np.full(shape, ghb_head),
        ghb_width=3. * ghb, ghb_alpha=1., ghb_aq_thickness=300.)


def run_case(*, spatial, controls, device, rates, dt, sy, ss):
    result = run_warp_transient_replay(spatial=spatial,
        recharge_rates=rates, sy=sy, ss=ss, dt=dt, device=device,
        warm_start_mode='artifact_initial', warm_start_head=spatial['initial_head'],
        solve_controls=controls, formulation='unconfined')
    budget = compute_replay_mass_balance(spatial=spatial, recharge_rates=rates,
        sy=sy, ss=ss, dt=dt, formulation='unconfined',
        unconfined_storage_mode=result['unconfined_storage_mode'], warp_result=result,
        min_sat=controls['min_saturated_thickness'])
    for info in result['period_infos']:
        assert info['strict_picard_convergence_passed']
        assert info['adaptive_dt_substep_dts'] == [dt]
        assert info['adaptive_dt_retry_count'] == 0
    assert budget['max_abs_percent_discrepancy'] < 1.e-3
    return result


@pytest.mark.parametrize('device,mixed,graphs', [
    ('cpu', False, False), ('cpu', True, False),
    ('cuda:0', False, True), ('cuda:0', True, False), ('cuda:0', True, True),
])
@pytest.mark.parametrize('initial_head,ghb_head', [(250., 80.), (80., 250.)])
def test_runtime_ghb_matches_startup_and_conserves_water(device, mixed, graphs, initial_head, ghb_head):
    spatial = make_case(nx=24, ny=16, dx=100., initial_head=initial_head, ghb_head=ghb_head)
    controls = default_solve_controls()
    controls.update(save_transient_diagnostics=True, max_levels=3, min_coarse_cells=20)
    arguments = dict(spatial=spatial, device=device,
        rates=np.asarray([.001, .002, .0005]), dt=7., sy=.1, ss=1.e-5)
    baseline = run_case(controls=controls, **arguments)
    profile = json.loads((Path(__file__).resolve().parents[1]
        / 'working_tests/2d_transient_runtime_controls.json').read_text())['runtime']
    assert profile['supports_ghb']
    controls.update(profile['solve_controls'])
    controls.update(transient_mixed_precision_enabled=mixed,
        transient_face_graphs_enabled=graphs)
    candidate = run_case(controls=controls, **arguments)
    assert all(info['transient_mixed_precision'] == mixed for info in candidate['period_infos'])
    assert all(info['picard_relax'] == profile['solve_controls']['unconfined_ghb_relax']
        for info in candidate['period_infos'])
    assert all(info['transient_face_graphs'] == (graphs and device.startswith('cuda'))
        for info in candidate['period_infos'])
    np.testing.assert_allclose(actual=candidate['heads_per_period'],
        desired=baseline['heads_per_period'], rtol=0., atol=2.e-4)
    assert all(info['startup_inner_kcycles'] == 0 for info in candidate['period_infos'])


def test_early_shrink_uses_only_unclipped_contraction_history():
    arguments = dict(tol=1.e-4, outer_iterations_done=10, budget=40, min_outer=6)
    assert _adaptive_dt_should_early_shrink(dh_history=[10.] * 10, **arguments)
    assert not _adaptive_dt_should_early_shrink(
        dh_history=[10.] * 10, update_limit=10., **arguments)
    # Once clipping ends, a slow contraction still triggers the safety net.
    assert _adaptive_dt_should_early_shrink(
        dh_history=[10.] * 8 + [9., 8.9], update_limit=10., **arguments)
    assert not _adaptive_dt_should_early_shrink(
        dh_history=[10.] * 8 + [1., .1], update_limit=10., **arguments)


@pytest.mark.parametrize('device', ['cuda:0'])
def test_clipped_ghb_updates_do_not_force_false_timestep_retries(device):
    spatial = make_case(nx=24, ny=16, dx=100., initial_head=250., ghb_head=80.)
    controls = default_solve_controls()
    controls.update(save_transient_diagnostics=True, max_levels=3, min_coarse_cells=20,
        unconfined_startup_mode='initial_head', omega=1., omega_max=1.,
        adaptive_dt_strict_max_outer=40, transient_mixed_precision_enabled=True)
    result = run_case(spatial=spatial, controls=controls, device=device,
        rates=np.asarray([.001]), dt=7., sy=.1, ss=1.e-5)
    history = result['period_infos'][0]['outer_iteration_summaries']
    assert sum(row['final_max_abs_head_change'] >= 10. - 1.e-10 for row in history) >= 8

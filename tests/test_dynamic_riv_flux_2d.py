"""Dynamic RIV fields and accepted extraction share the nonlinear relation."""
import numpy as np
import pytest
import warp as wp

from DARCY_WARP_PACKAGE.physics.riv_2d import riv_discharge_2d
from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_shared_riv_sign_and_disconnected_relation(device):
    if device.startswith('cuda') and not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    head = np.array([[-1., 1., 1., -1.]])
    stage = np.array([[0., 0., 2., 2.]])
    out = riv_discharge_2d(head=head, stage=stage, bottom=np.zeros((1,4)),
        conductance=np.full((1,4), 5.), mask=np.ones((1,4)), device=device)
    np.testing.assert_array_equal(out, [[0., 5., -5., -10.]])


def test_dynamic_riv_validation_preserves_allocations_and_other_boundaries():
    solver = WarpDarcySolver.__new__(WarpDarcySolver)
    solver.ny, solver.nx = 2, 2
    solver.active_host = np.ones((2, 2), dtype=np.int32)
    solver.bc_mask_host = np.zeros((2, 2), dtype=np.int32)
    solver.drn_mask_host = np.array([[1, 0], [0, 0]])
    solver.update_riv_boundary_2d(mask=np.ones((2,2)), stage=1., bottom=0., conductance=5.)
    stage = solver.riv_stage_host
    cond = solver.riv_cond_host
    solver.update_riv_boundary_2d(mask=np.ones((2,2)), stage=2., bottom=0., conductance=5.)
    assert solver.riv_stage_host is stage
    assert solver.riv_cond_host is cond
    np.testing.assert_array_equal(solver.drn_mask_host, [[1,0],[0,0]])
    for bad in (np.nan, np.ones((1,2))):
        with pytest.raises(ValueError, match='finite scalar or matching'):
            solver.update_riv_boundary_2d(mask=np.ones((2,2)), stage=bad, bottom=0., conductance=5.)
    np.testing.assert_array_equal(solver.riv_stage_host, 2.)
    solver.bc_mask_host[0,0] = 1
    with pytest.raises(ValueError, match='prescribed-head'):
        solver.update_riv_boundary_2d(mask=np.ones((2,2)), stage=2., bottom=0., conductance=5.)


def test_accepted_riv_requires_matching_returned_nonlinear_state():
    solver = WarpDarcySolver.__new__(WarpDarcySolver)
    solver.ny, solver.nx = 1, 2
    solver.device_str = 'cpu'
    solver.riv_mask_host = np.ones((1,2), dtype=np.int32)
    solver.riv_stage_host = np.array([[1., 1.]])
    solver.riv_rbot_host = np.zeros((1,2))
    solver.riv_cond_host = np.full((1,2), 5.)
    solver.T_field_host = np.full((1,2), 999.)
    head = np.array([[2., 0.]])
    info = dict(converged=True, solver_backend='unconfined_semismooth_newton_kcycle',
        transmissivity_array=np.ones((1,2)), saturated_thickness_array=np.ones((1,2)),
        riv_discharge_rate_array=np.array([[5., -5.]]))
    np.testing.assert_array_equal(solver.accepted_riv_discharge_2d(head=head, info=info), [[5.,-5.]])
    with pytest.raises(RuntimeError, match='converged'):
        solver.accepted_riv_discharge_2d(head=head, info={**info, 'converged':False})
    with pytest.raises(RuntimeError, match='lacks transmissivity'):
        solver.accepted_riv_discharge_2d(head=head, info={key:value for key,value in info.items() if key!='transmissivity_array'})
    with pytest.raises(RuntimeError, match='does not match'):
        solver.accepted_riv_discharge_2d(head=head+.1, info=info)


def test_period_budget_preserves_combined_packages_and_unequal_substeps():
    shape = (4, 4)
    ones = np.ones(shape)
    zeros = np.zeros(shape)
    initial = 10.0 * ones
    with WarpDarcySolver(nx=4, ny=4, dx=1.0, device="cpu", use_ghb=True,
                         use_drn=True, use_riv=True, solver_type="kcycle") as solver:
        solver.build_from_fields(
            T_field=8.0 * ones, R_field=zeros, active=ones, bc_mask=zeros,
            bc_values=initial, gh_mask=ones, gh_head=9.0 * ones,
            gh_width=ones, gh_alpha=0.0075, aq_thickness=2.0,
            drn_mask=ones, drn_elev=9.5 * ones, drn_cond=0.1 * ones,
            riv_mask=ones, riv_stage=11.0 * ones, riv_rbot=9.0 * ones, riv_cond=0.2 * ones,
        )
        heads, info = solver.solve_transient_2d_unconfined(
            solver="unconfined_semismooth_newton_kcycle", initial_head=initial,
            recharge_rates=[0.0], k_field=ones, zbot_field=zeros, ztop_field=8.0 * ones,
            sy=0.2, ss=0.025, dt=1.0, active=ones, bc_mask=zeros, bc_values=initial,
            solve_controls={"experimental_max_dt": 0.4},
        )

    # With uniform heads all internal flows vanish. The constant conductances
    # per cell are 0.03 for GHB, 0.1 for DRN and 0.2 for RIV; S=0.025*8.
    old = 10.0
    drain_volume = 0.0
    river_volume = 0.0
    ghb_volume = 0.0
    records = info["experimental_timestep_records"]
    assert [record["dt"] for record in records] == pytest.approx([0.4, 0.4, 0.2])
    for record in records:
        assert record["accepted"]
        dt = record["dt"]
        new = (0.2 * old / dt + 0.03 * 9.0 + 0.1 * 9.5 + 0.2 * 11.0) / (0.2 / dt + 0.33)
        drain_volume += ones.size * dt * 0.1 * (new - 9.5)
        river_volume += ones.size * dt * 0.2 * (11.0 - new)
        ghb_volume += ones.size * dt * 0.03 * (new - 9.0)
        old = new
    np.testing.assert_allclose(heads[0], new, rtol=0.0, atol=2e-5)
    budget = info["period_infos"][0]["budget_summary"]
    assert budget["drn_out"] == pytest.approx(drain_volume, rel=1e-5)
    assert budget["riv_in"] == pytest.approx(river_volume, rel=1e-5)
    assert budget["ghb_out"] == pytest.approx(ghb_volume, rel=1e-5)
    assert budget["total_in"] == pytest.approx(river_volume, rel=1e-5)
    assert budget["total_out"] == pytest.approx(river_volume, rel=1e-5)


def test_failed_newton_with_gated_package_returns_its_own_trial():
    shape = (8, 8)
    ones = np.ones(shape)
    zeros = np.zeros(shape)
    fixed = np.zeros(shape, dtype=np.int32)
    fixed[:, [0, -1]] = 1
    initial = 10.0 * ones
    with WarpDarcySolver(nx=8, ny=8, dx=1.0, device="cpu", use_riv=True) as solver:
        solver.build_from_fields(
            T_field=initial, R_field=zeros, active=ones, bc_mask=fixed, bc_values=initial,
            riv_mask=1 - fixed, riv_stage=zeros, riv_rbot=zeros, riv_cond=5.0 * ones,
        )
        _, info = solver.solve(
            formulation="unconfined", solver="unconfined_semismooth_newton_kcycle",
            initial_head=initial, K_field=ones, zbot_field=zeros, ztop_field=20.0 * ones,
            newton_max_iterations=1, newton_fallback_to_picard=True,
        )
    assert not info["converged"]
    assert info["newton_failure_reason"]
    assert not info["newton_fallback_used"]
    assert info["fallback_state"] == "unavailable_for_drn_riv"


def test_period_driver_preserves_closed_dynamic_river():
    ones = np.ones((4, 4))
    zeros = np.zeros((4, 4))
    initial = 10.0 * ones
    with WarpDarcySolver(nx=4, ny=4, dx=1.0, device="cpu", use_riv=True) as solver:
        solver.build_from_fields(
            T_field=8.0 * ones, R_field=zeros, active=ones, bc_mask=zeros, bc_values=initial,
            riv_mask=ones, riv_stage=12.0 * ones, riv_rbot=8.0 * ones, riv_cond=ones,
        )
        solver.update_riv_boundary_2d(mask=ones, stage=12.0, bottom=8.0, conductance=0.0)
        heads, info = solver.solve_transient_2d_unconfined(
            solver="unconfined_semismooth_newton_kcycle", initial_head=initial,
            recharge_rates=[0.0], k_field=ones, zbot_field=zeros, ztop_field=8.0 * ones,
            sy=0.2, ss=0.025, dt=1.0, active=ones, bc_mask=zeros, bc_values=initial,
        )
    np.testing.assert_array_equal(heads[0], initial)
    assert info["period_infos"][0]["budget_summary"]["riv_in"] == 0.0

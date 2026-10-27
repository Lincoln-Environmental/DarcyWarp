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

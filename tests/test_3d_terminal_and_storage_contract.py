"""Terminal diagnostics and explicit unconfined storage contracts."""
import numpy as np
import pytest

from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D
from test_3d_scientific_audit import device


def make_solver(*, initial):
    shape = initial.shape
    k = np.ones(shape)
    bc = np.zeros(shape, dtype=np.int32)
    bc[..., 0] = bc[..., -1] = 1
    solver = WarpDarcySolver3D(nx=shape[2], ny=shape[1], nz=shape[0], dx=1,
                               device=device(), implementation="fast")
    solver.build_from_K_fields(kx_field=k, ky_field=k, kz_field=k,
        active=np.ones(shape, dtype=np.int32), bc_mask=bc,
        bc_values=initial, rhs=np.zeros(shape), initial_head=initial)
    return solver


@pytest.mark.parametrize("hydrostatic", [False, True])
@pytest.mark.parametrize("transient", [False, True])
def test_exact_initial_and_repeated_solution_zero_cycles(hydrostatic, transient):
    initial = np.full((1, 1, 5), 7.0) if hydrostatic else np.arange(5.0).reshape(1, 1, 5)
    with make_solver(initial=initial) as solver:
        controls = dict(max_cycles=0, abs_tol_min=0.0, rel_tol=1e-12,
                        dh_rms_tol=1e-12)
        if transient:
            controls.update(transient=True, storage_coeff=0.125, dt=2, head_prev=initial)
        for repeat in range(2):
            head, info = solver.solve(**controls)
            assert info["converged"]
            assert info["n_cycles_used"] == 0
            for key in ("r_rms0", "r_rms_end", "h_rms_end", "dh_rms_lastcheck", "dh_max_lastcheck"):
                assert np.isfinite(info[key])
                assert info[key] < 1e-12
            assert info["budget"]["cell_residual_max"] < 1e-12
            np.testing.assert_array_equal(head, initial)
            solver.update_initial_head(initial_head=head)


@pytest.mark.parametrize("cycles", [0, 1])
def test_iteration_limit_has_terminal_residual(cycles):
    initial = np.array([[[4.0, 0.0, 0.0, 0.0, 0.0]]])
    with make_solver(initial=initial) as solver:
        head, info = solver.solve(max_cycles=cycles, max_levels=1,
            nu_pre=1, nu_post=1, nu_coarse=1, abs_tol_min=0, rel_tol=1e-14)
    independent = head[..., :-2] - 2*head[..., 1:-1] + head[..., 2:]
    assert not info["converged"]
    assert info["r_rms_end"] == pytest.approx(np.sqrt(np.mean(independent**2)))
    assert np.isfinite(info["h_rms_end"])


@pytest.mark.parametrize("bad", [dict(sy=0.2, storage_coeff=0.1),
    dict(sy=0.2, storage_diag=np.zeros((1, 1, 5))),
    dict(storage_coeff=0.1), dict(unconfined_storage_mode="confined_volume", storage_coeff=0.1, ss=0.01)])
def test_ignored_storage_combinations_are_rejected(bad):
    initial = np.full((1, 1, 5), 0.5)
    with make_solver(initial=initial) as solver:
        with pytest.raises(ValueError, match="incompatible|unsupported"):
            solver.solve(unconfined=True, transient=True, dt=1, head_prev=initial,
                         zbot_field=np.zeros(initial.shape), **bad)


@pytest.mark.parametrize("implementation", ["fast","classic"])
def test_legacy_volume_initial_head_is_fixed_previous_timestep(implementation):
    shape = (1,1,1)
    zero = np.zeros(shape)
    with WarpDarcySolver3D(nx=1,ny=1,nz=1,dx=1,device=device(),implementation=implementation) as solver:
        solver.build_from_K_fields(kx_field=zero,ky_field=zero,kz_field=zero,
            active=np.ones(shape,dtype=np.int32),bc_mask=np.zeros(shape,dtype=np.int32),
            bc_values=zero,rhs=np.full(shape,.05),initial_head=np.full(shape,.5))
        h,info = solver.solve(unconfined=True,transient=True,dt=1,storage_coeff=.1,
            unconfined_storage_mode="confined_volume",zbot_field=zero,
            unconfined_head_tol=1e-9,final_head_residual_tol=1e-9,
            unconfined_max_picard_iter=100,chebyshev_enabled=False)
    assert info["converged"]
    assert h.item() == pytest.approx(1,abs=2e-8)
    assert info["budget"]["storage_uptake"] == pytest.approx(.05,abs=2e-9)


@pytest.mark.parametrize("bad", [-.1,np.inf,np.nan])
def test_scalar_specific_storage_is_validated(bad):
    initial = np.full((1,1,5),.5)
    with make_solver(initial=initial) as solver:
        with pytest.raises(ValueError,match="finite|>= 0"):
            solver.solve(unconfined=True,transient=True,dt=1,head_prev=initial,
                sy=.2,ss=bad,zbot_field=np.zeros(initial.shape))


@pytest.mark.parametrize("implementation", ["fast","classic"])
@pytest.mark.parametrize("old,new", [(4.,3.8),(5.,3.8),(-1.,.3)])
def test_cartesian_zero_tangent_storage_states_can_start_a_valid_step(implementation,old,new):
    shape = (1,1,1)
    zero = np.zeros(shape)
    source = .3*(np.clip(new,0,4)-np.clip(old,0,4))
    with WarpDarcySolver3D(nx=1,ny=1,nz=1,dx=1,dz=4,device=device(),implementation=implementation) as solver:
        solver.build_from_K_fields(kx_field=zero,ky_field=zero,kz_field=zero,
            active=np.ones(shape,dtype=np.int32),bc_mask=np.zeros(shape,dtype=np.int32),
            bc_values=zero,rhs=np.full(shape,source),initial_head=np.full(shape,old))
        head,info = solver.solve(unconfined=True,transient=True,dt=1,sy=.3,head_prev=np.full(shape,old),
            zbot_field=zero,unconfined_head_tol=1e-9,final_head_residual_tol=1e-9)
    assert info["converged"]
    assert head.item() == pytest.approx(new,abs=1e-10)
    assert info["budget"]["storage_uptake"] == pytest.approx(source,abs=1e-12)

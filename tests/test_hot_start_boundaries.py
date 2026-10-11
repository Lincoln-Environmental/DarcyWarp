"""Native drain accounting and selected hydrostatic coastal subface parity."""
import numpy as np
import pytest


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_native_coastal_jacobian_and_true_norm(device):
    from DARCY_WARP_PACKAGE.nonlinear import from_arrays, NonlinearOperator2D
    from DARCY_WARP_PACKAGE.nonlinear.coastal import CoastalSubfaces2D
    shape = (3, 4)
    coast = CoastalSubfaces2D(rows=np.array([1, 1, 1]), columns=np.array([0, 0, 0]),
        head_m=np.array([0., 1., 3.]), conductance_m2_per_time=np.array([4., 2., 2.]),
        submerged=np.array([1, 0, 0]))
    ctx = from_arrays(nx=4, ny=3, dx=5., K=np.ones(shape), zbot=np.full(shape, -5.),
        ztop=np.full(shape, 5.), active=np.ones(shape, int), dirichlet_mask=np.zeros(shape, int),
        dirichlet_values=np.zeros(shape), R_field=np.zeros(shape), coastal_boundary=coast, device=device)
    with NonlinearOperator2D(ctx) as op:
        head = np.full(shape, 2.)
        residual = op.residual(head=head).numpy().copy()
        assert residual[1, 0] == pytest.approx(10.)
        assert op.current_reduced_norms().l2 == pytest.approx(np.linalg.norm(residual))
        vector = np.arange(12).reshape(shape)/10.
        analytic = op.jacobian_vector(head=head, vector=vector).numpy().copy()
        eps = 1e-6
        fd = (op.residual(head=head+eps*vector).numpy().copy()
              -op.residual(head=head-eps*vector).numpy().copy())/(2*eps)
        np.testing.assert_allclose(analytic, fd, rtol=1e-6, atol=1e-7)
        _, diagonal = op.freeze_picard_device(head=op.head_device)
        assert diagonal.numpy()[1, 0] == pytest.approx(6.)
    np.testing.assert_allclose(coast.fluxes(head=np.full(shape, -1.)), [-4., 0., 0.])


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('boundary_kind', ['drn', 'riv'])
@pytest.mark.parametrize('restriction', ['average', 'conservative_sum'])
@pytest.mark.parametrize('warm_steps', [0, 3])
def test_native_gated_solver_budget_matches_independent_recharge(device, boundary_kind, restriction, warm_steps):
    from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver
    shape = (6, 8)
    mask = np.zeros(shape, int)
    mask[:, -1] = 1
    zero = np.zeros(shape)
    fields = dict(T_field=np.ones(shape)*50., R_field=np.ones(shape)*.001,
        active=np.ones(shape, int), bc_mask=np.zeros(shape, int), bc_values=zero)
    if boundary_kind == 'drn':
        fields.update(drn_mask=mask, drn_elev=np.zeros(shape), drn_cond=np.ones(shape)*10.)
    else:
        fields.update(riv_mask=mask, riv_stage=zero, riv_rbot=np.ones(shape)*-3., riv_cond=np.ones(shape)*10.)
    model = WarpDarcySolver(nx=8, ny=6, dx=5., device=device,
        use_drn=boundary_kind == 'drn', use_riv=boundary_kind == 'riv', diag_preconditioner_backend='device')
    try:
        model.build_from_fields(**fields)
        head, info = model.solve(formulation='unconfined', solver='unconfined_semismooth_newton_kcycle',
            initial_head=np.ones(shape)*.5, K_field=np.ones(shape)*10., zbot_field=np.ones(shape)*-5.,
            ztop_field=np.ones(shape)*5., newton_fallback_to_picard=False, newton_residual_rms_tolerance=1e-8,
            newton_fgmres_relative_tolerance=1e-7, newton_fgmres_max_iterations=80, min_coarse_cells=1,
            newton_max_iterations=20, max_levels=3, newton_coarse_residual_restriction=restriction,
            newton_picard_warm_start_iterations=warm_steps)
        assert info['converged'], info['newton_failure_reason']
        assert not info['newton_fallback_used']
        independent = float((10.*head[mask != 0]).sum())
        assert independent == pytest.approx(.001*25*48, abs=1e-7)
        assert info['budget_summary'][boundary_kind+'_out'] == pytest.approx(independent)
        assert abs(info['budget_summary']['in_minus_out']) < 1e-7
    finally:
        model.close()


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_integrated_residual_restriction_preserves_unsigned_flux(device):
    import warp as wp
    from DARCY_WARP_PACKAGE.solvers.newton_kernels import restrict_integrated_flux_kernel
    from DARCY_WARP_PACKAGE.nonlinear.kernels import WP_FLOAT
    fine = np.arange(1., 16.).reshape(3, 5)
    active = np.ones(fine.shape, np.int32)
    active[0, 1] = 0
    prescribed = np.zeros(fine.shape, np.int32)
    prescribed[2, 4] = 1
    coarse = wp.zeros((2, 3), dtype=WP_FLOAT, device=device)
    wp.launch(kernel=restrict_integrated_flux_kernel, dim=(2, 3),
        inputs=[wp.array(fine, dtype=WP_FLOAT, device=device),
            wp.array(active, dtype=wp.int32, device=device),
            wp.array(prescribed, dtype=wp.int32, device=device), coarse, 5, 3, 3, 2], device=device)
    result = coarse.numpy()
    assert result.sum() == pytest.approx(fine[(active != 0) & (prescribed == 0)].sum())
    assert result[0, 0] == pytest.approx(1.+6.+7.)


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('preconditioner', ['steady_cpu_amg', 'steady_cpu_jacobian_amg', 'steady_cpu_ilu', 'steady_cpu_sparse_lu'])
def test_sparse_amg_prepares_native_steady_solution_with_true_fgmres_acceptance(device, preconditioner):
    if preconditioner in ('steady_cpu_amg', 'steady_cpu_jacobian_amg'):
        pytest.importorskip('pyamg')
    from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver
    shape = (12, 16)
    mask = np.zeros(shape, int)
    mask[:, -1] = 1
    zero = np.zeros(shape)
    model = WarpDarcySolver(nx=16, ny=12, dx=5., device=device, use_drn=True,
        diag_preconditioner_backend='device')
    try:
        model.build_from_fields(T_field=np.ones(shape)*50., R_field=np.ones(shape)*.001,
            active=np.ones(shape, int), bc_mask=mask, bc_values=zero,
            drn_mask=np.roll(mask, -3, axis=1), drn_elev=zero, drn_cond=np.ones(shape)*2.5)
        head, info = model.solve(formulation='unconfined', solver='unconfined_semismooth_newton_kcycle',
            initial_head=np.ones(shape), K_field=np.ones(shape)*10., zbot_field=np.ones(shape)*-5.,
            ztop_field=np.ones(shape)*5., newton_preconditioner=preconditioner,
            newton_fgmres_relative_tolerance=1e-6, newton_fgmres_max_iterations=80,
            newton_residual_rms_tolerance=1e-8, newton_fallback_to_picard=False)
        assert info['converged'], info['newton_failure_reason']
        assert info['preconditioner_full_vector_transfer_bytes'] > 0
        for linear in info['linear_history']:
            assert linear['converged']
            assert linear['final_true_residual'] <= max(linear['requested_absolute_tolerance'],
                linear['requested_relative_tolerance']*linear['initial_true_residual'])
        assert abs(info['budget_summary']['in_minus_out']) < 1e-6
    finally:
        model.close()


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_sparse_steady_jacobian_matches_native_strong_gradient_action(device):
    from DARCY_WARP_PACKAGE.nonlinear import from_arrays, NonlinearOperator2D
    from DARCY_WARP_PACKAGE.solvers.steady_sparse_preconditioner import steady_jacobian_matrix
    shape = (8, 10)
    y, x = np.mgrid[:8, :10]
    bottom = x*2.+y*.2-5.
    head = bottom+2.+.2*np.sin(x+y)
    ctx = from_arrays(nx=10, ny=8, dx=5., K=np.ones(shape)*10., zbot=bottom, ztop=bottom+5.,
        active=np.ones(shape, int), dirichlet_mask=np.zeros(shape, int), dirichlet_values=np.zeros(shape),
        R_field=np.ones(shape)*.001, device=device, min_sat=.001)
    vector = np.cos(x+2*y)
    with NonlinearOperator2D(ctx) as op:
        native = op.jacobian_vector(head=head, vector=vector).numpy().copy()
        matrix = steady_jacobian_matrix(context=ctx, head=head, boundary_diagonal=np.zeros(shape))
        np.testing.assert_allclose((matrix@vector.ravel()).reshape(shape), native, rtol=1e-11, atol=1e-10)


def test_sparse_preconditioning_cannot_be_used_for_transient_timesteps():
    from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver
    shape = (2, 3)
    model = WarpDarcySolver(nx=3, ny=2, dx=5., device='cpu')
    try:
        model.build_from_fields(T_field=np.ones(shape), R_field=np.zeros(shape),
            active=np.ones(shape, int), bc_mask=np.zeros(shape, int), bc_values=np.zeros(shape))
        with pytest.raises(ValueError, match='steady-preparation'):
            model.solve(formulation='unconfined', solver='unconfined_semismooth_newton_kcycle',
                initial_head=np.ones(shape), K_field=np.ones(shape), zbot_field=np.zeros(shape),
                ztop_field=np.ones(shape)*5., transient=True, head_prev=np.ones(shape), dt=1., sy=.1,
                newton_preconditioner='steady_cpu_sparse_lu')
    finally:
        model.close()

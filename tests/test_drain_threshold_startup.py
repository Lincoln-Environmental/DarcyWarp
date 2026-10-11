"""Drain-threshold startup must use a nonsingular semismooth selection."""
import numpy as np
import pytest


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_recharged_all_drain_domain_starts_at_exact_drain_elevation(device):
    """Flat closed domain: H=DEM initially, H=DEM+R*A/C at steady state."""
    from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver

    shape = (40, 60)
    elevation = np.full(shape, 4.0)
    recharge = 0.3 / 365.25
    conductance = 250.0
    initial = elevation.copy()
    model = WarpDarcySolver(nx=shape[1], ny=shape[0], dx=5.0,
        device=device, use_drn=True, diag_preconditioner_backend="device")
    try:
        model.build_from_fields(T_field=np.full(shape, 50.0),
            R_field=np.full(shape, recharge), active=np.ones(shape, dtype=int),
            bc_mask=np.zeros(shape, dtype=int), bc_values=np.zeros(shape),
            drn_mask=np.ones(shape, dtype=int), drn_elev=elevation,
            drn_cond=np.full(shape, conductance))
        head, info = model.solve(formulation="unconfined",
            solver="unconfined_semismooth_newton_kcycle", initial_head=initial,
            reference_transmissivity_field=np.full(shape, 50.0),
            zbot_field=np.full(shape, -50.0), ztop_field=elevation,
            unconfined_startup_mode="confined_pre_solve", max_levels=7,
            newton_fallback_to_picard=False, inner_implementation="fast",
            newton_profile_timing=True)
        assert info["converged"], info["newton_failure_reason"]
        assert info["preconditioner_implementation"] == "fast"
        assert len(info["multigrid_level_shapes"]) > 1
        assert info["preconditioner_setup_wall_s"] > 0.0
        assert info["preconditioner_apply_wall_s"] > 0.0
        if device.startswith("cuda"):
            assert info["preconditioner_setup_gpu_s"] > 0.0
            assert info["preconditioner_apply_gpu_s"] > 0.0
        else:
            assert info["preconditioner_apply_gpu_s"] is None
        np.testing.assert_allclose(head, elevation + recharge*25.0/conductance,
            rtol=0.0, atol=1e-8)
        np.testing.assert_array_equal(initial, elevation)
        independent_outflow = float(np.sum(conductance*np.maximum(head-elevation, 0.0)))
        assert independent_outflow == pytest.approx(recharge*25.0*elevation.size, abs=1e-7)
        assert info["budget_summary"]["drn_out"] == pytest.approx(independent_outflow)
        assert not info["newton_fallback_used"]
        workspace = model._resource_owner.get_experimental_workspace("semismooth_newton_kcycle")
        cached_graphs = workspace.graph_cache
        graph_count = len(cached_graphs)
        model.update_uniform_recharge_in_place(recharge_rate=2.0*recharge)
        repeated, repeated_info = model.solve(formulation="unconfined",
            solver="unconfined_semismooth_newton_kcycle", initial_head=head,
            reference_transmissivity_field=np.full(shape, 50.0),
            zbot_field=np.full(shape, -50.0), ztop_field=elevation,
            unconfined_startup_mode="initial_head", max_levels=7,
            newton_residual_rms_tolerance=1.0e-10,
            newton_fallback_to_picard=False, inner_implementation="fast",
            newton_profile_timing=True)
        assert repeated_info["converged"]
        assert repeated_info["newton_workspace_reused"]
        assert workspace.graph_cache is cached_graphs
        np.testing.assert_allclose(repeated, elevation+2.0*recharge*25.0/conductance,
            rtol=0.0, atol=1e-8)
        assert repeated_info["budget_summary"]["drn_out"] == pytest.approx(
            2.0*recharge*25.0*elevation.size, abs=1e-7)
        if device.startswith("cuda"):
            assert graph_count > 0
            assert repeated_info["kcycle_graph_capture_fallback_entries"] == 0
    finally:
        model.close()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_drain_threshold_jacobian_matches_active_one_sided_derivative(device):
    from DARCY_WARP_PACKAGE.nonlinear import from_arrays, NonlinearOperator2D

    shape = (3, 4)
    elevation = np.full(shape, 4.0)
    conductance = 250.0
    context = from_arrays(nx=shape[1], ny=shape[0], dx=5.0,
        K=np.ones(shape), zbot=np.full(shape, -50.0), ztop=elevation,
        active=np.ones(shape, dtype=int), dirichlet_mask=np.zeros(shape, dtype=int),
        dirichlet_values=np.zeros(shape), R_field=np.zeros(shape),
        drn_mask=np.ones(shape, dtype=int), drn_elev=elevation,
        drn_cond=np.full(shape, conductance), device=device)
    with NonlinearOperator2D(context) as operator:
        base = operator.residual(head=elevation).numpy().copy()
        np.testing.assert_allclose(base, 0.0, atol=1e-10)
        vector = np.ones(shape)
        analytic = operator.jacobian_vector(head=elevation, vector=vector).numpy().copy()
        epsilon = 1e-6
        forward = (operator.residual(head=elevation+epsilon*vector).numpy().copy()-base)/epsilon
        np.testing.assert_allclose(analytic, forward, rtol=1e-7, atol=1e-5)
        np.testing.assert_allclose(analytic, conductance, rtol=0.0, atol=1e-9)
        operator.set_head(head=elevation)
        _, diagonal = operator.freeze_picard_device(head=operator.head_device)
        np.testing.assert_allclose(diagonal.numpy(), conductance, rtol=0.0, atol=1e-9)
        # Negative perturbations remain drain-inactive; this is a genuine kink.
        below = operator.residual(head=elevation-epsilon*vector).numpy().copy()
        np.testing.assert_allclose(below, 0.0, atol=1e-9)
        inactive_jacobian = operator.jacobian_vector(
            head=elevation-epsilon*vector, vector=vector).numpy().copy()
        np.testing.assert_allclose(inactive_jacobian, 0.0, atol=1e-9)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("drain_equality_active", [False, True])
def test_fixed_t_preparation_preserves_native_gated_sources(device, drain_equality_active):
    """Confined R/GHB/DRN/RIV/coast rows agree with the capped native operator."""
    import warp as wp
    from DARCY_WARP_PACKAGE.nonlinear import from_arrays, NonlinearOperator2D
    from DARCY_WARP_PACKAGE.nonlinear.coastal import CoastalSubfaces2D
    from DARCY_WARP_PACKAGE.nonlinear.kernels import WP_FLOAT

    shape = (3, 4)
    head = np.broadcast_to(np.array([-2.0, 0.0, 2.0, 4.0]), shape).copy()
    mask = np.ones(shape, dtype=int)
    zero = np.zeros(shape)
    coast = CoastalSubfaces2D(rows=np.array([0, 0, 1]), columns=np.array([0, 1, 0]),
        head_m=np.array([0.0, 1.0, 0.0]), conductance_m2_per_time=np.array([100.0, 10.0, 10.0]),
        submerged=np.array([1, 0, 0]))
    context = from_arrays(nx=shape[1], ny=shape[0], dx=5.0,
        K=np.full(shape, 1.25), zbot=np.full(shape, -50.0), ztop=np.full(shape, -10.0),
        active=mask, dirichlet_mask=np.zeros(shape, dtype=int), dirichlet_values=zero,
        R_field=np.full(shape, 0.001), ghb_mask=mask, ghb_external_head=np.full(shape, 2.0),
        ghb_factor=np.full(shape, 0.03), drn_mask=mask, drn_elev=zero,
        drn_cond=np.full(shape, 250.0), riv_mask=mask, riv_stage=np.full(shape, 3.0),
        riv_rbot=np.full(shape, -1.0), riv_cond=np.full(shape, 10.0),
        coastal_boundary=coast, device=device)
    with NonlinearOperator2D(context) as operator:
        native = operator.residual(head=head).numpy().copy()
        fixed_t = wp.array(np.full(shape, 50.0), dtype=WP_FLOAT, device=device)
        operator.freeze_confined_device(transmissivity=fixed_t, head=operator.head_device,
            drain_equality_active=drain_equality_active)
        result = operator.confined_residual_device(head=operator.head_device,
            out=operator.residual_device_array, drain_equality_active=drain_equality_active)
        np.testing.assert_allclose(result.numpy(), native, rtol=1e-12, atol=1e-10)
        assert operator.current_reduced_norms().l2 == pytest.approx(np.linalg.norm(native))


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_newly_activated_drain_stalls_smooth_newton_but_exact_gated_model_converges(device):
    """A nearly contacted drain activates under a nonlocal groundwater rise."""
    from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver
    from DARCY_WARP_PACKAGE.nonlinear.coastal import CoastalSubfaces2D
    shape = (1, 6)
    initial = np.zeros(shape)
    initial[0, 1] = .0008
    mask = np.zeros(shape, int)
    mask[0, 1] = 1
    elevations = np.full(shape, .0008+1e-9)
    recharge = .3/365.25
    coast = CoastalSubfaces2D(rows=np.array([0]), columns=np.array([0]), head_m=np.zeros(1),
        conductance_m2_per_time=np.ones(1), submerged=np.ones(1, int))
    outcomes = {}
    for enabled in (False, True):
        model = WarpDarcySolver(nx=shape[1], ny=shape[0], dx=5., device=device,
            use_drn=True, diag_preconditioner_backend='device')
        try:
            model.build_from_fields(T_field=np.full(shape, 50.), R_field=np.full(shape, recharge),
                active=np.ones(shape, int), bc_mask=np.zeros(shape, int), bc_values=np.zeros(shape),
                drn_mask=mask, drn_elev=elevations, drn_cond=np.full(shape, 250.), coastal_boundary=coast)
            head, info = model.solve(formulation='unconfined', solver='unconfined_semismooth_newton_kcycle',
                initial_head=initial.copy(), reference_transmissivity_field=np.full(shape, 50.),
                zbot_field=np.full(shape, -50.), ztop_field=np.full(shape, .1),
                newton_max_iterations=10, newton_fgmres_relative_tolerance=1e-9,
                newton_gated_model_enabled=enabled, newton_fallback_to_picard=False)
            outcomes[enabled] = (head, info)
        finally:
            model.close()
    failed_head, failed = outcomes[False]
    assert not failed['converged'] and failed['newton_failure_reason'] == 'line_search_failed'
    np.testing.assert_array_equal(failed_head, initial)
    head, repaired = outcomes[True]
    assert repaired['converged'], repaired['newton_failure_reason']
    assert repaired['gated_corrective_active_set_iterations'] > 0
    assert repaired['true_nonlinear_residual_rms'] <= 1e-6
    assert head[0, 1] > elevations[0, 1]
    outflow = head[0, 0]+250.*(head[0, 1]-elevations[0, 1])
    assert outflow == pytest.approx(recharge*25.*initial.size, abs=1e-7)
    assert repaired['budget_summary']['drn_out'] > 0.
    assert not repaired['newton_fallback_used']


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_exact_gated_newton_model_keeps_constant_t_interior_and_native_fluxes(device):
    import warp as wp
    from DARCY_WARP_PACKAGE.nonlinear import from_arrays, NonlinearOperator2D
    from DARCY_WARP_PACKAGE.nonlinear.coastal import CoastalSubfaces2D
    from DARCY_WARP_PACKAGE.nonlinear.kernels import WP_FLOAT
    shape = (3, 4)
    head = np.broadcast_to(np.array([-2., 0., 2., 4.]), shape).copy()
    change = np.broadcast_to(np.array([3., -1., -5., 2.]), shape).copy()
    mask, zero = np.ones(shape, int), np.zeros(shape)
    coast = CoastalSubfaces2D(rows=np.array([0, 0, 1]), columns=np.array([0, 1, 0]),
        head_m=np.array([0., 1., 0.]), conductance_m2_per_time=np.array([100., 10., 10.]), submerged=np.array([1, 0, 0]))
    context = from_arrays(nx=4, ny=3, dx=5., K=np.full(shape, 1.25), zbot=np.full(shape, -50.),
        ztop=np.full(shape, -10.), active=mask, dirichlet_mask=np.zeros(shape, int), dirichlet_values=zero,
        R_field=np.full(shape, .001), drn_mask=mask, drn_elev=zero, drn_cond=np.full(shape, 250.),
        riv_mask=mask, riv_stage=np.full(shape, 3.), riv_rbot=np.full(shape, -1.), riv_cond=np.full(shape, 10.),
        coastal_boundary=coast, device=device)
    with NonlinearOperator2D(context) as operator:
        original = operator.residual(head=head).numpy().copy()
        proposed = operator.residual(head=head+change).numpy().copy()
        current = wp.array(head, dtype=WP_FLOAT, device=device)
        branch = wp.array(head+change, dtype=WP_FLOAT, device=device)
        vector = wp.array(change, dtype=WP_FLOAT, device=device)
        action = operator.jacobian_vector_device(head=current, vector=vector, boundary_head=branch).numpy().copy()
        offset = wp.zeros(shape, dtype=WP_FLOAT, device=device)
        operator.gated_model_rhs_device(head=current, branch_head=branch, out=offset)
        np.testing.assert_allclose(action-offset.numpy(), proposed-original, rtol=1e-12, atol=1e-9)
        count = wp.zeros(1, dtype=wp.int32, device=device)
        assert operator.gated_branch_changes_device(first=current, second=branch, count=count) == 10

"""Drain-threshold startup must use a nonsingular semismooth selection."""
import numpy as np
import pytest


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_recharged_all_drain_domain_starts_at_exact_drain_elevation(device):
    """Flat closed domain: H=DEM initially, H=DEM+R*A/C at steady state."""
    from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver

    shape = (12, 16)
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
            newton_fallback_to_picard=False)
        assert info["converged"], info["newton_failure_reason"]
        np.testing.assert_allclose(head, elevation + recharge*25.0/conductance,
            rtol=0.0, atol=1e-8)
        np.testing.assert_array_equal(initial, elevation)
        independent_outflow = float(np.sum(conductance*np.maximum(head-elevation, 0.0)))
        assert independent_outflow == pytest.approx(recharge*25.0*elevation.size, abs=1e-7)
        assert info["budget_summary"]["drn_out"] == pytest.approx(independent_outflow)
        assert not info["newton_fallback_used"]
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

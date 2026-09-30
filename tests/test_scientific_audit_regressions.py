"""Physical budget and disconnected-cell regressions from the SHANZ audit."""

import numpy as np
import pytest
import warp as wp

from DARCY_WARP_PACKAGE.nonlinear import NonlinearOperator2D, from_arrays
from DARCY_WARP_PACKAGE.nonlinear.reference import nonlinear_residual_host
from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("datum", [0.0, 100.0])
def test_isolated_transient_cell_has_only_physical_storage(device, datum):
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA is unavailable")
    shape = (3, 3)
    active = np.zeros(shape, dtype=np.int32)
    active[1, 1] = 1
    previous = np.full(shape, 8.0 + datum)
    context = from_arrays(nx=3, ny=3, dx=10.0,
        K=np.ones(shape), zbot=np.full(shape, datum), ztop=np.full(shape, 10.0 + datum),
        active=active, dirichlet_mask=np.zeros(shape, dtype=np.int32),
        dirichlet_values=np.zeros(shape), R_field=np.zeros(shape),
        transient=True, sy=0.1, ss=1.0e-5, dt=0.04, head_prev=previous, device=device)
    operator = NonlinearOperator2D(context)
    try:
        assert operator.residual(head=previous).numpy()[1, 1] == pytest.approx(0.0, abs=1e-12)
        assert nonlinear_residual_host(head=previous, ctx=context)[1, 1] == pytest.approx(0.0, abs=1e-12)
        trial = previous.copy()
        trial[1, 1] += 0.01
        expected = operator.exact_storage_terms(head=trial).total[1, 1]
        assert operator.residual(head=trial).numpy()[1, 1] == pytest.approx(expected, rel=1e-10)
        direction = np.zeros(shape)
        direction[1, 1] = 1.0
        expected_derivative = (0.1 + 1.0e-5 * 8.01) * 100.0 / 0.04
        assert operator.jacobian_vector(head=trial, vector=direction).numpy()[1, 1] == pytest.approx(
            expected_derivative, rel=1e-10)
    finally:
        operator.close()


@pytest.mark.parametrize("stage", [8.5, 9.0, 10.0])
def test_transient_riv_is_present_in_accepted_newton_budget(stage):
    shape = (5, 5)
    previous = np.full(shape, 8.99)
    bottom = np.full(shape, 9.2 if stage == 10.0 else 8.0)
    with WarpDarcySolver(nx=5, ny=5, dx=10.0, device="cpu", use_riv=True,
                          solver_type="kcycle", diag_preconditioner_backend="device") as solver:
        solver.build_from_fields(T_field=previous, R_field=np.zeros(shape),
            active=np.ones(shape, dtype=np.int32), bc_mask=np.zeros(shape, dtype=np.int32),
            bc_values=previous, riv_mask=np.ones(shape, dtype=np.int32),
            riv_stage=np.full(shape, stage), riv_rbot=bottom, riv_cond=np.full(shape, 100.0))
        head, info = solver.solve(formulation="unconfined",
            solver="unconfined_semismooth_newton_kcycle", initial_head=previous,
            K_field=np.ones(shape), zbot_field=np.zeros(shape), ztop_field=np.full(shape, 12.0),
            transient=True, sy=0.1, ss=1.0e-5, dt=0.0001 if stage == 10.0 else 0.04, head_prev=previous,
            return_info=True, newton_fallback_to_picard=False, max_levels=2, min_coarse_cells=1)
        assert info["converged"]
        q = 100.0 * (np.maximum(head, bottom) - stage)
        budget = info["budget_summary"]
        assert budget["riv_in"] == pytest.approx(float(np.maximum(-q, 0.0).sum()), abs=1e-9)
        assert budget["riv_out"] == pytest.approx(float(np.maximum(q, 0.0).sum()), abs=1e-9)
        assert abs(budget["in_minus_out"]) < 1e-4


@pytest.mark.parametrize("recharge", [0.0, 0.05])
def test_isolated_cells_are_pruned_from_transient_domain(recharge):
    """Pruned cells are outside the model, including its storage/recharge ledger."""
    shape = (3, 3)
    active = np.zeros(shape, dtype=np.int32)
    active[1, 1] = 1
    previous = np.full(shape, 8.0)
    with WarpDarcySolver(nx=3, ny=3, dx=10.0, device="cpu", solver_type="kcycle",
                          diag_preconditioner_backend="device") as solver:
        solver.build_from_fields(T_field=previous, R_field=active * recharge,
            active=active, bc_mask=np.zeros(shape, dtype=np.int32), bc_values=previous)
        assert not np.any(solver.active_host)
        assert not np.any(solver.R_field_host)
        head, info = solver.solve(formulation="unconfined",
            solver="unconfined_semismooth_newton_kcycle", initial_head=previous,
            K_field=np.ones(shape), zbot_field=np.zeros(shape), ztop_field=np.full(shape, 10.0),
            transient=True, sy=0.1, ss=1.0e-5, dt=0.04, head_prev=previous,
            return_info=True, newton_fallback_to_picard=False, max_levels=1, min_coarse_cells=1)
        assert info["converged"]
        assert not np.any(head)
        budget = info["budget_summary"]
        assert budget["rcha_in"] == pytest.approx(0.0, abs=1e-12)
        assert budget["storage_in"] == pytest.approx(0.0, abs=1e-12)
        assert budget["storage_out"] == pytest.approx(0.0, abs=1e-12)

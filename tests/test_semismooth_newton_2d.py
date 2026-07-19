"""Stage-2 regression tests for experimental semismooth Newton--K-cycle."""

from __future__ import annotations

import gc
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("WARP_CACHE_PATH", str(Path("/tmp/darcywarp-warp-cache")))


def _operator_context(
    *,
    transient: bool = False,
    threshold_head: float | None = None,
    device: str = "cpu",
):
    from DARCY_WARP_PACKAGE.nonlinear import from_arrays

    ny, nx = 6, 7
    y, x = np.mgrid[:ny, :nx]
    bottom = (0.1 * x + 0.05 * y).astype(np.float64)
    top = bottom + 10.0
    conductivity = np.exp(np.linspace(-1.0, 1.0, nx))[None, :] * np.ones((ny, 1))
    active = np.ones((ny, nx), dtype=np.int32)
    active[2, 3] = 0
    prescribed = np.zeros((ny, nx), dtype=np.int32)
    prescribed[:, 0] = 1
    prescribed[:, -1] = 1
    values = np.zeros((ny, nx), dtype=np.float64)
    values[:, 0] = 8.0
    values[:, -1] = 6.0
    gh_mask = np.zeros((ny, nx), dtype=np.int32)
    gh_mask[1:-1, 2] = 1
    gh_head = np.full((ny, nx), 7.0)
    gh_factor = np.zeros((ny, nx))
    gh_factor[gh_mask != 0] = 0.03
    previous = bottom + 4.0
    ctx = from_arrays(
        nx=nx,
        ny=ny,
        dx=25.0,
        K=conductivity,
        zbot=bottom,
        ztop=top,
        active=active,
        dirichlet_mask=prescribed,
        dirichlet_values=values,
        R_field=np.full((ny, nx), 2.0e-5),
        ghb_mask=gh_mask,
        ghb_external_head=gh_head,
        ghb_factor=gh_factor,
        sy=0.18,
        ss=2.0e-4,
        head_prev=previous if transient else None,
        dt=3.0 if transient else None,
        transient=transient,
        min_sat=0.1,
        device=device,
    )
    head = bottom + 4.5 + 0.1 * np.sin(x + y)
    if threshold_head is not None:
        head[3, 3] = bottom[3, 3] + float(threshold_head)
    head[prescribed != 0] = values[prescribed != 0]
    head[active == 0] = 0.0
    return ctx, head


@pytest.mark.parametrize("transient", [False, True])
def test_analytic_jv_matches_directional_finite_difference_away_from_kinks(transient):
    from DARCY_WARP_PACKAGE.nonlinear import NonlinearOperator2D

    ctx, head = _operator_context(transient=transient)
    rng = np.random.default_rng(44)
    vector = rng.normal(size=head.shape)
    vector[~ctx.free_mask] = 0.0
    epsilon = 1.0e-7
    op = NonlinearOperator2D(ctx)
    analytic = np.asarray(op.jacobian_vector(head, vector).numpy()).copy()
    plus = np.asarray(op.residual(head + epsilon * vector).numpy()).copy()
    minus = np.asarray(op.residual(head - epsilon * vector).numpy()).copy()
    finite_difference = (plus - minus) / (2.0 * epsilon)
    np.testing.assert_allclose(
        analytic[ctx.free_mask],
        finite_difference[ctx.free_mask],
        rtol=2.0e-6,
        atol=2.0e-6,
    )
    op.close()


def test_cuda_analytic_jv_matches_directional_finite_difference_when_available():
    import warp as wp
    from DARCY_WARP_PACKAGE.nonlinear import NonlinearOperator2D

    if not wp.is_cuda_available():
        pytest.skip("CUDA is not available")
    ctx, head = _operator_context(transient=True, device="cuda:0")
    rng = np.random.default_rng(91)
    vector = rng.normal(size=head.shape)
    vector[~ctx.free_mask] = 0.0
    epsilon = 1.0e-7
    op = NonlinearOperator2D(ctx)
    analytic = np.asarray(op.jacobian_vector(head, vector).numpy()).copy()
    plus = np.asarray(op.residual(head + epsilon * vector).numpy()).copy()
    minus = np.asarray(op.residual(head - epsilon * vector).numpy()).copy()
    finite_difference = (plus - minus) / (2.0 * epsilon)
    np.testing.assert_allclose(
        analytic[ctx.free_mask],
        finite_difference[ctx.free_mask],
        rtol=2.0e-6,
        atol=2.0e-6,
    )
    op.close()


@pytest.mark.parametrize("relative_head", [0.1, 10.0])
def test_generalized_derivative_is_zero_exactly_at_flow_clip_threshold(relative_head):
    from DARCY_WARP_PACKAGE.nonlinear import NonlinearOperator2D

    ctx, head = _operator_context(threshold_head=relative_head)
    vector = np.zeros(head.shape)
    vector[3, 3] = 1.0
    op = NonlinearOperator2D(ctx)
    at_threshold = np.asarray(op.jacobian_vector(head, vector).numpy()).copy()
    # The central cell's transmissivity derivative is zero at the threshold;
    # a perturbation strictly inside activates it and changes the action.
    inside = head.copy()
    inside[3, 3] += 1.0e-5 if relative_head == 0.1 else -1.0e-5
    inside_action = np.asarray(op.jacobian_vector(inside, vector).numpy()).copy()
    assert np.max(np.abs(at_threshold - inside_action)) > 1.0e-8
    assert np.all(np.isfinite(at_threshold))
    op.close()


def _build_solver(*, use_ghb: bool = False, device: str = "cpu"):
    from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver

    ny, nx = 8, 10
    y, x = np.mgrid[:ny, :nx]
    active = np.ones((ny, nx), dtype=np.int32)
    active[3, 4] = 0
    prescribed = np.zeros((ny, nx), dtype=np.int32)
    prescribed[:, 0] = 1
    prescribed[:, -1] = 1
    values = np.zeros((ny, nx))
    values[:, 0] = 12.0
    values[:, -1] = 9.0
    build = dict(
        T_field=np.full((ny, nx), 10.0),
        R_field=np.full((ny, nx), 1.0e-5),
        active=active,
        bc_mask=prescribed,
        bc_values=values,
    )
    if use_ghb:
        gh_mask = np.zeros((ny, nx), dtype=np.int32)
        gh_mask[1:-1, 2] = 1
        gh_head = np.full((ny, nx), 10.5)
        gh_width = np.zeros((ny, nx))
        gh_width[gh_mask != 0] = 20.0
        build.update(gh_mask=gh_mask, gh_head=gh_head, gh_width=gh_width)
    solver = WarpDarcySolver(
        nx=nx,
        ny=ny,
        dx=50.0,
        device=device,
        use_ghb=use_ghb,
        solver_type="kcycle",
        diag_preconditioner_backend="device",
    )
    solver.build_from_fields(**build)
    K = (0.5 + np.exp(np.linspace(-1.0, 1.0, nx))[None, :] * np.ones((ny, 1))).astype(np.float64)
    bottom = (0.05 * x + 0.02 * y).astype(np.float64)
    top = bottom + 20.0
    return solver, K, bottom, top


def _newton_controls() -> dict:
    return {
        "max_levels": 2,
        "min_coarse_cells": 1,
        "nu_pre": 1,
        "nu_post": 1,
        "nu_coarse": 20,
        "newton_fgmres_restart": 12,
        "newton_fgmres_max_iterations": 60,
        "newton_max_iterations": 12,
        "newton_residual_rms_tolerance": 1.0e-6,
        "newton_head_equivalent_rms_tolerance": 1.0e-6,
        "newton_max_head_change": 20.0,
        "newton_fallback_to_picard": False,
    }


@pytest.mark.parametrize("use_ghb", [False, True])
def test_newton_solves_steady_heterogeneous_sloping_masked_problem(use_ghb):
    solver, K, bottom, top = _build_solver(use_ghb=use_ghb)
    try:
        initial = bottom + 8.0
        head, info = solver.solve(
            formulation="unconfined",
            solver="unconfined_semismooth_newton_kcycle",
            initial_head=initial,
            K_field=K,
            zbot_field=bottom,
            ztop_field=top,
            **_newton_controls(),
        )
        assert info["converged"], info
        assert info["newton_fallback_used"] is False
        assert info["true_nonlinear_residual_rms"] <= 1.0e-6
        assert np.all(np.isfinite(head))
        assert info["kcycle_preconditioner_applications"] == info["fgmres_iterations"]
    finally:
        solver.close()


def test_newton_solves_high_contrast_conductivity_with_shared_kcycle():
    solver, K, bottom, top = _build_solver()
    K[:, ::2] *= 0.2
    K[:, 1::2] *= 5.0
    assert float(np.max(K) / np.min(K)) > 90.0
    try:
        _, info = solver.solve(
            formulation="unconfined",
            solver="unconfined_semismooth_newton_kcycle",
            K_field=K,
            zbot_field=bottom,
            ztop_field=top,
            **dict(
                _newton_controls(),
                newton_fgmres_restart=20,
                newton_fgmres_max_iterations=300,
                newton_preconditioner_kcycles=2,
            ),
        )
        assert info["converged"], info
        assert info["newton_fallback_used"] is False
        assert info["true_nonlinear_residual_rms"] <= 1.0e-6
    finally:
        solver.close()


def test_newton_exact_combined_transient_storage_and_timestep_change():
    solver, K, bottom, top = _build_solver()
    try:
        previous = bottom + 8.0
        controls = _newton_controls()
        for dt in (2.0, 0.75):
            head, info = solver.solve(
                formulation="unconfined",
                solver="unconfined_semismooth_newton_kcycle",
                initial_head=previous,
                K_field=K,
                zbot_field=bottom,
                ztop_field=top,
                transient=True,
                storage_coeff=0.2,
                sy=0.2,
                ss=1.0e-4,
                dt=dt,
                head_prev=previous,
                **controls,
            )
            assert info["converged"], info
            assert info["storage_sy_array"] is not None
            assert info["storage_ss_array"] is not None
            assert "storage_in" in info["budget_summary"]
            previous = head
    finally:
        solver.close()


def test_transient_newton_matches_picard_head_and_exact_storage_state():
    from DARCY_WARP_PACKAGE.nonlinear import NonlinearOperator2D, from_unconfined_solve_inputs

    solver, K, bottom, top = _build_solver()
    previous = bottom + 8.0
    try:
        newton, newton_info = solver.solve(
            formulation="unconfined",
            solver="unconfined_semismooth_newton_kcycle",
            initial_head=previous,
            K_field=K,
            zbot_field=bottom,
            ztop_field=top,
            transient=True,
            storage_coeff=0.2,
            sy=0.2,
            ss=1.0e-5,
            dt=2.0,
            head_prev=previous,
            **_newton_controls(),
        )
        picard, picard_info = solver.solve(
            formulation="unconfined",
            solver="unconfined_picard_kcycle",
            initial_head=previous,
            K_field=K,
            zbot_field=bottom,
            ztop_field=top,
            transient=True,
            storage_coeff=0.2,
            sy=0.2,
            ss=1.0e-5,
            dt=2.0,
            head_prev=previous,
            storage_reference="current_picard",
            unconfined_storage_mode_2d="mf6_convertible_secant_sy",
            max_levels=2,
            min_coarse_cells=1,
            max_outer_iterations=60,
            max_cycles=20,
            hclose=1.0e-6,
            abs_tol_min=1.0e-8,
            rel_tol=1.0e-8,
            save_transient_diagnostics=True,
        )
        assert newton_info["converged"] and picard_info["converged"]
        np.testing.assert_allclose(newton, picard, atol=2.0e-5, rtol=0.0)
        ctx = from_unconfined_solve_inputs(
            solver,
            K_field=K,
            zbot_field=bottom,
            ztop_field=top,
            transient=True,
            sy=0.2,
            ss=1.0e-5,
            dt=2.0,
            head_prev=previous,
            min_sat=0.1,
        )
        op = NonlinearOperator2D(ctx)
        picard_storage = op.exact_storage_terms(picard)
        np.testing.assert_allclose(
            newton_info["storage_total_array"],
            picard_storage.total,
            atol=5.0e-3,
            rtol=1.0e-5,
        )
        op.close()
    finally:
        solver.close()


def test_newton_matches_picard_head_saturation_and_steady_budget():
    solver, K, bottom, top = _build_solver()
    try:
        newton, newton_info = solver.solve(
            formulation="unconfined",
            solver="unconfined_semismooth_newton_kcycle",
            K_field=K,
            zbot_field=bottom,
            ztop_field=top,
            **_newton_controls(),
        )
        picard, picard_info = solver.solve(
            formulation="unconfined",
            solver="unconfined_picard_kcycle",
            K_field=K,
            zbot_field=bottom,
            ztop_field=top,
            max_levels=2,
            min_coarse_cells=1,
            max_outer_iterations=60,
            max_cycles=20,
            hclose=1.0e-6,
            abs_tol_min=1.0e-8,
            rel_tol=1.0e-8,
        )
        assert newton_info["converged"] and picard_info["converged"]
        np.testing.assert_allclose(newton, picard, atol=5.0e-6, rtol=0.0)
        picard_saturation = np.clip(picard - bottom, 0.1, np.maximum(top - bottom, 0.1))
        np.testing.assert_allclose(
            newton_info["saturated_thickness_array"],
            picard_saturation,
            atol=5.0e-6,
            rtol=0.0,
        )
        assert abs(float(newton_info["budget_summary"]["percent_discrepancy"])) < 1.0e-4
    finally:
        solver.close()


@pytest.mark.parametrize("initial_offset", [0.11, 30.0])
def test_newton_near_bottom_and_poor_initial_estimates(initial_offset):
    solver, K, bottom, top = _build_solver()
    try:
        head, info = solver.solve(
            formulation="unconfined",
            solver="unconfined_semismooth_newton_kcycle",
            initial_head=bottom + initial_offset,
            K_field=K,
            zbot_field=bottom,
            ztop_field=top,
            **dict(
                _newton_controls(),
                newton_max_head_change=50.0,
                newton_fallback_to_picard=(initial_offset < 1.0),
                max_cycles=20,
                max_outer_iterations=60,
            ),
        )
        assert info["converged"], info
        assert np.all(np.isfinite(head))
        assert info["newton_fallback_used"] is (initial_offset < 1.0)
    finally:
        solver.close()


def test_confined_linear_limit_converges_in_one_newton_correction():
    solver, K, bottom, _ = _build_solver()
    top = bottom + 2.0
    initial = bottom + 10.0
    try:
        _, info = solver.solve(
            formulation="unconfined",
            solver="unconfined_semismooth_newton_kcycle",
            initial_head=initial,
            K_field=K,
            zbot_field=bottom,
            ztop_field=top,
            **dict(
                _newton_controls(),
                newton_fgmres_relative_tolerance=1.0e-10,
                newton_fgmres_absolute_tolerance=1.0e-12,
            ),
        )
        assert info["converged"], info
        assert info["newton_iterations"] <= 1
    finally:
        solver.close()


def test_signed_recharge_and_aggregated_withdrawal_sources():
    solver, K, bottom, top = _build_solver()
    sources = np.full((solver.ny, solver.nx), 2.0e-5)
    sources[2:6, 4:7] = -8.0e-5
    solver.update_R_in_place(sources)
    try:
        head, info = solver.solve(
            formulation="unconfined",
            solver="unconfined_semismooth_newton_kcycle",
            initial_head=bottom + 8.0,
            K_field=K,
            zbot_field=bottom,
            ztop_field=top,
            **_newton_controls(),
        )
        assert info["converged"], info
        assert info["budget_summary"]["rcha_in"] > 0.0
        assert info["budget_summary"]["rcha_out"] > 0.0
        assert np.all(np.isfinite(head))
    finally:
        solver.close()


def test_failure_falls_back_explicitly_and_backend_switching_is_stable():
    solver, K, bottom, top = _build_solver()
    try:
        picard_before, _ = solver.solve(
            formulation="unconfined",
            solver="unconfined_picard_kcycle",
            K_field=K,
            zbot_field=bottom,
            ztop_field=top,
            max_levels=2,
            min_coarse_cells=1,
            max_outer_iterations=30,
            max_cycles=10,
            hclose=1.0e-5,
        )
        head, info = solver.solve(
            formulation="unconfined",
            solver="unconfined_semismooth_newton_kcycle",
            initial_head=bottom + 2.0,
            K_field=K,
            zbot_field=bottom,
            ztop_field=top,
            newton_fgmres_restart=4,
            newton_fgmres_max_iterations=1,
            newton_max_iterations=1,
            newton_fallback_to_picard=True,
            max_levels=2,
            min_coarse_cells=1,
            max_outer_iterations=40,
            max_cycles=10,
            hclose=1.0e-5,
        )
        assert info["newton_fallback_used"] is True
        assert info["fallback_backend"] == "unconfined_picard_kcycle"
        picard_after, _ = solver.solve(
            formulation="unconfined",
            solver="unconfined_picard_kcycle",
            K_field=K,
            zbot_field=bottom,
            ztop_field=top,
            max_levels=2,
            min_coarse_cells=1,
            max_outer_iterations=30,
            max_cycles=10,
            hclose=1.0e-5,
        )
        np.testing.assert_allclose(picard_after, picard_before, atol=1.0e-10, rtol=0.0)
        assert np.all(np.isfinite(head))
    finally:
        solver.close()


def test_repeated_newton_runs_are_deterministic_and_reuse_workspaces():
    solver, K, bottom, top = _build_solver()
    try:
        controls = _newton_controls()
        first, first_info = solver.solve(
            formulation="unconfined",
            solver="unconfined_semismooth_newton_kcycle",
            K_field=K,
            zbot_field=bottom,
            ztop_field=top,
            **controls,
        )
        ids = {
            key: id(value)
            for key, value in solver._resource_owner.experimental_workspaces.items()
        }
        second, second_info = solver.solve(
            formulation="unconfined",
            solver="unconfined_semismooth_newton_kcycle",
            K_field=K,
            zbot_field=bottom,
            ztop_field=top,
            **controls,
        )
        np.testing.assert_allclose(second, first, atol=0.0, rtol=0.0)
        assert second_info["fgmres_iterations"] == first_info["fgmres_iterations"]
        assert ids == {
            key: id(value)
            for key, value in solver._resource_owner.experimental_workspaces.items()
        }
    finally:
        solver.close()


def test_repeated_cuda_newton_memory_is_stable_after_warmup():
    import warp as wp

    if not wp.is_cuda_available():
        pytest.skip("CUDA is not available")
    solver, K, bottom, top = _build_solver(device="cuda:0")
    try:
        controls = _newton_controls()
        solver.solve(
            formulation="unconfined",
            solver="unconfined_semismooth_newton_kcycle",
            K_field=K,
            zbot_field=bottom,
            ztop_field=top,
            **controls,
        )
        wp.synchronize_device("cuda:0")
        used = wp.get_mempool_used_mem_current("cuda:0")
        solver.solve(
            formulation="unconfined",
            solver="unconfined_semismooth_newton_kcycle",
            K_field=K,
            zbot_field=bottom,
            ztop_field=top,
            **controls,
        )
        wp.synchronize_device("cuda:0")
        assert wp.get_mempool_used_mem_current("cuda:0") <= used
    finally:
        solver.close()
        gc.collect()

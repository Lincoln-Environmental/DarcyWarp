"""Stage-3 regression tests for the experimental two-dimensional FAS backend."""

from __future__ import annotations

import gc
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("WARP_CACHE_PATH", str(Path("/tmp/darcywarp-warp-cache")))


def _physical_level(
    *,
    ny: int = 5,
    nx: int = 7,
    prescribed: bool = False,
    ghb: bool = False,
):
    from DARCY_WARP_PACKAGE.solvers.fas_hierarchy import make_fine_physical_level

    y, x = np.mgrid[:ny, :nx]
    active = np.ones((ny, nx), dtype=np.int32)
    active[-1, -1] = 0
    active[2, 3] = 0
    bc_mask = np.zeros((ny, nx), dtype=np.int32)
    bc_values = np.zeros((ny, nx), dtype=np.float64)
    if prescribed:
        bc_mask[:, 0] = 1
        bc_values[:, 0] = 12.0 + 0.1 * y[:, 0]
    ghb_mask = np.zeros((ny, nx), dtype=np.int32)
    ghb_factor = np.zeros((ny, nx), dtype=np.float64)
    ghb_head = np.zeros((ny, nx), dtype=np.float64)
    if ghb:
        ghb_mask[1:4, 2:4] = 1
        ghb_factor[ghb_mask != 0] = np.linspace(0.01, 0.06, np.count_nonzero(ghb_mask))
        ghb_head[ghb_mask != 0] = np.linspace(8.0, 11.0, np.count_nonzero(ghb_mask))
    bottom = 0.05 * x + 0.1 * y
    top = bottom + 10.0 + 0.2 * np.sin(x)
    return make_fine_physical_level(
        conductivity=0.5 + np.exp(0.25 * x),
        top=top,
        bottom=bottom,
        active=active,
        dirichlet_mask=bc_mask,
        dirichlet_values=bc_values,
        source_rate=1.0e-5 * (1.0 + x - 0.25 * y),
        ghb_mask=ghb_mask,
        ghb_factor=ghb_factor,
        ghb_external_head=ghb_head,
        sy=0.12 + 0.01 * y,
        ss=1.0e-5 * (1.0 + x),
        previous_head=bottom + 5.0 + 0.1 * y,
        dx=25.0,
    )


def _build_solver(*, use_ghb: bool = False, device: str = "cpu"):
    from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver

    ny, nx = 8, 10
    y, x = np.mgrid[:ny, :nx]
    active = np.ones((ny, nx), dtype=np.int32)
    active[3, 4] = 0
    active[4, 4] = 0
    prescribed = np.zeros((ny, nx), dtype=np.int32)
    prescribed[:, 0] = 1
    prescribed[:, -1] = 1
    values = np.zeros((ny, nx), dtype=np.float64)
    values[:, 0] = 12.0
    values[:, -1] = 9.0
    build = {
        "T_field": np.full((ny, nx), 10.0),
        "R_field": np.full((ny, nx), 1.0e-5),
        "active": active,
        "bc_mask": prescribed,
        "bc_values": values,
    }
    if use_ghb:
        gh_mask = np.zeros((ny, nx), dtype=np.int32)
        gh_mask[1:-1, 2] = 1
        gh_head = np.full((ny, nx), 10.5)
        gh_width = np.zeros((ny, nx), dtype=np.float64)
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
    conductivity = 0.5 + np.exp(np.linspace(-1.0, 1.0, nx))[None, :] * np.ones((ny, 1))
    bottom = 0.05 * x + 0.02 * y
    top = bottom + 20.0
    return solver, conductivity.astype(np.float64), bottom.astype(np.float64), top.astype(np.float64)


def _fas_controls() -> dict:
    return {
        "fas_max_levels": 3,
        "fas_min_coarse_cells": 1,
        "fas_pre_smoothing_sweeps": 3,
        "fas_post_smoothing_sweeps": 3,
        "fas_coarse_smoothing_sweeps": 40,
        "fas_damping": 0.65,
        "fas_max_cycles": 30,
        "fas_residual_rms_tolerance": 1.0e-6,
        "fas_head_equivalent_rms_tolerance": 1.0e-6,
        "fas_fallback_enabled": False,
    }


def _newton_controls() -> dict:
    return {
        "max_levels": 2,
        "min_coarse_cells": 1,
        "nu_pre": 1,
        "nu_post": 1,
        "nu_coarse": 20,
        "newton_fgmres_restart": 12,
        "newton_fgmres_max_iterations": 80,
        "newton_max_iterations": 12,
        "newton_residual_rms_tolerance": 1.0e-6,
        "newton_head_equivalent_rms_tolerance": 1.0e-6,
        "newton_fallback_to_picard": False,
    }


def _picard_controls() -> dict:
    return {
        "max_levels": 2,
        "min_coarse_cells": 1,
        "max_outer_iterations": 60,
        "max_cycles": 20,
        "hclose": 1.0e-6,
        "abs_tol_min": 1.0e-8,
        "rel_tol": 1.0e-8,
    }


def test_coarse_rediscretization_preserves_source_volume_on_odd_grid():
    from DARCY_WARP_PACKAGE.solvers.fas_hierarchy import coarsen_physical_level

    fine = _physical_level()
    coarse = coarsen_physical_level(fine, min_sat=0.1)
    assert coarse.shape == (3, 4)
    fine_free = (fine.active != 0) & (fine.dirichlet_mask == 0)
    coarse_free = (coarse.active != 0) & (coarse.dirichlet_mask == 0)
    fine_volume = float(np.sum(fine.source_rate[fine_free]) * fine.area)
    coarse_volume = float(np.sum(coarse.source_rate[coarse_free]) * coarse.area)
    assert coarse_volume == pytest.approx(fine_volume, abs=1.0e-12)
    assert coarse.active_fraction[-1, -1] == 0.0
    assert np.all(coarse.top[coarse.active != 0] > coarse.bottom[coarse.active != 0])


def test_coarse_dirichlet_previous_state_and_intensive_properties_are_mask_aware():
    from DARCY_WARP_PACKAGE.solvers.fas_hierarchy import coarsen_physical_level

    fine = _physical_level(prescribed=True)
    coarse = coarsen_physical_level(fine, min_sat=0.1)
    assert np.all(coarse.dirichlet_mask[:, 0] == 1)
    for cj in range(coarse.ny):
        rows = slice(2 * cj, min(2 * cj + 2, fine.ny))
        expected = np.mean(fine.dirichlet_values[rows, 0])
        assert coarse.dirichlet_values[cj, 0] == pytest.approx(expected)
    assert np.all(np.isfinite(coarse.previous_head[coarse.active != 0]))
    assert np.all(coarse.sy[coarse.active != 0] >= 0.0)
    assert np.all(coarse.ss[coarse.active != 0] >= 0.0)


def test_coarse_ghb_reference_conductance_and_stage_are_preserved():
    from DARCY_WARP_PACKAGE.solvers.fas_hierarchy import coarsen_physical_level

    fine = _physical_level(ghb=True)
    coarse = coarsen_physical_level(fine, min_sat=0.1)
    for cj in range(coarse.ny):
        for ci in range(coarse.nx):
            js = slice(2 * cj, min(2 * cj + 2, fine.ny))
            is_ = slice(2 * ci, min(2 * ci + 2, fine.nx))
            mask = (fine.active[js, is_] != 0) & (fine.ghb_mask[js, is_] != 0)
            if not np.any(mask):
                continue
            fine_conductance = (
                fine.conductivity[js, is_]
                * np.maximum(fine.top[js, is_] - fine.bottom[js, is_], 0.1)
                * fine.ghb_factor[js, is_]
            )
            expected = float(np.sum(fine_conductance[mask]))
            actual = float(
                coarse.conductivity[cj, ci]
                * max(coarse.top[cj, ci] - coarse.bottom[cj, ci], 0.1)
                * coarse.ghb_factor[cj, ci]
            )
            assert actual == pytest.approx(expected, rel=1.0e-12, abs=1.0e-12)
            expected_stage = float(np.sum(fine_conductance[mask] * fine.ghb_external_head[js, is_][mask]) / expected)
            assert coarse.ghb_external_head[cj, ci] == pytest.approx(expected_stage)


def test_tau_identity_and_restricted_constant_solution_invariance():
    import warp as wp
    from DARCY_WARP_PACKAGE.solvers.fas import _restrict_head, _restrict_integrated
    from DARCY_WARP_PACKAGE.solvers.fas_hierarchy import build_fas_physical_hierarchy, make_fine_physical_level
    from DARCY_WARP_PACKAGE.solvers.fas_kernels import fas_build_coarse_forcing_kernel
    from DARCY_WARP_PACKAGE.solvers.fas_state import FASWorkspace

    ny, nx = 6, 8
    zeros = np.zeros((ny, nx), dtype=np.float64)
    active = np.ones((ny, nx), dtype=np.int32)
    physical = make_fine_physical_level(
        conductivity=np.ones((ny, nx)), top=np.full((ny, nx), 10.0), bottom=zeros,
        active=active, dirichlet_mask=np.zeros((ny, nx), dtype=np.int32),
        dirichlet_values=zeros, source_rate=zeros, ghb_mask=np.zeros_like(active),
        ghb_factor=zeros, ghb_external_head=zeros, sy=0.0, ss=0.0,
        previous_head=zeros, dx=10.0,
    )
    levels = build_fas_physical_hierarchy(physical, max_levels=2, min_coarse_cells=1, min_sat=0.1)
    workspace = FASWorkspace(physical_levels=levels, transient=False, dt=None, min_sat=0.1, device="cpu")
    try:
        fine, coarse = workspace.levels
        fine.nonlinear_operator.operator.set_head(np.full((ny, nx), 5.0))
        wp.copy(fine.head, fine.nonlinear_operator.operator.head_device)
        wp.copy(fine.forcing, fine.physical_forcing)
        fine.nonlinear_operator.evaluate(head=fine.head, state=fine)
        _restrict_head(fine=fine, coarse=coarse, workspace=workspace)
        _restrict_integrated(fine=fine, coarse=coarse, fine_value=fine.defect, coarse_value=coarse.restricted_defect, workspace=workspace)
        _restrict_integrated(fine=fine, coarse=coarse, fine_value=fine.forcing, coarse_value=coarse.restricted_forcing, workspace=workspace)
        coarse.nonlinear_operator.evaluate(head=coarse.head_initial, state=coarse)
        wp.launch(
            kernel=fas_build_coarse_forcing_kernel,
            dim=coarse.shape,
            inputs=[
                coarse.physical_residual, coarse.physical_forcing,
                coarse.restricted_defect, coarse.restricted_forcing,
                coarse.nonlinear_operator.operator.active_device,
                coarse.nonlinear_operator.operator.dirichlet_mask_device,
                coarse.forcing, coarse.tau, coarse.physical.nx, coarse.physical.ny,
            ],
            device="cpu",
        )
        coarse_n = coarse.physical_residual.numpy() + coarse.physical_forcing.numpy()
        np.testing.assert_allclose(coarse.forcing.numpy() - coarse_n, coarse.restricted_defect.numpy(), atol=1.0e-13)
        np.testing.assert_allclose(coarse.tau.numpy(), 0.0, atol=1.0e-13)
        np.testing.assert_allclose(coarse.head_initial.numpy(), 5.0, atol=0.0)
        np.testing.assert_allclose(coarse.defect.numpy(), 0.0, atol=1.0e-13)
    finally:
        workspace.close()


@pytest.mark.parametrize("use_ghb", [False, True])
def test_fas_solves_steady_heterogeneous_sloping_irregular_problem(use_ghb):
    solver, conductivity, bottom, top = _build_solver(use_ghb=use_ghb)
    try:
        head, info = solver.solve(
            formulation="unconfined", solver="unconfined_fas",
            initial_head=bottom + 8.0, K_field=conductivity,
            zbot_field=bottom, ztop_field=top, **_fas_controls(),
        )
        assert info["converged"], info
        assert info["fas_fallback_used"] is False
        assert info["true_nonlinear_residual_rms"] <= 1.0e-6
        assert info["n_levels"] == 3
        assert info["tau_formulation"] == "N_coarse(R_head)-R(N_fine(head))"
        assert info["smoothing_factors_by_level"]
        assert all(np.isfinite(item["factor"]) for item in info["smoothing_factors_by_level"])
        assert np.all(np.isfinite(head))
        assert np.all(head[info["active_mask"] & ~solver.bc_mask_host.astype(bool)] != 0.0)
    finally:
        solver.close()


def test_fas_high_contrast_near_bottom_and_signed_sources():
    solver, conductivity, bottom, top = _build_solver()
    conductivity[:, ::2] *= 0.2
    conductivity[:, 1::2] *= 5.0
    sources = np.full((solver.ny, solver.nx), 2.0e-5)
    sources[2:6, 4:7] = -8.0e-5
    solver.update_R_in_place(sources)
    try:
        head, info = solver.solve(
            formulation="unconfined", solver="unconfined_fas",
            initial_head=bottom + 0.11, K_field=conductivity,
            zbot_field=bottom, ztop_field=top,
            **dict(_fas_controls(), fas_max_cycles=90, fas_correction_head_change_limit=30.0),
        )
        assert info["converged"], info
        assert info["budget_summary"]["rcha_in"] > 0.0
        assert info["budget_summary"]["rcha_out"] > 0.0
        assert np.any(info["dry_mask"])
        assert np.all(np.isfinite(head))
    finally:
        solver.close()


def test_fas_preserves_a_narrow_active_connection_and_isolated_side_cell():
    from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver

    ny, nx = 7, 9
    active = np.zeros((ny, nx), dtype=np.int32)
    active[3, :] = 1
    active[2, 4] = 1
    prescribed = np.zeros_like(active)
    prescribed[3, 0] = 1
    prescribed[3, -1] = 1
    values = np.zeros((ny, nx), dtype=np.float64)
    values[3, 0] = 10.0
    values[3, -1] = 8.0
    solver = WarpDarcySolver(nx=nx, ny=ny, dx=10.0, device="cpu", solver_type="kcycle")
    solver.build_from_fields(
        T_field=np.full((ny, nx), 5.0), R_field=np.zeros((ny, nx)),
        active=active, bc_mask=prescribed, bc_values=values,
    )
    bottom = np.zeros((ny, nx))
    top = np.full((ny, nx), 10.0)
    try:
        head, info = solver.solve(
            formulation="unconfined", solver="unconfined_fas",
            initial_head=np.where(active != 0, 9.0, 0.0),
            K_field=np.ones((ny, nx)), zbot_field=bottom, ztop_field=top,
            **dict(_fas_controls(), fas_max_cycles=40),
        )
        assert info["converged"], info
        assert np.all(np.diff(head[3, :]) < 0.0)
        assert np.isfinite(head[2, 4])
        assert np.all(head[active == 0] == 0.0)
    finally:
        solver.close()


def test_fas_linear_capped_limit_matches_picard_and_newton():
    solver, conductivity, bottom, _ = _build_solver()
    top = bottom + 2.0
    initial = bottom + 10.0
    try:
        fas, fas_info = solver.solve(
            formulation="unconfined", solver="unconfined_fas", initial_head=initial,
            K_field=conductivity, zbot_field=bottom, ztop_field=top,
            **dict(
                _fas_controls(),
                fas_residual_rms_tolerance=1.0e-8,
                fas_head_equivalent_rms_tolerance=1.0e-8,
            ),
        )
        newton, newton_info = solver.solve(
            formulation="unconfined", solver="unconfined_semismooth_newton_kcycle",
            initial_head=initial, K_field=conductivity, zbot_field=bottom,
            ztop_field=top, **_newton_controls(),
        )
        picard, picard_info = solver.solve(
            formulation="unconfined", solver="unconfined_picard_kcycle",
            initial_head=initial, K_field=conductivity, zbot_field=bottom,
            ztop_field=top, **_picard_controls(),
        )
        assert fas_info["converged"] and newton_info["converged"] and picard_info["converged"]
        np.testing.assert_allclose(fas, picard, atol=2.0e-6, rtol=0.0)
        np.testing.assert_allclose(fas, newton, atol=2.0e-6, rtol=0.0)
    finally:
        solver.close()


@pytest.mark.parametrize("sy,ss", [(0.2, 0.0), (0.0, 1.0e-4), (0.2, 1.0e-4)])
def test_fas_exact_transient_storage_and_previous_head_coarsening(sy, ss):
    solver, conductivity, bottom, top = _build_solver()
    previous = bottom + 8.0
    try:
        head, info = solver.solve(
            formulation="unconfined", solver="unconfined_fas",
            initial_head=previous, K_field=conductivity, zbot_field=bottom,
            ztop_field=top, transient=True, storage_coeff=sy, sy=sy, ss=ss,
            dt=2.0, head_prev=previous, **_fas_controls(),
        )
        assert info["converged"], info
        assert info["storage_sy_array"] is not None
        assert info["storage_ss_array"] is not None
        assert "storage_in" in info["budget_summary"]
        workspace = solver._resource_owner.get_experimental_workspace("unconfined_fas")
        expected = np.mean(previous[:2, :2])
        assert workspace.levels[1].physical.previous_head[0, 0] == pytest.approx(expected)
        assert np.all(np.isfinite(head))
    finally:
        solver.close()


def test_fas_transient_timestep_changes_and_bottom_top_storage_crossings():
    from DARCY_WARP_PACKAGE.nonlinear import NonlinearOperator2D, from_unconfined_solve_inputs

    solver, conductivity, bottom, top = _build_solver()
    previous_states = (bottom - 0.01, top + 0.01)
    initial_states = (bottom + 0.2, top - 0.2)
    timesteps = (100.0, 3.0)
    try:
        for previous, initial, dt in zip(previous_states, initial_states, timesteps):
            head, info = solver.solve(
                formulation="unconfined", solver="unconfined_fas",
                initial_head=initial, K_field=conductivity,
                zbot_field=bottom, ztop_field=top, transient=True,
                storage_coeff=0.2, sy=0.2, ss=1.0e-4, dt=dt,
                head_prev=previous, **_fas_controls(),
            )
            assert info["converged"], info
            context = from_unconfined_solve_inputs(
                solver, K_field=conductivity, zbot_field=bottom,
                ztop_field=top, transient=True, sy=0.2, ss=1.0e-4,
                dt=dt, head_prev=previous, min_sat=0.1,
            )
            operator = NonlinearOperator2D(context)
            try:
                exact = operator.exact_storage_terms(head)
                np.testing.assert_allclose(info["storage_sy_array"], exact.sy, atol=0.0, rtol=0.0)
                np.testing.assert_allclose(info["storage_ss_array"], exact.ss, atol=0.0, rtol=0.0)
            finally:
                operator.close()
    finally:
        solver.close()


def test_transient_fas_matches_authoritative_picard_storage_and_head():
    solver, conductivity, bottom, top = _build_solver()
    previous = bottom + 8.0
    try:
        fas, fas_info = solver.solve(
            formulation="unconfined", solver="unconfined_fas",
            initial_head=previous, K_field=conductivity, zbot_field=bottom,
            ztop_field=top, transient=True, storage_coeff=0.2, sy=0.2,
            ss=1.0e-5, dt=2.0, head_prev=previous, **_fas_controls(),
        )
        picard, picard_info = solver.solve(
            formulation="unconfined", solver="unconfined_picard_kcycle",
            initial_head=previous, K_field=conductivity, zbot_field=bottom,
            ztop_field=top, transient=True, storage_coeff=0.2, sy=0.2,
            ss=1.0e-5, dt=2.0, head_prev=previous,
            storage_reference="current_picard",
            unconfined_storage_mode_2d="mf6_convertible_secant_sy",
            save_transient_diagnostics=True, **_picard_controls(),
        )
        assert fas_info["converged"] and picard_info["converged"]
        np.testing.assert_allclose(fas, picard, atol=2.0e-5, rtol=0.0)
        from DARCY_WARP_PACKAGE.nonlinear import NonlinearOperator2D, from_unconfined_solve_inputs
        context = from_unconfined_solve_inputs(
            solver, K_field=conductivity, zbot_field=bottom, ztop_field=top,
            transient=True, sy=0.2, ss=1.0e-5, dt=2.0,
            head_prev=previous, min_sat=0.1,
        )
        operator = NonlinearOperator2D(context)
        try:
            reference_storage = operator.exact_storage_terms(picard)
            np.testing.assert_allclose(fas_info["storage_total_array"], reference_storage.total, atol=5.0e-3, rtol=1.0e-5)
        finally:
            operator.close()
    finally:
        solver.close()


def test_fas_failure_fallback_order_and_backend_switching_restore_state():
    solver, conductivity, bottom, top = _build_solver()
    try:
        picard_before, _ = solver.solve(
            formulation="unconfined", solver="unconfined_picard_kcycle",
            K_field=conductivity, zbot_field=bottom, ztop_field=top,
            **_picard_controls(),
        )
        newton, newton_info = solver.solve(
            formulation="unconfined", solver="unconfined_semismooth_newton_kcycle",
            K_field=conductivity, zbot_field=bottom, ztop_field=top,
            **_newton_controls(),
        )
        fas, fas_info = solver.solve(
            formulation="unconfined", solver="unconfined_fas",
            initial_head=bottom + 8.0, K_field=conductivity,
            zbot_field=bottom, ztop_field=top, fas_max_cycles=1,
            fas_residual_rms_tolerance=1.0e-14,
            fas_head_equivalent_rms_tolerance=1.0e-14,
            fas_fallback_enabled=True,
            fas_fallback_order=("unconfined_semismooth_newton_kcycle", "unconfined_picard_kcycle"),
            **_newton_controls(),
        )
        assert newton_info["converged"] and fas_info["converged"]
        assert fas_info["fas_fallback_used"] is True
        assert fas_info["fallback_backend"] == "unconfined_semismooth_newton_kcycle"
        picard_after, _ = solver.solve(
            formulation="unconfined", solver="unconfined_picard_kcycle",
            K_field=conductivity, zbot_field=bottom, ztop_field=top,
            **_picard_controls(),
        )
        np.testing.assert_allclose(picard_after, picard_before, atol=1.0e-10, rtol=0.0)
        np.testing.assert_allclose(fas, newton, atol=2.0e-6, rtol=0.0)
    finally:
        solver.close()


def test_repeated_fas_runs_are_deterministic_and_reuse_clean_workspace():
    solver, conductivity, bottom, top = _build_solver()
    try:
        controls = _fas_controls()
        first, first_info = solver.solve(
            formulation="unconfined", solver="unconfined_fas",
            K_field=conductivity, zbot_field=bottom, ztop_field=top, **controls,
        )
        workspace = solver._resource_owner.get_experimental_workspace("unconfined_fas")
        workspace_id = id(workspace)
        first_tau = [item["tau_norms"] for item in first_info["fas_cycle_history"]]
        second, second_info = solver.solve(
            formulation="unconfined", solver="unconfined_fas",
            K_field=conductivity, zbot_field=bottom, ztop_field=top, **controls,
        )
        np.testing.assert_allclose(second, first, atol=0.0, rtol=0.0)
        assert second_info["fas_cycles"] == first_info["fas_cycles"]
        assert [item["tau_norms"] for item in second_info["fas_cycle_history"]] == first_tau
        assert id(solver._resource_owner.get_experimental_workspace("unconfined_fas")) == workspace_id
    finally:
        solver.close()


def test_repeated_cuda_fas_memory_is_stable_after_warmup():
    import warp as wp

    if not wp.is_cuda_available():
        pytest.skip("CUDA is not available")
    solver, conductivity, bottom, top = _build_solver(device="cuda:0")
    try:
        controls = _fas_controls()
        solver.solve(
            formulation="unconfined", solver="unconfined_fas",
            K_field=conductivity, zbot_field=bottom, ztop_field=top, **controls,
        )
        wp.synchronize_device("cuda:0")
        used = wp.get_mempool_used_mem_current("cuda:0")
        solver.solve(
            formulation="unconfined", solver="unconfined_fas",
            K_field=conductivity, zbot_field=bottom, ztop_field=top, **controls,
        )
        wp.synchronize_device("cuda:0")
        assert wp.get_mempool_used_mem_current("cuda:0") <= used
    finally:
        solver.close()
        gc.collect()

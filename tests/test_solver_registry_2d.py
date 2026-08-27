"""Regression coverage for the explicit 2D solver-backend architecture."""

from __future__ import annotations

import gc
from types import SimpleNamespace

import numpy as np
import pytest


def _warp_available() -> bool:
    try:
        import warp  # noqa: F401
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(not _warp_available(), reason="warp is not available")


def _build_solver(*, use_ghb: bool = False, device: str = "cpu"):
    from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver

    nx, ny = 8, 6
    solver = WarpDarcySolver(
        nx=nx,
        ny=ny,
        dx=10.0,
        device=device,
        use_ghb=use_ghb,
        solver_type="kcycle",
        diag_preconditioner_backend="host",
    )
    y, x = np.mgrid[:ny, :nx]
    active = np.ones((ny, nx), dtype=np.int32)
    active[2, 3] = 0  # irregular active mask
    bc_mask = np.zeros((ny, nx), dtype=np.int32)
    bc_mask[:, 0] = 1
    bc_mask[:, -1] = 1
    bc_values = np.zeros((ny, nx), dtype=np.float64)
    bc_values[:, 0] = 12.0
    bc_values[:, -1] = 9.0
    transmissivity = (5.0 + 0.25 * x + 0.5 * y).astype(np.float64)
    recharge = np.full((ny, nx), 1.0e-4, dtype=np.float64)
    build_kwargs = dict(
        T_field=transmissivity,
        R_field=recharge,
        active=active,
        bc_mask=bc_mask,
        bc_values=bc_values,
    )
    if use_ghb:
        gh_mask = np.zeros((ny, nx), dtype=np.int32)
        gh_head = np.zeros((ny, nx), dtype=np.float64)
        gh_width = np.zeros((ny, nx), dtype=np.float64)
        gh_mask[1:-1, 1] = 1
        gh_head[1:-1, 1] = 10.75
        gh_width[1:-1, 1] = 3.0
        build_kwargs.update(gh_mask=gh_mask, gh_head=gh_head, gh_width=gh_width)
    solver.build_from_fields(**build_kwargs)
    return solver


def _kcycle_controls() -> dict:
    return dict(
        max_cycles=8,
        max_levels=2,
        min_coarse_cells=1,
        check_every_no=1,
        nu_coarse=30,
        cheby_lambda_min=0.05,
        cheby_lambda_max=1.95,
        rel_tol=1.0e-8,
        abs_tol_min=1.0e-8,
        return_info=True,
    )


def _assert_equivalent(actual, reference) -> None:
    actual_head, actual_info = actual
    reference_head, reference_info = reference
    np.testing.assert_allclose(actual_head, reference_head, rtol=0.0, atol=1.0e-12)
    for key in ("converged", "n_cycles_used", "n_iter", "r_rms_end", "h_rms_end", "final_residual"):
        if key in reference_info:
            assert actual_info[key] == reference_info[key]
    assert set(reference_info).issubset(actual_info)


def test_confined_kcycle_backend_applies_tuned_cold_solve_defaults(monkeypatch):
    """The confined backend owns the tuned defaults without affecting Picard."""
    from DARCY_WARP_PACKAGE.solvers import multigrid_kcycle

    captured: dict = {}

    def record_backend_call(*, model, **kwargs):
        captured["model"] = model
        captured.update(kwargs)
        return "result"

    monkeypatch.setattr(
        multigrid_kcycle,
        "solve_multigrid_kcycle_backend",
        record_backend_call,
    )
    model = object()
    context = SimpleNamespace(model=model)

    result = multigrid_kcycle.ConfinedKCycleBackend().solve(
        context,
        check_every_no=7,
    )

    assert result == "result"
    assert captured["model"] is model
    assert captured["unconfined"] is False
    assert captured["nu_pre"] == 2
    assert captured["nu_post"] == 2
    assert captured["nu_coarse"] == 2
    assert captured["max_levels"] == 6
    assert captured["check_every_no"] == 7
    assert captured["smoother"] == "chebyshev"
    assert captured["cheby_lambda_min"] == 0.1
    assert captured["cheby_lambda_max"] == 2.0


def test_registry_selection_and_kcycle_backend_are_numerically_equivalent():
    """Named backends retain the canonical K-cycle result exactly on CPU."""
    # Confined steady K-cycle with heterogeneous T, inactive cell, Dirichlet,
    # and GHB data: direct backend entry versus registry dispatch.
    direct = _build_solver(use_ghb=True)
    reference = direct._solve_multigrid_kcycle_backend(**_kcycle_controls())
    selected = _build_solver(use_ghb=True)
    actual = selected.solve(solver="confined_kcycle", **_kcycle_controls())
    _assert_equivalent(actual, reference)

    # Confined transient K-cycle retains storage construction and budget inputs.
    head_prev = np.full((6, 8), 10.0, dtype=np.float64)
    transient_controls = _kcycle_controls()
    transient_controls.update(transient=True, storage_coeff=0.15, dt=2.0, head_prev=head_prev)
    direct = _build_solver()
    reference = direct._solve_multigrid_kcycle_backend(**transient_controls)
    selected = _build_solver()
    actual = selected.solve(solver="confined_kcycle", **transient_controls)
    _assert_equivalent(actual, reference)

    # Unconfined steady and transient Picard calls use the same canonical Picard
    # loop, including near-dry cells and its diagnostic/fallback fields.
    K = np.full((6, 8), 1.0, dtype=np.float64)
    bottom = np.full((6, 8), 9.85, dtype=np.float64)
    initial = np.full((6, 8), 10.0, dtype=np.float64)
    picard_controls = _kcycle_controls()
    picard_controls.update(
        unconfined=True,
        K_field=K,
        zbot_field=bottom,
        initial_head=initial,
        max_outer_iterations=4,
        hclose=1.0e-4,
    )
    direct = _build_solver()
    reference = direct._solve_multigrid_kcycle_backend(**picard_controls)
    selected = _build_solver()
    actual = selected.solve(
        formulation="unconfined",
        solver="unconfined_picard_kcycle",
        **picard_controls,
    )
    _assert_equivalent(actual, reference)
    assert "outer_history" in actual[1]

    transient_picard = dict(picard_controls)
    transient_picard.update(transient=True, storage_coeff=0.15, dt=2.0, head_prev=initial)
    direct = _build_solver()
    reference = direct._solve_multigrid_kcycle_backend(**transient_picard)
    selected = _build_solver()
    actual = selected.solve(
        formulation="unconfined",
        solver="kcycle",  # retained legacy alias
        **transient_picard,
    )
    _assert_equivalent(actual, reference)


def test_confined_pcg_named_backend_matches_extracted_device_loop():
    direct = _build_solver()
    reference = direct._solve_pcg_device_loop(
        max_iter=80,
        rel_tol=1.0e-8,
        abs_tol_min=1.0e-8,
        initial_head=None,
        history_every=None,
    )
    selected = _build_solver()
    actual = selected.solve(
        solver="confined_pcg",
        max_iter=80,
        rel_tol=1.0e-8,
        abs_tol_min=1.0e-8,
    )
    _assert_equivalent(actual, reference)


def test_solver_selection_validation_and_aliases():
    from DARCY_WARP_PACKAGE.factory import create_solver
    from DARCY_WARP_PACKAGE.solvers import (
        CAPABILITIES,
        available_backends,
        canonical_solver_name,
        select_backend,
    )

    assert set(available_backends()) == {
        "confined_pcg",
        "confined_kcycle",
        "unconfined_picard_kcycle",
        "unconfined_semismooth_newton_kcycle",
        "unconfined_fas",
    }
    assert canonical_solver_name("pcg", formulation="confined", default="kcycle") == "confined_pcg"
    assert canonical_solver_name("mg", formulation="confined", default="pcg") == "confined_kcycle"
    assert canonical_solver_name("kcycle", formulation="unconfined", default="pcg") == "unconfined_picard_kcycle"
    assert canonical_solver_name(
        "unconfined_semismooth_newton_kcycle",
        formulation="unconfined",
        default="kcycle",
    ) == "unconfined_semismooth_newton_kcycle"
    assert CAPABILITIES["unconfined_picard_kcycle"].production_default
    assert CAPABILITIES["unconfined_picard_kcycle"].supports_production_period_driver
    assert CAPABILITIES["confined_kcycle"].production_default
    assert not CAPABILITIES["unconfined_semismooth_newton_kcycle"].experimental
    assert not CAPABILITIES["unconfined_semismooth_newton_kcycle"].production_default
    # Non-default nonlinear backends may drive the multi-period dispatcher
    # through the alternate driver branch. Only FAS remains experimental.
    assert CAPABILITIES["unconfined_semismooth_newton_kcycle"].supports_production_period_driver
    assert CAPABILITIES["unconfined_fas"].experimental
    assert not CAPABILITIES["unconfined_fas"].production_default
    assert CAPABILITIES["unconfined_fas"].supports_production_period_driver
    assert select_backend(
        solver="unconfined_semismooth_newton_kcycle",
        formulation="unconfined",
        transient=True,
        default="unconfined_picard_kcycle",
    ).name == "unconfined_semismooth_newton_kcycle"
    with pytest.warns(UserWarning, match="experimental"):
        assert select_backend(
            solver="unconfined_fas",
            formulation="unconfined",
            transient=True,
            default="unconfined_picard_kcycle",
        ).name == "unconfined_fas"
    with pytest.raises(ValueError, match="requires formulation='unconfined'"):
        canonical_solver_name("unconfined_picard_kcycle", formulation="confined", default="pcg")
    with pytest.raises(ValueError, match="requires formulation='unconfined'"):
        canonical_solver_name(
            "unconfined_semismooth_newton_kcycle",
            formulation="confined",
            default="pcg",
        )
    with pytest.raises(ValueError, match="currently requires"):
        canonical_solver_name("pcg", formulation="unconfined", default="kcycle")
    with pytest.raises(ValueError, match="unknown 2D solver backend"):
        canonical_solver_name("not-a-solver", formulation="confined", default="pcg")

    # Construction remains formulation-neutral; the explicit unconfined
    # backend simply chooses a K-cycle-capable model preference.
    factory_solver = create_solver(
        dim=2,
        nx=4,
        ny=3,
        dx=1.0,
        device="cpu",
        solver="unconfined_picard_kcycle",
    )
    assert factory_solver.solver_type == "kcycle"
    factory_solver.close()

    newton_factory_solver = create_solver(
        dim=2,
        nx=4,
        ny=3,
        dx=1.0,
        device="cpu",
        solver="unconfined_semismooth_newton_kcycle",
    )
    assert newton_factory_solver.solver_type == "kcycle"
    newton_factory_solver.close()

    fas_factory_solver = create_solver(
        dim=2,
        nx=4,
        ny=3,
        dx=1.0,
        device="cpu",
        solver="unconfined_fas",
    )
    assert fas_factory_solver.solver_type == "kcycle"
    fas_factory_solver.close()

    solver = _build_solver()
    with pytest.raises(NotImplementedError, match="does not support transient storage"):
        solver.solve(solver="confined_pcg", transient=True, storage_coeff=0.1, dt=1.0)


def test_typed_context_borrows_model_resources_and_close_is_idempotent():
    from DARCY_WARP_PACKAGE.physics.operator_data import (
        BoundaryFields,
        GridSpec,
        OperatorFields,
        StorageState,
    )
    from DARCY_WARP_PACKAGE.solvers.context import (
        ConvergenceControls,
        MultigridHierarchy,
        SolverWorkspace,
    )

    solver = _build_solver()
    context = solver._make_solver_context(formulation="confined", transient=False)
    assert isinstance(context.grid, GridSpec)
    assert isinstance(context.fields, OperatorFields)
    assert isinstance(context.boundaries, BoundaryFields)
    assert isinstance(context.storage, StorageState)
    assert isinstance(context.hierarchy, MultigridHierarchy)
    assert isinstance(context.workspace, SolverWorkspace)
    assert isinstance(context.convergence, ConvergenceControls)
    assert context.fields.transmissivity is solver.T_wp
    assert context.boundaries.active is solver.active_wp
    solver.close()
    solver.close()
    assert solver._resource_owner.closed


@pytest.mark.skipif(
    not _warp_available(), reason="warp is not available",
)
def test_repeated_cuda_kcycle_solves_reuse_allocations_when_cuda_available():
    import warp as wp

    if not wp.is_cuda_available():
        pytest.skip("CUDA is not available")
    device = "cuda:0"
    solver = _build_solver(device=device)
    controls = _kcycle_controls()
    try:
        first_head, _ = solver.solve(solver="confined_kcycle", **controls)
        wp.synchronize_device(device)
        used_after_first = wp.get_mempool_used_mem_current(device)
        second_head, _ = solver.solve(solver="confined_kcycle", **controls)
        wp.synchronize_device(device)
        used_after_second = wp.get_mempool_used_mem_current(device)
        np.testing.assert_allclose(second_head, first_head, rtol=0.0, atol=1.0e-12)
        assert used_after_second <= used_after_first
    finally:
        solver.close()
        gc.collect()

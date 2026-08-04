"""Regression coverage for the fast steady-confined K-cycle implementation.

``implementation="fast"`` on the confined K-cycle backend (production, FP64
only): face-conductance kernels, block-reduced reductions, Jacobi-block
coarsest, CUDA-graph capture.  The classic implementation is the default and
must be unaffected.
"""

from __future__ import annotations

import numpy as np
import pytest


def _warp_available() -> bool:
    try:
        import warp  # noqa: F401
    except Exception:
        return False
    return True


def _cuda_available() -> bool:
    if not _warp_available():
        return False
    try:
        import warp as wp

        return bool(wp.is_cuda_available())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _warp_available(), reason="warp is not available")
requires_cuda = pytest.mark.skipif(not _cuda_available(), reason="CUDA is not available")

_NX, _NY = 48, 40


def _case_fields() -> dict:
    """Field arrays for the small het+GHB case (fresh copies per call)."""
    y, x = np.mgrid[:_NY, :_NX]
    active = np.ones((_NY, _NX), dtype=np.int32)
    active[2, 3] = 0
    bc_mask = np.zeros((_NY, _NX), dtype=np.int32)
    bc_mask[:, 0] = 1
    bc_mask[:, -1] = 1
    bc_values = np.zeros((_NY, _NX), dtype=np.float64)
    bc_values[:, 0] = 12.0
    bc_values[:, -1] = 9.0
    T = (50.0 + 2.5 * x + 5.0 * y).astype(np.float64)
    T[5, 10] = 4000.0  # high-contrast cell
    R = np.full((_NY, _NX), 1.0e-4, dtype=np.float64)
    gh_mask = np.zeros((_NY, _NX), dtype=np.int32)
    gh_head = np.zeros((_NY, _NX), dtype=np.float64)
    gh_width = np.zeros((_NY, _NX), dtype=np.float64)
    gh_mask[1:-1, 1] = 1
    gh_head[1:-1, 1] = 10.75
    gh_width[1:-1, 1] = 3.0
    return dict(
        T_field=T, R_field=R, active=active, bc_mask=bc_mask,
        bc_values=bc_values, gh_mask=gh_mask, gh_head=gh_head, gh_width=gh_width,
    )


def _build_solver(device: str = "cpu"):
    from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver

    solver = WarpDarcySolver(
        nx=_NX, ny=_NY, dx=100.0, device=device, use_ghb=True,
        solver_type="pcg", aq_thickness=300.0,
    )
    solver.build_from_fields(**_case_fields())
    return solver


def _controls(**extra):
    ctrl = dict(
        max_cycles=60, nu_pre=2, nu_post=2, omega=0.7,
        rel_tol=5.0e-7, abs_tol_min=5.0e-7,
        return_info=True, max_levels=3, min_coarse_cells=50, check_every_no=5,
    )
    ctrl.update(extra)
    return ctrl


@requires_cuda
def test_fast_matches_classic_on_small_case():
    """Heterogeneous T + GHB + inactive cell: fast and classic both converge
    and agree far below the production MF6 gate (2e-4 m)."""
    classic_solver = _build_solver(device="cuda:0")
    fast_solver = _build_solver(device="cuda:0")
    try:
        head_c, info_c = classic_solver.solve_multigrid_kcycle(
            nu_coarse=2, **_controls())
        head_f, info_f = fast_solver.solve_multigrid_kcycle(
            nu_coarse=10, implementation="fast", **_controls())
        assert info_c["converged"] is True
        assert info_f["converged"] is True
        assert info_f["implementation"] == "fast"
        # Same initial residual up to reduction-order round-off.
        assert np.isclose(info_f["tol_abs"], info_c["tol_abs"], rtol=1.0e-12, atol=0.0)
        # Measured agreement is ~3e-6 m (different cycle mechanics at the same
        # convergence criterion); the regression bound stays 100x tighter than
        # the production MF6 gate.
        np.testing.assert_allclose(
            np.asarray(head_f, dtype=np.float64),
            np.asarray(head_c, dtype=np.float64),
            rtol=0.0, atol=2.0e-5,
        )
    finally:
        classic_solver.close()
        fast_solver.close()


@requires_cuda
def test_fast_repeated_solves_reuse_graph_and_match():
    solver = _build_solver(device="cuda:0")
    try:
        ctrl = _controls(nu_coarse=10, implementation="fast")
        head1, info1 = solver.solve_multigrid_kcycle(**ctrl)
        head2, info2 = solver.solve_multigrid_kcycle(**ctrl)
        assert info2["cuda_graph_reused"] is True
        # Capture launches the new graph immediately, so both solves execute
        # identical cycles and the converged iterates agree well within the
        # production MF6 gate (measured ~3e-7 m).
        np.testing.assert_allclose(
            np.asarray(head1, dtype=np.float64),
            np.asarray(head2, dtype=np.float64),
            rtol=0.0, atol=2.0e-5,
        )
    finally:
        solver.close()


@requires_cuda
def test_fast_face_cache_invalidated_by_transmissivity_update():
    """After update_T_in_place, the fast path rebuilds faces and still agrees
    with classic on the updated operator."""
    classic_solver = _build_solver(device="cuda:0")
    fast_solver = _build_solver(device="cuda:0")
    try:
        ctrl_c = _controls(nu_coarse=2)
        ctrl_f = _controls(nu_coarse=10, implementation="fast")
        classic_solver.solve_multigrid_kcycle(**ctrl_c)
        fast_solver.solve_multigrid_kcycle(**ctrl_f)

        y, x = np.mgrid[:_NY, :_NX]
        T_new = (80.0 + 1.5 * x + 3.0 * y).astype(np.float64)
        classic_solver.update_T_in_place(T_new)
        fast_solver.update_T_in_place(T_new)

        head_c, info_c = classic_solver.solve_multigrid_kcycle(**ctrl_c)
        head_f, info_f = fast_solver.solve_multigrid_kcycle(**ctrl_f)
        assert info_c["converged"] is True
        assert info_f["converged"] is True
        np.testing.assert_allclose(
            np.asarray(head_f, dtype=np.float64),
            np.asarray(head_c, dtype=np.float64),
            rtol=0.0, atol=2.0e-5,
        )
    finally:
        classic_solver.close()
        fast_solver.close()


def test_fast_accepts_confined_transient_and_rejects_unconfined():
    solver = _build_solver(device="cpu")
    try:
        head_prev = np.full((_NY, _NX), 10.0, dtype=np.float64)
        head, info = solver.solve_multigrid_kcycle(
            implementation="fast", transient=True, storage_coeff=0.1, dt=1.0,
            head_prev=head_prev, **_controls())
        assert info["transient"] is True
        assert info["implementation"] == "fast_transient_face_f64"
        assert np.isfinite(head).all()
        with pytest.raises(ValueError, match="steady confined only"):
            solver.solve_multigrid_kcycle(
                implementation="fast", unconfined=True, **_controls())
        with pytest.raises(ValueError, match="'classic' or 'fast'"):
            solver.solve_multigrid_kcycle(
                implementation="ludicrous", **_controls())
    finally:
        solver.close()


def test_transient_face_cache_releases_on_close_and_rebuild():
    """Transient face levels are hierarchy-owned and never outlive it."""
    solver = _build_solver(device="cpu")
    try:
        head_prev = np.full((_NY, _NX), 10.0, dtype=np.float64)
        solver.solve_multigrid_kcycle(
            implementation="fast", transient=True, storage_coeff=0.1, dt=1.0,
            head_prev=head_prev, transient_face_graphs_enabled=False, **_controls())
        assert solver._transient_face_cache is not None
        solver.build_from_fields(**_case_fields())
        solver.solve_multigrid_kcycle(**_controls())
        assert solver._transient_face_cache is None
    finally:
        solver.close()
    assert solver._transient_face_cache is None


def test_confined_transient_fast_matches_classic_for_scalar_and_array_storage():
    """The opt-in face path preserves transient heads as dt/storage change."""
    classic_solver = _build_solver(device="cpu")
    fast_solver = _build_solver(device="cpu")
    try:
        head_prev = np.full((_NY, _NX), 10.0, dtype=np.float64)
        controls = _controls(max_cycles=80, nu_coarse=2)
        head_classic, info_classic = classic_solver.solve_multigrid_kcycle(
            transient=True, storage_coeff=0.1, dt=1.0, head_prev=head_prev, **controls)
        head_fast, info_fast = fast_solver.solve_multigrid_kcycle(
            implementation="fast", transient=True, storage_coeff=0.1, dt=1.0,
            head_prev=head_prev, **controls)
        assert info_classic["converged"] is True
        assert info_fast["converged"] is True
        np.testing.assert_allclose(head_fast, head_classic, rtol=0.0, atol=2.0e-5)

        storage = np.full((_NY, _NX), 0.08, dtype=np.float64)
        head_classic_2, info_classic_2 = classic_solver.solve_multigrid_kcycle(
            transient=True, storage_coeff=storage, dt=2.0, head_prev=head_classic, **controls)
        head_fast_2, info_fast_2 = fast_solver.solve_multigrid_kcycle(
            implementation="fast", transient=True, storage_coeff=storage, dt=2.0,
            head_prev=head_fast, **controls)
        assert info_classic_2["converged"] is True
        assert info_fast_2["converged"] is True
        np.testing.assert_allclose(head_fast_2, head_classic_2, rtol=0.0, atol=2.0e-5)
    finally:
        classic_solver.close()
        fast_solver.close()


def test_confined_transient_fast_matches_classic_odd_grid_no_ghb():
    """Odd, non-square, no-GHB confined transient: fast parity with classic."""
    from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver

    nx, ny = 45, 31
    y, x = np.mgrid[:ny, :nx]
    active = np.ones((ny, nx), dtype=np.int32)
    bc_mask = np.zeros((ny, nx), dtype=np.int32)
    bc_mask[:, 0] = 1
    bc_mask[:, -1] = 1
    bc_values = np.zeros((ny, nx), dtype=np.float64)
    bc_values[:, 0] = 12.0
    bc_values[:, -1] = 9.0
    T = (60.0 + 1.0 * x + 2.0 * y).astype(np.float64)
    R = np.full((ny, nx), 1.0e-4, dtype=np.float64)
    fields = dict(T_field=T, R_field=R, active=active, bc_mask=bc_mask, bc_values=bc_values)
    head_prev = np.full((ny, nx), 10.0, dtype=np.float64)
    head_prev[:, 0] = 12.0
    head_prev[:, -1] = 9.0

    classic_solver = WarpDarcySolver(nx=nx, ny=ny, dx=100.0, device="cpu", use_ghb=False, solver_type="pcg")
    fast_solver = WarpDarcySolver(nx=nx, ny=ny, dx=100.0, device="cpu", use_ghb=False, solver_type="pcg")
    try:
        classic_solver.build_from_fields(**fields)
        fast_solver.build_from_fields(**fields)
        controls = _controls(max_cycles=80, nu_coarse=2)
        head_classic, info_classic = classic_solver.solve_multigrid_kcycle(
            transient=True, storage_coeff=0.1, dt=1.0, head_prev=head_prev, **controls)
        head_fast, info_fast = fast_solver.solve_multigrid_kcycle(
            implementation="fast", transient=True, storage_coeff=0.1, dt=1.0,
            head_prev=head_prev, **controls)
        assert info_classic["converged"] is True
        assert info_fast["converged"] is True
        np.testing.assert_allclose(head_fast, head_classic, rtol=0.0, atol=2.0e-5)
    finally:
        classic_solver.close()
        fast_solver.close()


@requires_cuda
def test_confined_transient_fast_graph_matches_eager():
    """Captured scalar-info K-cycle replays are numerically identical to eager."""
    solver = _build_solver(device="cuda:0")
    try:
        head_prev = np.full((_NY, _NX), 10.0, dtype=np.float64)
        controls = _controls(max_cycles=80, nu_coarse=2)
        head_eager, info_eager = solver.solve_multigrid_kcycle(
            implementation="fast", transient=True, storage_coeff=0.1, dt=1.0,
            head_prev=head_prev, transient_face_graphs_enabled=False, **controls)
        head_graph, info_graph = solver.solve_multigrid_kcycle(
            implementation="fast", transient=True, storage_coeff=0.1, dt=1.0,
            head_prev=head_prev, transient_face_graphs_enabled=True, **controls)
        assert info_eager["converged"] is True
        assert info_graph["converged"] is True
        assert int(info_eager["graph_count"]) == 0
        assert int(info_graph["graph_count"]) >= 1
        assert int(info_graph["graph_fallback_count"]) == 0
        np.testing.assert_allclose(
            np.asarray(head_graph, dtype=np.float64),
            np.asarray(head_eager, dtype=np.float64),
            rtol=0.0, atol=1.0e-9,
        )
    finally:
        solver.close()


@requires_cuda
def test_fast_graph_invalidated_by_classic_hierarchy_rebuild():
    """One instance: fast -> dirty/rebuild via classic -> fast.

    The classic rebuild replaces the hierarchy buffers, so the second fast
    call must NOT replay the graph captured against the old buffers: it must
    build a new graph, converge, and match a fresh classic reference.
    """
    solver = _build_solver(device="cuda:0")
    reference = _build_solver(device="cuda:0")
    try:
        ctrl_f = _controls(nu_coarse=10, implementation="fast")
        head_f1, info_f1 = solver.solve_multigrid_kcycle(**ctrl_f)
        assert info_f1["converged"] is True
        assert info_f1["cuda_graph_built_this_call"] is True

        # Dirty the operator and rebuild the hierarchy through the CLASSIC
        # backend (build_from_fields sets _operator_dirty; the classic solve
        # consumes it and replaces mg_levels).
        solver.build_from_fields(**_case_fields())
        head_c, info_c = solver.solve_multigrid_kcycle(**_controls(nu_coarse=2))
        assert info_c["converged"] is True

        head_f2, info_f2 = solver.solve_multigrid_kcycle(**ctrl_f)
        assert info_f2["converged"] is True
        assert info_f2["cuda_graph_built_this_call"] is True, (
            "fast graph survived a classic hierarchy rebuild (stale-buffer replay)"
        )

        head_ref, info_ref = reference.solve_multigrid_kcycle(**_controls(nu_coarse=2))
        assert info_ref["converged"] is True
        np.testing.assert_allclose(
            np.asarray(head_f2, dtype=np.float64),
            np.asarray(head_ref, dtype=np.float64),
            rtol=0.0, atol=2.0e-5,
        )
    finally:
        solver.close()
        reference.close()


@requires_cuda
def test_close_releases_fast_graph_and_face_cache():
    solver = _build_solver(device="cuda:0")
    _, info = solver.solve_multigrid_kcycle(
        nu_coarse=10, implementation="fast", **_controls()
    )
    assert info["converged"] is True
    solver.close()
    assert solver._kcycle_graph is None
    assert solver._kcycle_fast_graph is None
    assert solver._kcycle_fast_graph_shape is None
    assert getattr(solver, "_fast_face_cache", None) is None


@requires_cuda
def test_fast_honours_per_call_ghb_parameters():
    """aq_thickness / gh_alpha overrides must reach the fast backend."""
    classic_solver = _build_solver(device="cuda:0")
    fast_solver = _build_solver(device="cuda:0")
    try:
        # Scalar override (model was built with aq_thickness=300.0).
        head_c, info_c = classic_solver.solve_multigrid_kcycle(
            nu_coarse=2, aq_thickness=150.0, **_controls()
        )
        head_f, info_f = fast_solver.solve_multigrid_kcycle(
            nu_coarse=10, implementation="fast", aq_thickness=150.0, **_controls()
        )
        assert info_c["converged"] is True
        assert info_f["converged"] is True
        assert info_f["aq_thickness"] == 150.0
        np.testing.assert_allclose(
            np.asarray(head_f, dtype=np.float64),
            np.asarray(head_c, dtype=np.float64),
            rtol=0.0, atol=2.0e-5,
        )

        # Array-valued gh_alpha override.
        y, x = np.mgrid[:_NY, :_NX]
        gh_alpha = (0.5 + 0.01 * (x + y)).astype(np.float64)
        head_c2, info_c2 = classic_solver.solve_multigrid_kcycle(
            nu_coarse=2, gh_alpha=gh_alpha, **_controls()
        )
        head_f2, info_f2 = fast_solver.solve_multigrid_kcycle(
            nu_coarse=10, implementation="fast", gh_alpha=gh_alpha, **_controls()
        )
        assert info_c2["converged"] is True
        assert info_f2["converged"] is True
        np.testing.assert_allclose(
            np.asarray(head_f2, dtype=np.float64),
            np.asarray(head_c2, dtype=np.float64),
            rtol=0.0, atol=2.0e-5,
        )
        # The override must change the answer (i.e. not silently ignored).
        assert not np.allclose(
            np.asarray(head_f, dtype=np.float64),
            np.asarray(head_f2, dtype=np.float64),
            rtol=0.0, atol=1.0e-8,
        )
    finally:
        classic_solver.close()
        fast_solver.close()


def test_classic_remains_default_and_unchanged():
    """The default call path is the classic implementation."""
    from DARCY_WARP_PACKAGE.solvers import multigrid_kcycle

    captured = {}

    original = multigrid_kcycle.solve_multigrid_kcycle_backend

    def spy(*, model, **kwargs):
        captured["implementation"] = kwargs.get("implementation", "classic")
        return original(model=model, **kwargs)

    from DARCY_WARP_PACKAGE.solvers.multigrid_kcycle import ConfinedKCycleBackend

    backend = ConfinedKCycleBackend()
    from types import SimpleNamespace

    solver = _build_solver(device="cpu")
    try:
        context = solver._make_solver_context(formulation="confined", transient=False)
        multigrid_kcycle.solve_multigrid_kcycle_backend = spy
        try:
            backend.solve(context, max_cycles=1, check_every_no=1, nu_coarse=2)
        finally:
            multigrid_kcycle.solve_multigrid_kcycle_backend = original
        assert captured["implementation"] == "classic"
    finally:
        solver.close()

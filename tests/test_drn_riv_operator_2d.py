# SPDX-License-Identifier: AGPL-3.0-only
"""Validation for the gated MODFLOW DRN/RIV boundary terms in the 2D nonlinear operator.

Covers:
1. all four gating regimes with hand-computed expectations on a tiny grid
   (DRN active / DRN inactive / RIV coupled / RIV decoupled);
2. device residual vs the host reference on a mixed DRN+RIV+GHB case, and
   the sparse operator diagonal carrying the gated-on conductances;
3. Jacobian-vector action vs central finite differences away from the kinks;
4. ``WarpDarcySolver.build_from_fields`` sanitisation and backend guards;
5. end-to-end small unconfined semismooth-Newton solves with DRN and RIV
   (incl. a decoupled cell) against a host scipy fixed-point reference.
"""

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("WARP_CACHE_PATH", str(Path("/tmp/darcywarp-warp-cache")))


def _warp_available() -> bool:
    try:
        import warp  # noqa: F401
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(not _warp_available(), reason="warp is not available")

_DEVICE = "cpu"
_ATOL = 1.0e-9
_RTOL = 1.0e-9


def _make_ctx(
    *,
    ny=3,
    nx=3,
    dx=10.0,
    K=None,
    zbot=None,
    ztop=20.0,
    active=None,
    dirichlet_mask=None,
    dirichlet_values=None,
    R_field=None,
    gh_mask=None,
    gh_head=None,
    ghb_factor=None,
    drn_mask=None,
    drn_elev=None,
    drn_cond=None,
    riv_mask=None,
    riv_stage=None,
    riv_rbot=None,
    riv_cond=None,
    min_sat=0.1,
):
    from DARCY_WARP_PACKAGE.nonlinear import from_arrays

    shape = (ny, nx)
    if K is None:
        K = np.full(shape, 2.0, dtype=np.float64)
    if zbot is None:
        zbot = np.zeros(shape, dtype=np.float64)
    if np.ndim(ztop) == 0:
        ztop = np.full(shape, float(ztop), dtype=np.float64)
    if active is None:
        active = np.ones(shape, dtype=np.int32)
    if dirichlet_mask is None:
        dirichlet_mask = np.zeros(shape, dtype=np.int32)
    if dirichlet_values is None:
        dirichlet_values = np.zeros(shape, dtype=np.float64)
    if R_field is None:
        R_field = np.zeros(shape, dtype=np.float64)

    return from_arrays(
        nx=nx,
        ny=ny,
        dx=dx,
        K=K,
        zbot=zbot,
        ztop=ztop,
        active=active,
        dirichlet_mask=dirichlet_mask,
        dirichlet_values=dirichlet_values,
        R_field=R_field,
        ghb_mask=gh_mask,
        ghb_external_head=gh_head,
        ghb_factor=ghb_factor,
        drn_mask=drn_mask,
        drn_elev=drn_elev,
        drn_cond=drn_cond,
        riv_mask=riv_mask,
        riv_stage=riv_stage,
        riv_rbot=riv_rbot,
        riv_cond=riv_cond,
        min_sat=min_sat,
        device=_DEVICE,
    )


def _residual_np(op, head):
    out = op.residual(head)
    return np.array(out.numpy() if hasattr(out, "numpy") else out, copy=True)


def _jacobian_vector_np(op, head, vector):
    out = op.jacobian_vector(head, vector)
    return np.array(out.numpy() if hasattr(out, "numpy") else out, copy=True)


# ---------------------------------------------------------------------------
# 1. Gating regimes with hand-computed expectations
# ---------------------------------------------------------------------------


def test_drn_active_contribution():
    """DRN with h > elev contributes C*(h - elev) to the residual."""
    from DARCY_WARP_PACKAGE.nonlinear import NonlinearOperator2D

    drn_mask = np.zeros((3, 3), dtype=np.int32)
    drn_mask[1, 1] = 1
    drn_elev = np.zeros((3, 3))
    drn_elev[1, 1] = 5.0
    drn_cond = np.zeros((3, 3))
    drn_cond[1, 1] = 7.0
    ctx = _make_ctx(drn_mask=drn_mask, drn_elev=drn_elev, drn_cond=drn_cond)
    head = np.full((3, 3), 9.0)  # uniform -> zero flow gradients; h > elev
    op = NonlinearOperator2D(ctx)
    F = _residual_np(op, head)
    assert F[1, 1] == pytest.approx(7.0 * (9.0 - 5.0), abs=_ATOL)
    Foff = F.copy()
    Foff[1, 1] = 0.0
    np.testing.assert_allclose(Foff, 0.0, atol=_ATOL, rtol=0.0)
    op.close()


def test_drn_inactive_contributes_nothing():
    """DRN with h <= elev is gated off: zero residual contribution."""
    from DARCY_WARP_PACKAGE.nonlinear import NonlinearOperator2D

    drn_mask = np.zeros((3, 3), dtype=np.int32)
    drn_mask[1, 1] = 1
    drn_elev = np.zeros((3, 3))
    drn_elev[1, 1] = 5.0
    drn_cond = np.zeros((3, 3))
    drn_cond[1, 1] = 7.0
    ctx = _make_ctx(drn_mask=drn_mask, drn_elev=drn_elev, drn_cond=drn_cond)
    head = np.full((3, 3), 4.5)  # h < elev -> inactive; uniform -> no flow
    op = NonlinearOperator2D(ctx)
    F = _residual_np(op, head)
    np.testing.assert_allclose(F, 0.0, atol=_ATOL, rtol=0.0)
    # Exactly at the elevation the gate is also off (h > elev strictly).
    F_at = _residual_np(op, np.full((3, 3), 5.0))
    np.testing.assert_allclose(F_at, 0.0, atol=_ATOL, rtol=0.0)
    op.close()


def test_riv_coupled_contribution():
    """RIV with h > rbot contributes C*(h - stage) to the residual."""
    from DARCY_WARP_PACKAGE.nonlinear import NonlinearOperator2D

    riv_mask = np.zeros((3, 3), dtype=np.int32)
    riv_mask[1, 1] = 1
    riv_stage = np.zeros((3, 3))
    riv_stage[1, 1] = 12.0
    riv_rbot = np.zeros((3, 3))
    riv_rbot[1, 1] = 4.0
    riv_cond = np.zeros((3, 3))
    riv_cond[1, 1] = 7.0
    ctx = _make_ctx(
        riv_mask=riv_mask, riv_stage=riv_stage, riv_rbot=riv_rbot, riv_cond=riv_cond
    )
    head = np.full((3, 3), 9.0)  # h > rbot -> coupled; uniform -> no flow
    op = NonlinearOperator2D(ctx)
    F = _residual_np(op, head)
    assert F[1, 1] == pytest.approx(7.0 * (9.0 - 12.0), abs=_ATOL)
    op.close()


def test_riv_decoupled_is_constant_in_h():
    """RIV with h <= rbot contributes the constant -C*(stage - rbot)."""
    from DARCY_WARP_PACKAGE.nonlinear import NonlinearOperator2D

    riv_mask = np.zeros((3, 3), dtype=np.int32)
    riv_mask[1, 1] = 1
    riv_stage = np.zeros((3, 3))
    riv_stage[1, 1] = 12.0
    riv_rbot = np.zeros((3, 3))
    riv_rbot[1, 1] = 4.0
    riv_cond = np.zeros((3, 3))
    riv_cond[1, 1] = 7.0
    ctx = _make_ctx(
        riv_mask=riv_mask, riv_stage=riv_stage, riv_rbot=riv_rbot, riv_cond=riv_cond
    )
    op = NonlinearOperator2D(ctx)
    # Two different heads, both below rbot (uniform -> no flow gradients).
    F_a = _residual_np(op, np.full((3, 3), 3.5))
    F_b = _residual_np(op, np.full((3, 3), 2.0))
    expected = -7.0 * (12.0 - 4.0)  # -C*(stage - rbot)
    assert F_a[1, 1] == pytest.approx(expected, abs=_ATOL)
    # The decoupled contribution must NOT change with h.
    assert F_a[1, 1] == pytest.approx(F_b[1, 1], abs=_ATOL)
    op.close()


# ---------------------------------------------------------------------------
# 2. Device residual vs host reference on a mixed DRN+RIV+GHB case
# ---------------------------------------------------------------------------


def test_mixed_drn_riv_ghb_device_matches_host_reference():
    from DARCY_WARP_PACKAGE.nonlinear import NonlinearOperator2D
    from DARCY_WARP_PACKAGE.nonlinear import reference as ref

    ny, nx = 8, 10
    rng = np.random.default_rng(17)
    K = (0.5 + 4.0 * rng.random((ny, nx))).astype(np.float64)
    zbot = np.zeros((ny, nx))
    ztop = np.full((ny, nx), 25.0)
    active = np.ones((ny, nx), dtype=np.int32)
    active[3, 4] = 0
    dirichlet = np.zeros((ny, nx), dtype=np.int32)
    dirichlet[0, :] = 1
    dhv = np.zeros((ny, nx))
    dhv[0, :] = 18.0

    gh_mask = np.zeros((ny, nx), dtype=np.int32)
    gh_head = np.zeros((ny, nx))
    ghb_factor = np.zeros((ny, nx))
    gh_mask[2, 2:6] = 1
    gh_head[2, :] = 11.0
    ghb_factor[gh_mask != 0] = 0.03

    drn_mask = np.zeros((ny, nx), dtype=np.int32)
    drn_mask[4, 2:8] = 1
    drn_elev = np.full((ny, nx), 6.0)
    # West part of the drain row active, east part gated off at the test head
    # (row-4 heads span ~10.5-11.5).
    drn_elev[4, 5:8] = 11.2
    drn_cond = np.zeros((ny, nx))
    drn_cond[drn_mask != 0] = 3.0 + 2.0 * rng.random(int(np.count_nonzero(drn_mask)))

    riv_mask = np.zeros((ny, nx), dtype=np.int32)
    riv_mask[6, 1:9] = 1
    riv_stage = np.full((ny, nx), 13.0)
    riv_rbot = np.full((ny, nx), 5.0)
    # East part of the river row decoupled at the test head (row-6 heads span
    # ~14.4-15.8): rbot=15.5 with stage=17 keeps stage > rbot physical.
    riv_rbot[6, 5:9] = 15.5
    riv_stage[6, 5:9] = 17.0
    riv_cond = np.zeros((ny, nx))
    riv_cond[riv_mask != 0] = 4.0

    # Head field spans all regimes: some DRN cells above and below elev,
    # some RIV cells coupled and decoupled; all away from the kinks.
    head = zbot + 2.0 + np.linspace(0.0, 16.0, ny * nx).reshape(ny, nx)
    head[0, :] = 18.0

    ctx = _make_ctx(
        ny=ny, nx=nx, K=K, zbot=zbot, ztop=ztop, active=active,
        dirichlet_mask=dirichlet, dirichlet_values=dhv,
        R_field=(rng.random((ny, nx)) - 0.5) * 2.0e-4,
        gh_mask=gh_mask, gh_head=gh_head, ghb_factor=ghb_factor,
        drn_mask=drn_mask, drn_elev=drn_elev, drn_cond=drn_cond,
        riv_mask=riv_mask, riv_stage=riv_stage, riv_rbot=riv_rbot, riv_cond=riv_cond,
    )
    assert np.any(head[drn_mask != 0] > drn_elev[drn_mask != 0])
    assert np.any(head[drn_mask != 0] <= drn_elev[drn_mask != 0])
    assert np.any(head[riv_mask != 0] > riv_rbot[riv_mask != 0])
    assert np.any(head[riv_mask != 0] <= riv_rbot[riv_mask != 0])

    op = NonlinearOperator2D(ctx)
    Fdev = _residual_np(op, head)
    Fhost = ref.nonlinear_residual_host(head, ctx)
    np.testing.assert_allclose(Fdev, Fhost, atol=_ATOL, rtol=_RTOL)

    # The sparse assembly diagonal carries the gated-ON conductances only.
    A = ref.assemble_flow_operator_sparse(head, ctx)
    diag = A.diagonal().reshape(ny, nx)
    gated_diag = ref._gated_boundary_diagonal(head, ctx)
    drn_on = (drn_mask != 0) & (head > drn_elev)
    riv_on = (riv_mask != 0) & (head > riv_rbot)
    for (j, i) in zip(*np.where(drn_on | riv_on)):
        # diagonal = neighbour coupling + gated conductance (no GHB on these rows)
        assert gated_diag[j, i] > 0.0
        assert diag[j, i] >= gated_diag[j, i]
    for (j, i) in zip(*np.where((drn_mask != 0) & ~drn_on)):
        assert gated_diag[j, i] == 0.0
    for (j, i) in zip(*np.where((riv_mask != 0) & ~riv_on)):
        assert gated_diag[j, i] == 0.0
    op.close()


# ---------------------------------------------------------------------------
# 3. Jacobian-vector action vs central finite differences
# ---------------------------------------------------------------------------


def test_jacobian_vector_matches_finite_differences_away_from_kinks():
    from DARCY_WARP_PACKAGE.nonlinear import NonlinearOperator2D

    ny, nx = 5, 6
    rng = np.random.default_rng(23)
    K = (1.0 + 3.0 * rng.random((ny, nx))).astype(np.float64)

    drn_mask = np.zeros((ny, nx), dtype=np.int32)
    drn_mask[2, 2:4] = 1
    drn_elev = np.full((ny, nx), 5.0)
    drn_cond = np.zeros((ny, nx))
    drn_cond[drn_mask != 0] = 6.0

    riv_mask = np.zeros((ny, nx), dtype=np.int32)
    riv_mask[3, 1:5] = 1
    riv_stage = np.full((ny, nx), 12.0)
    riv_rbot = np.full((ny, nx), 4.0)
    riv_cond = np.zeros((ny, nx))
    riv_cond[riv_mask != 0] = 5.0

    ctx = _make_ctx(
        ny=ny, nx=nx, K=K, drn_mask=drn_mask, drn_elev=drn_elev, drn_cond=drn_cond,
        riv_mask=riv_mask, riv_stage=riv_stage, riv_rbot=riv_rbot, riv_cond=riv_cond,
        R_field=np.full((ny, nx), 1.0e-4),
    )
    # Heads strictly away from every kink (elev=5, rbot=4, and sat clip bounds).
    head = 8.0 + 2.0 * rng.random((ny, nx))
    assert np.min(head - drn_elev[drn_mask != 0][0]) > 1.0
    assert np.min(head - riv_rbot[riv_mask != 0][0]) > 1.0

    op = NonlinearOperator2D(ctx)
    v = rng.random((ny, nx))
    Jv = _jacobian_vector_np(op, head, v)
    eps = 1.0e-6
    FDv = (_residual_np(op, head + eps * v) - _residual_np(op, head - eps * v)) / (2.0 * eps)
    scale = max(1.0, float(np.max(np.abs(Jv))))
    assert np.max(np.abs(Jv - FDv)) <= 1.0e-5 * scale
    op.close()


# ---------------------------------------------------------------------------
# 4. build_from_fields sanitisation and backend guards
# ---------------------------------------------------------------------------


def _solver_kwargs(**over):
    kw = dict(
        nx=6, ny=5, dx=50.0, device=_DEVICE,
        solver_type="kcycle", diag_preconditioner_backend="device",
    )
    kw.update(over)
    return kw


def _base_fields(ny=5, nx=6):
    active = np.ones((ny, nx), dtype=np.int32)
    bc_mask = np.zeros((ny, nx), dtype=np.int32)
    bc_values = np.zeros((ny, nx))
    bc_mask[:, 0] = 1
    bc_values[:, 0] = 15.0
    bc_mask[:, -1] = 1
    bc_values[:, -1] = 8.0
    T0 = np.full((ny, nx), 40.0)
    R = np.full((ny, nx), 2.0e-4)
    return dict(T_field=T0, R_field=R, active=active, bc_mask=bc_mask, bc_values=bc_values)


def test_build_from_fields_clears_masks_on_inactive_and_dirichlet_cells():
    from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver

    fields = _base_fields()
    fields["active"][2, 2] = 0  # inactive
    drn_mask = np.zeros((5, 6), dtype=np.int32)
    drn_mask[2, 2] = 1  # on an inactive cell -> must be cleared
    drn_mask[2, 0] = 1  # on a Dirichlet cell -> must be cleared
    drn_mask[1, 3] = 1  # ordinary cell -> kept
    drn_elev = np.full((5, 6), 9.0)
    drn_cond = np.full((5, 6), 80.0)

    solver = WarpDarcySolver(**_solver_kwargs(use_drn=True))
    solver.build_from_fields(**fields, drn_mask=drn_mask, drn_elev=drn_elev, drn_cond=drn_cond)
    assert solver.drn_mask_host[2, 2] == 0
    assert solver.drn_mask_host[2, 0] == 0
    assert solver.drn_mask_host[1, 3] == 1
    solver.close()


def test_build_from_fields_rejects_nonpositive_conductance_on_masked_cells():
    from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver

    drn_mask = np.zeros((5, 6), dtype=np.int32)
    drn_mask[1, 3] = 1
    drn_elev = np.full((5, 6), 9.0)
    drn_cond = np.full((5, 6), 80.0)
    drn_cond[1, 3] = 0.0

    solver = WarpDarcySolver(**_solver_kwargs(use_drn=True))
    try:
        with pytest.raises(ValueError, match="drn_cond"):
            solver.build_from_fields(
                **_base_fields(), drn_mask=drn_mask, drn_elev=drn_elev, drn_cond=drn_cond
            )
    finally:
        solver.close()


def test_build_from_fields_requires_elev_on_masked_cells():
    from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver

    drn_mask = np.zeros((5, 6), dtype=np.int32)
    drn_mask[1, 3] = 1
    solver = WarpDarcySolver(**_solver_kwargs(use_drn=True))
    try:
        with pytest.raises(ValueError, match="drn_elev"):
            solver.build_from_fields(
                **_base_fields(), drn_mask=drn_mask, drn_cond=np.full((5, 6), 80.0)
            )
    finally:
        solver.close()


def test_drn_requires_newton_backend_at_solve_time():
    from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver

    drn_mask = np.zeros((5, 6), dtype=np.int32)
    drn_mask[1, 3] = 1
    solver = WarpDarcySolver(**_solver_kwargs(use_drn=True))
    try:
        solver.build_from_fields(
            **_base_fields(),
            drn_mask=drn_mask,
            drn_elev=np.full((5, 6), 9.0),
            drn_cond=np.full((5, 6), 80.0),
        )
        with pytest.raises(NotImplementedError, match="unconfined_semismooth_newton_kcycle"):
            solver.solve(
                formulation="unconfined",
                solver="unconfined_picard_kcycle",
                K_field=np.full((5, 6), 2.0),
                zbot_field=np.zeros((5, 6)),
                ztop_field=np.full((5, 6), 20.0),
                max_levels=2, min_coarse_cells=1,
            )
    finally:
        solver.close()


# ---------------------------------------------------------------------------
# 5. End-to-end small unconfined Newton solves vs host scipy fixed point
# ---------------------------------------------------------------------------


def _host_fixed_point(K, zbot, ztop, active, bc, bcv, R, dx, *,
                      drn_mask=None, drn_elev=None, drn_cond=None,
                      riv_mask=None, riv_stage=None, riv_rbot=None, riv_cond=None):
    """Fixed-point host reference: freeze gates at the current head, solve the
    sparse linear system with T(h) = K*clip(h-zbot, min_sat, top-zbot), iterate."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    ny, nx = K.shape
    h = np.full((ny, nx), 12.0)
    for _ in range(300):
        sat = np.clip(h - zbot, 0.1, ztop - zbot)
        T = K * sat
        rows, cols, vals, b = [], [], [], np.zeros(ny * nx)
        for j in range(ny):
            for i in range(nx):
                k = j * nx + i
                if bc[j, i]:
                    rows.append(k); cols.append(k); vals.append(1.0)
                    b[k] = bcv[j, i]
                    continue
                diag = 0.0
                for dj, di in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    nj, ni = j + dj, i + di
                    if 0 <= nj < ny and 0 <= ni < nx and active[nj, ni]:
                        a_, b_ = T[j, i], T[nj, ni]
                        if a_ > 0 and b_ > 0:
                            C = 2 * a_ * b_ / (a_ + b_ + 1e-12)
                            diag += C
                            rows.append(k); cols.append(nj * nx + ni); vals.append(-C)
                if drn_mask is not None and drn_mask[j, i] and h[j, i] > drn_elev[j, i]:
                    diag += drn_cond[j, i]
                    b[k] += drn_cond[j, i] * drn_elev[j, i]
                if riv_mask is not None and riv_mask[j, i]:
                    if h[j, i] > riv_rbot[j, i]:
                        diag += riv_cond[j, i]
                        b[k] += riv_cond[j, i] * riv_stage[j, i]
                    else:
                        b[k] += riv_cond[j, i] * (riv_stage[j, i] - riv_rbot[j, i])
                b[k] += R[j, i] * dx * dx
                rows.append(k); cols.append(k); vals.append(diag)
        A = sp.csr_matrix((vals, (rows, cols)), shape=(ny * nx, ny * nx))
        h_new = spla.spsolve(A, b).reshape(ny, nx)
        if np.max(np.abs(h_new - h)) < 1e-12:
            h = h_new
            break
        h = h_new
    return h


def _small_case_fields():
    ny, nx, dx = 5, 9, 50.0
    K = np.full((ny, nx), 2.0)
    zbot = np.zeros((ny, nx))
    ztop = np.full((ny, nx), 20.0)
    active = np.ones((ny, nx), dtype=np.int32)
    bc = np.zeros((ny, nx), dtype=np.int32)
    bcv = np.zeros((ny, nx))
    bc[:, 0] = 1
    bcv[:, 0] = 15.0
    bc[:, -1] = 1
    bcv[:, -1] = 8.0
    R = np.full((ny, nx), 2.0e-4)
    T0 = K * (ztop - zbot)
    return ny, nx, dx, K, zbot, ztop, active, bc, bcv, R, T0


def test_end_to_end_unconfined_newton_with_drn_matches_host_fixed_point():
    from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver

    ny, nx, dx, K, zbot, ztop, active, bc, bcv, R, T0 = _small_case_fields()
    drn_mask = np.zeros((ny, nx), dtype=np.int32)
    drn_mask[ny // 2, 2:7] = 1
    drn_elev = np.full((ny, nx), 9.5)  # below expected heads -> drains active mid-row
    drn_cond = np.full((ny, nx), 80.0)

    with WarpDarcySolver(
        nx=nx, ny=ny, dx=dx, device=_DEVICE, use_drn=True,
        solver_type="kcycle", diag_preconditioner_backend="device",
    ) as solver:
        solver.build_from_fields(
            T_field=T0, R_field=R, active=active, bc_mask=bc, bc_values=bcv,
            drn_mask=drn_mask, drn_elev=drn_elev, drn_cond=drn_cond,
        )
        head, info = solver.solve(
            formulation="unconfined", K_field=K, zbot_field=zbot, ztop_field=ztop,
            solver="unconfined_semismooth_newton_kcycle",
            max_levels=2, min_coarse_cells=1, return_info=True,
        )
    assert info.get("converged", False), info.get("newton_failure_reason")

    h_ref = _host_fixed_point(
        K, zbot, ztop, active, bc, bcv, R, dx,
        drn_mask=drn_mask, drn_elev=drn_elev, drn_cond=drn_cond,
    )
    assert np.max(np.abs(head - h_ref)) < 1.0e-6
    # At least one drain is active, and the gate pattern matches the reference.
    assert np.any(head[ny // 2, 2:7] > drn_elev[ny // 2, 2:7])
    np.testing.assert_array_equal(
        head[ny // 2, 2:7] > drn_elev[ny // 2, 2:7],
        h_ref[ny // 2, 2:7] > drn_elev[ny // 2, 2:7],
    )


def test_end_to_end_unconfined_newton_with_riv_matches_host_fixed_point():
    from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver

    ny, nx, dx, K, zbot, ztop, active, bc, bcv, R, T0 = _small_case_fields()
    riv_mask = np.zeros((ny, nx), dtype=np.int32)
    riv_mask[ny // 2, 2:7] = 1
    riv_stage = np.full((ny, nx), 14.0)
    riv_rbot = np.full((ny, nx), 10.5)
    riv_cond = np.full((ny, nx), 80.0)
    # Force the last river cell decoupled: rbot above its plausible head
    # (kept below stage=14 to stay physical).
    riv_rbot[ny // 2, 6] = 13.2

    with WarpDarcySolver(
        nx=nx, ny=ny, dx=dx, device=_DEVICE, use_riv=True,
        solver_type="kcycle", diag_preconditioner_backend="device",
    ) as solver:
        solver.build_from_fields(
            T_field=T0, R_field=R, active=active, bc_mask=bc, bc_values=bcv,
            riv_mask=riv_mask, riv_stage=riv_stage, riv_rbot=riv_rbot, riv_cond=riv_cond,
        )
        head, info = solver.solve(
            formulation="unconfined", K_field=K, zbot_field=zbot, ztop_field=ztop,
            solver="unconfined_semismooth_newton_kcycle",
            max_levels=2, min_coarse_cells=1, return_info=True,
        )
    assert info.get("converged", False), info.get("newton_failure_reason")

    h_ref = _host_fixed_point(
        K, zbot, ztop, active, bc, bcv, R, dx,
        riv_mask=riv_mask, riv_stage=riv_stage, riv_rbot=riv_rbot, riv_cond=riv_cond,
    )
    assert np.max(np.abs(head - h_ref)) < 1.0e-6
    assert (head[ny // 2, 2:6] > 10.5).all()  # coupled cells
    assert head[ny // 2, 6] < 13.2  # decoupled cell
    assert h_ref[ny // 2, 6] < 13.2


# ---------------------------------------------------------------------------
# 6. Flow budget terms (physics/budgets_2d)
# ---------------------------------------------------------------------------


def test_flow_budget_gated_terms_and_closure():
    """DRN/RIV budget columns carry the gated fluxes and the budget closes."""
    from DARCY_WARP_PACKAGE.physics.budgets_2d import compute_mass_balance_budget
    from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver

    ny, nx, dx, K, zbot, ztop, active, bc, bcv, R, T0 = _small_case_fields()
    drn_mask = np.zeros((ny, nx), dtype=np.int32)
    drn_mask[ny // 2, 2:5] = 1
    drn_elev = np.full((ny, nx), 9.5)
    drn_cond = np.full((ny, nx), 80.0)
    riv_mask = np.zeros((ny, nx), dtype=np.int32)
    riv_mask[ny // 2, 5:7] = 1
    riv_stage = np.full((ny, nx), 14.0)
    riv_rbot = np.full((ny, nx), 10.5)
    riv_cond = np.full((ny, nx), 80.0)

    with WarpDarcySolver(
        nx=nx, ny=ny, dx=dx, device=_DEVICE, use_drn=True, use_riv=True,
        solver_type="kcycle", diag_preconditioner_backend="device",
    ) as solver:
        solver.build_from_fields(
            T_field=T0, R_field=R, active=active, bc_mask=bc, bc_values=bcv,
            drn_mask=drn_mask, drn_elev=drn_elev, drn_cond=drn_cond,
            riv_mask=riv_mask, riv_stage=riv_stage, riv_rbot=riv_rbot, riv_cond=riv_cond,
        )
        head, info = solver.solve(
            formulation="unconfined", K_field=K, zbot_field=zbot, ztop_field=ztop,
            solver="unconfined_semismooth_newton_kcycle",
            max_levels=2, min_coarse_cells=1, return_info=True,
        )
    assert info.get("converged", False), info.get("newton_failure_reason")

    T_conv = K * np.clip(head - zbot, 0.1, ztop - zbot)
    bud = compute_mass_balance_budget(
        T_field=T_conv, R_field=R, head=head, active=active,
        bc_mask=bc, bc_values=bcv, dx=dx,
        drn_mask=drn_mask, drn_elev=drn_elev, drn_cond=drn_cond,
        riv_mask=riv_mask, riv_stage=riv_stage, riv_rbot=riv_rbot, riv_cond=riv_cond,
    )
    row = bud.iloc[0]

    # Gated fluxes match the hand formulas at the converged head.
    q_drn = drn_cond[ny // 2, 2:5] * np.maximum(head[ny // 2, 2:5] - drn_elev[ny // 2, 2:5], 0.0)
    q_riv = riv_cond[ny // 2, 5:7] * (
        np.maximum(head[ny // 2, 5:7], riv_rbot[ny // 2, 5:7]) - riv_stage[ny // 2, 5:7]
    )
    assert float(row["drn_out"]) == pytest.approx(float(np.sum(q_drn)), rel=1e-12)
    assert float(row["drn_in"]) == 0.0
    assert float(row["riv_in"]) == pytest.approx(float(np.sum(np.maximum(-q_riv, 0.0))), rel=1e-12)
    assert float(row["riv_out"]) == pytest.approx(float(np.sum(np.maximum(q_riv, 0.0))), rel=1e-12)
    # A converged solve closes the discrete budget.
    assert abs(float(row["percent_discrepancy"])) < 1.0e-6

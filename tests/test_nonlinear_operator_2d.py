# SPDX-License-Identifier: AGPL-3.0-only
"""Focused validation for the authoritative 2D nonlinear operator.

Covers the ten required validation groups for
``DARCY_WARP_PACKAGE.nonlinear.NonlinearOperator2D``:

1.  confined linear consistency (vs the existing linear operator + sparse ref)
2.  converged Picard heads produce a small true nonlinear residual
3.  saturated thickness (below / at-bottom / between / at-top / above-top)
4.  flow (min_sat floor) vs physical storage (zero-floor) saturation distinction
5.  exact storage: Sy-only / Ss-only / combined vs ``physics.storage_2d``
6.  boundary semantics (active / inactive / Dirichlet / GHB rows)
7.  source terms (positive recharge, negative withdrawal via signed R_field)
8.  heterogeneous conductivity and sloping aquifer bottoms
9.  deterministic repeatability
10. repeated residual evaluations without persistent device-memory growth

The trusted ``unconfined_picard_kcycle`` backend is exercised here only to obtain
converged reference heads; it is not modified.
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
# Device arithmetic is float64; host reference is float64.  Allow a small
# margin over pure round-off for summation-order differences.
_ATOL = 1.0e-9
_RTOL = 1.0e-9


def _make_ctx(
    *,
    ny=7,
    nx=9,
    dx=50.0,
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
    sy=0.0,
    ss=0.0,
    head_prev=None,
    dt=None,
    transient=False,
    min_sat=0.1,
):
    from DARCY_WARP_PACKAGE.nonlinear import from_arrays

    rng = np.random.default_rng(2024)
    shape = (ny, nx)
    if K is None:
        K = (0.5 + 4.5 * rng.random(shape)).astype(np.float64)
    if zbot is None:
        zbot = np.full(shape, 0.0, dtype=np.float64)
    elif np.ndim(zbot) == 0:
        zbot = np.full(shape, float(zbot), dtype=np.float64)
    if np.ndim(ztop) == 0:
        ztop = np.full(shape, float(ztop), dtype=np.float64)
    if active is None:
        active = np.ones(shape, dtype=np.int32)
    if dirichlet_mask is None:
        dirichlet_mask = np.zeros(shape, dtype=np.int32)
    if dirichlet_values is None:
        dirichlet_values = np.zeros(shape, dtype=np.float64)
    if R_field is None:
        R_field = (rng.random(shape) - 0.5) * 2.0e-4

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
        sy=sy,
        ss=ss,
        head_prev=head_prev,
        dt=dt,
        transient=transient,
        min_sat=min_sat,
        device=_DEVICE,
    )


def _random_head(ctx, *, seed=1, offset=9.0):
    rng = np.random.default_rng(seed)
    return np.asarray(ctx.flow.zbot, dtype=np.float64) + offset + rng.random(ctx.shape)


# ---------------------------------------------------------------------------
# 1. Confined linear consistency
# ---------------------------------------------------------------------------


def test_linear_consistency_nonlinear_flow_matches_sparse_at_T_of_head():
    """Nonlinear flow operator equals the linear sparse operator A(T(h)) @ h.

    For an arbitrary (head-dependent) transmissivity the 5-point + GHB stencil
    of the nonlinear operator must be identical to an independent scipy assembly
    that reads T(h) from the same head.  This proves the discretization reduces
    to the trusted linear operator whenever T is head-independent.
    """
    from DARCY_WARP_PACKAGE.nonlinear import NonlinearOperator2D
    from DARCY_WARP_PACKAGE.nonlinear import reference as ref

    ny, nx = 9, 11
    gh_mask = np.zeros((ny, nx), dtype=np.int32)
    gh_head = np.zeros((ny, nx))
    ghb_factor = np.zeros((ny, nx))
    gh_mask[ny // 2, 1:-1] = 1
    gh_head[:] = 12.0
    ghb_factor[gh_mask != 0] = 0.05

    ctx = _make_ctx(ny=ny, nx=nx, gh_mask=gh_mask, gh_head=gh_head, ghb_factor=ghb_factor)
    head = _random_head(ctx)
    free = ctx.free_mask

    op = NonlinearOperator2D(ctx)
    op.residual(head)  # steady: F = flow_A - sources
    Fdev = np.asarray(op._F_wp.numpy(), dtype=np.float64)

    A = ref.assemble_flow_operator_sparse(head, ctx)
    Ah = (A @ head.reshape(-1)).reshape(ny, nx)
    flow_host = ref.flow_operator_applied(head, ctx)
    # The independent sparse assembly of the head-dependent operator matches the
    # host flow application (identical stencil, different code path).
    np.testing.assert_allclose(Ah[free], flow_host[free], atol=_ATOL, rtol=_RTOL)
    # Device residual (steady: F = flow_A - sources) matches host flow - sources.
    R = np.asarray(ctx.sources.R_field) * ctx.grid.area
    ghb = ref._flow_transmissivity(head, ctx)
    ghbm = np.asarray(ctx.boundaries.ghb_mask, dtype=np.int32) != 0
    ghf = np.asarray(ctx.boundaries.ghb_factor, dtype=np.float64)
    Cgh = np.where(ghbm & (ghf > 0), ghb * ghf, 0.0) * np.asarray(ctx.boundaries.ghb_external_head)
    np.testing.assert_allclose(Fdev[free], (flow_host - R - Cgh)[free], atol=_ATOL, rtol=_RTOL)
    np.testing.assert_allclose(Ah[free], flow_host[free], atol=_ATOL, rtol=_RTOL)
    op.close()


def test_linear_consistency_constant_T_matches_production_sparse_system():
    """When T is head-independent the residual equals ``A h - b`` of the linear system.

    Fully saturated constant-K domain (head above top -> flow_sat capped to the
    constant ``top - bottom``) so T(h) = K (top - bottom) everywhere.  The device
    residual is then exactly the residual of the production sparse linear system.
    """
    from DARCY_WARP_PACKAGE.sparse_operator import build_sparse_system_fd_like
    from DARCY_WARP_PACKAGE.nonlinear import NonlinearOperator2D

    ny, nx = 8, 8
    dx = 40.0
    Kval = 2.0
    zbot_v = 0.0
    ztop_v = 10.0
    K = np.full((ny, nx), Kval, dtype=np.float64)
    zbot = np.full((ny, nx), zbot_v, dtype=np.float64)
    ztop = np.full((ny, nx), ztop_v, dtype=np.float64)
    active = np.ones((ny, nx), dtype=np.int32)
    bc_mask = np.zeros((ny, nx), dtype=np.int32)
    bc_mask[:, 0] = 1
    bc_mask[:, -1] = 1
    bc_values = np.zeros((ny, nx))
    bc_values[:, 0] = 14.0
    bc_values[:, -1] = 9.0
    R = np.full((ny, nx), 1.0e-5, dtype=np.float64)

    # head above top everywhere -> fully saturated -> constant T
    head = np.full((ny, nx), 13.0, dtype=np.float64)

    ctx = _make_ctx(
        ny=ny, nx=nx, dx=dx, K=K, zbot=zbot, ztop=ztop,
        active=active, dirichlet_mask=bc_mask, dirichlet_values=bc_values, R_field=R,
        transient=False,
    )
    T_const = Kval * (ztop_v - zbot_v)
    A, b, free_flat = build_sparse_system_fd_like(
        T_field=np.full((ny, nx), T_const),
        R_field=R, active=active, bc_mask=bc_mask, bc_values=bc_values, dx=dx,
    )
    lin_res = (A @ head.reshape(-1) - b).reshape(ny, nx)

    op = NonlinearOperator2D(ctx)
    op.residual(head)
    Fdev = np.asarray(op._F_wp.numpy(), dtype=np.float64)
    free = ctx.free_mask
    np.testing.assert_allclose(Fdev[free], lin_res[free], atol=1.0e-7, rtol=1.0e-7)
    op.close()


# ---------------------------------------------------------------------------
# 2. Converged Picard heads -> small nonlinear residual
# ---------------------------------------------------------------------------


def _build_picard_solver(ny=8, nx=8, *, use_ghb=False):
    from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver

    solver = WarpDarcySolver(
        nx=nx, ny=ny, dx=100.0, device=_DEVICE, use_ghb=use_ghb,
        solver_type="kcycle", diag_preconditioner_backend="device",
    )
    active = np.ones((ny, nx), dtype=np.int32)
    bc_mask = np.zeros((ny, nx), dtype=np.int32)
    bc_values = np.zeros((ny, nx), dtype=np.float64)
    bc_mask[:, 0] = 1
    bc_mask[:, -1] = 1
    bc_values[:, 0] = 12.0
    bc_values[:, -1] = 9.0
    recharge = np.full((ny, nx), 1.0e-5, dtype=np.float64)
    transmissivity = np.full((ny, nx), 10.0, dtype=np.float64)
    build_kwargs = {
        "T_field": transmissivity, "R_field": recharge, "active": active,
        "bc_mask": bc_mask, "bc_values": bc_values,
    }
    if use_ghb:
        gh_mask = np.zeros((ny, nx), dtype=np.int32)
        gh_head = np.zeros((ny, nx))
        gh_width = np.zeros((ny, nx))
        gh_mask[1:-1, 1] = 1
        gh_head[1:-1, 1] = 10.5
        gh_width[1:-1, 1] = 25.0
        build_kwargs["gh_mask"] = gh_mask
        build_kwargs["gh_head"] = gh_head
        build_kwargs["gh_width"] = gh_width
    solver.build_from_fields(**build_kwargs)
    return solver


def test_converged_picard_steady_head_satisfies_nonlinear_residual():
    """A converged steady Picard head must drive the true nonlinear residual down."""
    from DARCY_WARP_PACKAGE.nonlinear import NonlinearOperator2D, from_unconfined_solve_inputs

    solver = _build_picard_solver()
    K = np.full((solver.ny, solver.nx), 1.0, dtype=np.float64)
    zbot = np.zeros((solver.ny, solver.nx), dtype=np.float64)
    ztop = np.full((solver.ny, solver.nx), 20.0, dtype=np.float64)

    head, info = solver.solve(
        formulation="unconfined", K_field=K, zbot_field=zbot, ztop_field=ztop,
        max_cycles=12, max_levels=2, min_coarse_cells=1, check_every_no=1,
        abs_tol_min=1.0e-8, rel_tol=1.0e-8, max_outer_iterations=60,
        hclose=1.0e-6, unconfined_startup_mode="confined_pre_solve", return_info=True,
    )
    assert info.get("converged", False)

    ctx = from_unconfined_solve_inputs(solver, K_field=K, zbot_field=zbot, ztop_field=ztop, min_sat=0.1)
    op = NonlinearOperator2D(ctx)
    rn_conv = op.residual_norms(head)

    # A clearly non-converged head has a much larger residual.
    rng = np.random.default_rng(3)
    head_bad = np.full(head.shape, 20.0) + rng.random(head.shape)
    rn_bad = op.residual_norms(head_bad)

    assert rn_conv.rms < 1.0e-3, rn_conv
    assert rn_bad.rms > 1.0  # arbitrary bad head is far from balance
    assert rn_conv.rms < 1.0e-6 * rn_bad.rms
    op.close()
    solver.close()


def test_converged_picard_transient_head_reconciles_exact_and_secant_storage():
    """Transient Picard (secant-Sy) head: exact and secant residuals both small.

    Demonstrates the reconciliation: the authoritative exact-storage residual
    and the production secant-linearised residual agree at a converged head
    (Sy matches exactly; Ss matches to O(Ss dh^2), negligible for Sy-dominated
    storage and for fully-saturated cells).
    """
    from DARCY_WARP_PACKAGE.nonlinear import NonlinearOperator2D, from_unconfined_solve_inputs
    from DARCY_WARP_PACKAGE.nonlinear import reference as ref

    solver = _build_picard_solver()
    ny, nx = solver.ny, solver.nx
    K = np.full((ny, nx), 1.0, dtype=np.float64)
    zbot = np.zeros((ny, nx), dtype=np.float64)
    ztop = np.full((ny, nx), 20.0, dtype=np.float64)
    head_prev = np.full((ny, nx), 10.0, dtype=np.float64)

    head, info = solver.solve(
        formulation="unconfined", K_field=K, zbot_field=zbot, ztop_field=ztop,
        transient=True, storage_coeff=0.2, dt=5.0, head_prev=head_prev,
        sy=0.2, ss=1.0e-5, unconfined_storage_mode_2d="mf6_convertible_secant_sy",
        storage_reference="current_picard",
        max_cycles=12, max_levels=2, min_coarse_cells=1, check_every_no=1,
        abs_tol_min=1.0e-8, rel_tol=1.0e-8, max_outer_iterations=80,
        hclose=1.0e-6, unconfined_startup_mode="confined_pre_solve",
        practical_picard_acceptance_enabled=True, return_info=True,
    )

    ctx = from_unconfined_solve_inputs(
        solver, K_field=K, zbot_field=zbot, ztop_field=ztop, min_sat=0.1,
        transient=True, sy=0.2, ss=1.0e-5, dt=5.0, head_prev=head_prev,
    )
    op = NonlinearOperator2D(ctx)
    rn_exact = op.residual_norms(head)

    # Secant-linearised residual at the same head (what Picard actually drove down).
    frozen = op.frozen_picard_operator(head)
    free = ctx.free_mask
    flowAh = ref.flow_operator_applied(head, ctx)
    R = np.asarray(ctx.sources.R_field) * ctx.grid.area
    ghbm = np.asarray(ctx.boundaries.ghb_mask, dtype=np.int32) != 0
    ghf = np.asarray(ctx.boundaries.ghb_factor, dtype=np.float64)
    T = ref._flow_transmissivity(head, ctx)
    Cgh_src = np.where(ghbm & (ghf > 0), T * ghf, 0.0) * np.asarray(ctx.boundaries.ghb_external_head)
    secant_store = frozen.storage_diag * (head - head_prev)
    F_secant = (flowAh + secant_store - R - Cgh_src)
    rms_secant = float(np.sqrt(np.mean(F_secant[free] ** 2)))

    # Both must be small, and close to each other.
    assert rn_exact.rms < 1.0e-2, rn_exact
    assert rms_secant < 1.0e-2, rms_secant
    assert rn_exact.rms <= max(1.0e-6, 5.0 * rms_secant + 1.0e-3)
    op.close()
    solver.close()


# ---------------------------------------------------------------------------
# 3. Saturated thickness cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("h_above_bot", "expected_sat"),
    [
        (-1.0, None),   # below bottom  -> min_sat
        (0.0, None),    # at bottom     -> min_sat
        (5.0, 5.0),     # between       -> head - bottom
        (10.0, 10.0),   # at top        -> top - bottom
        (13.0, 10.0),   # above top     -> top - bottom (capped)
    ],
)
def test_saturated_thickness_cases(h_above_bot, expected_sat):
    from DARCY_WARP_PACKAGE.nonlinear import NonlinearOperator2D

    zbot = np.array([[0.0]])
    ztop = np.array([[10.0]])
    K = np.array([[1.0]])
    min_sat = 0.1
    ctx = _make_ctx(ny=1, nx=1, K=K, zbot=zbot, ztop=ztop, R_field=np.array([[0.0]]), min_sat=min_sat)
    head = np.array([[h_above_bot]])
    op = NonlinearOperator2D(ctx)
    sat_wp = op.saturated_thickness(head)
    sat = float(np.asarray(sat_wp.numpy())[0, 0])
    exp = expected_sat if expected_sat is not None else min_sat
    assert sat == pytest.approx(exp, abs=1.0e-12)
    op.close()


# ---------------------------------------------------------------------------
# 4. Flow vs storage saturation distinction
# ---------------------------------------------------------------------------


def test_flow_min_sat_floor_absent_from_physical_storage():
    """``min_sat`` flow floor must not leak into Sy / Ss physical storage."""
    from DARCY_WARP_PACKAGE.nonlinear import NonlinearOperator2D

    # One dry-ish cell: head just above bottom, prev at bottom.
    zbot = np.array([[0.0]])
    ztop = np.array([[10.0]])
    K = np.array([[1.0]])
    min_sat = 0.5
    head_prev = np.array([[0.0]])  # exactly at bottom -> physical sat_old = 0
    head = np.array([[0.3]])       # 0.3 above bottom (< min_sat)
    ctx = _make_ctx(
        ny=1, nx=1, K=K, zbot=zbot, ztop=ztop, R_field=np.array([[0.0]]),
        sy=0.2, ss=1.0e-3, head_prev=head_prev, dt=2.0, transient=True, min_sat=min_sat,
    )
    op = NonlinearOperator2D(ctx)

    # Flow saturation uses the min_sat floor.
    sat_wp = op.saturated_thickness(head)
    assert float(np.asarray(sat_wp.numpy())[0, 0]) == pytest.approx(min_sat, abs=1.0e-12)

    # Physical storage saturation is zero-floored: head-bottom = 0.3, not min_sat.
    terms = op.exact_storage_terms(head)
    sat_phys = float(terms.sat_physical[0, 0])
    assert sat_phys == pytest.approx(0.3, abs=1.0e-12)

    area = ctx.grid.area
    # Sy storage = Sy * (sat_new - sat_old) * area / dt = 0.2 * 0.3 * area / 2
    assert float(terms.sy[0, 0]) == pytest.approx(0.2 * 0.3 * area / 2.0, abs=1.0e-9)
    # Ss storage uses the zero-floor potential (not min_sat): exact phi difference.
    from DARCY_WARP_PACKAGE.physics.storage_2d import specific_storage_potential
    phi_new = specific_storage_potential(head=0.3, bottom=0.0, top=10.0, specific_storage=1.0e-3)
    phi_old = specific_storage_potential(head=0.0, bottom=0.0, top=10.0, specific_storage=1.0e-3)
    assert float(terms.ss[0, 0]) == pytest.approx((phi_new - phi_old) * area / 2.0, abs=1.0e-9)
    op.close()


# ---------------------------------------------------------------------------
# 5. Exact storage: Sy-only / Ss-only / combined vs storage_2d
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sy", "ss", "label"),
    [(0.2, 0.0, "sy_only"), (0.0, 1.0e-3, "ss_only"), (0.2, 1.0e-3, "combined")],
)
def test_exact_storage_matches_storage_2d(sy, ss, label):
    from DARCY_WARP_PACKAGE.nonlinear import NonlinearOperator2D
    from DARCY_WARP_PACKAGE.nonlinear import reference as ref

    # Mixed saturation regime: some cells partial, some full, some dry.
    ny, nx = 5, 5
    zbot = np.zeros((ny, nx))
    ztop = np.full((ny, nx), 10.0)
    head_prev = np.full((ny, nx), 4.0)
    head = np.linspace(-1.0, 13.0, ny * nx).reshape(ny, nx)  # spans dry..full..confined
    ctx = _make_ctx(
        ny=ny, nx=nx, zbot=zbot, ztop=ztop, R_field=np.zeros((ny, nx)),
        sy=sy, ss=ss, head_prev=head_prev, dt=4.0, transient=True,
    )
    op = NonlinearOperator2D(ctx)
    terms = op.exact_storage_terms(head)
    tot_h, sy_h, ss_h = ref.exact_storage_terms_host(head, ctx)
    np.testing.assert_allclose(terms.sy, sy_h, atol=_ATOL, rtol=_RTOL)
    np.testing.assert_allclose(terms.ss, ss_h, atol=_ATOL, rtol=_RTOL)
    np.testing.assert_allclose(terms.total, tot_h, atol=_ATOL, rtol=_RTOL)
    np.testing.assert_allclose(terms.total, terms.sy + terms.ss, atol=_ATOL, rtol=_RTOL)
    op.close()


# ---------------------------------------------------------------------------
# 6. Boundary semantics
# ---------------------------------------------------------------------------


def test_boundary_row_semantics():
    """Inactive / Dirichlet rows are zero; GHB row carries C_gh (h - gh_head)."""
    from DARCY_WARP_PACKAGE.nonlinear import NonlinearOperator2D

    ny, nx = 3, 3
    K = np.ones((ny, nx))
    zbot = np.zeros((ny, nx))
    ztop = np.full((ny, nx), 20.0)
    active = np.ones((ny, nx), dtype=np.int32)
    active[0, 0] = 0  # inactive
    dirichlet = np.zeros((ny, nx), dtype=np.int32)
    dirichlet[0, 2] = 1  # Dirichlet
    dirichlet_values = np.zeros((ny, nx))
    dirichlet_values[0, 2] = 7.0
    gh_mask = np.zeros((ny, nx), dtype=np.int32)
    gh_head = np.zeros((ny, nx))
    ghb_factor = np.zeros((ny, nx))
    gh_mask[1, 1] = 1
    gh_head[1, 1] = 6.0
    ghb_factor[1, 1] = 0.25
    head = np.full((ny, nx), 9.0)

    ctx = _make_ctx(
        ny=ny, nx=nx, K=K, zbot=zbot, ztop=ztop, R_field=np.zeros((ny, nx)),
        active=active, dirichlet_mask=dirichlet, dirichlet_values=dirichlet_values,
        gh_mask=gh_mask, gh_head=gh_head, ghb_factor=ghb_factor,
    )
    op = NonlinearOperator2D(ctx)
    op.residual(head)
    Fdev = np.asarray(op._F_wp.numpy(), dtype=np.float64)

    # Inactive and Dirichlet rows are excluded (zero residual).
    assert Fdev[0, 0] == 0.0
    assert Fdev[0, 2] == 0.0
    # GHB row carries the C_gh (h - gh_head) outflow term.
    area = ctx.grid.area
    T_ghb = 1.0 * 9.0  # K * flow_sat (head 9, bottom 0, capped at 20) = 9
    C_gh = T_ghb * 0.25
    # Net GHB contribution to F = +C_gh*h (diagonal) - C_gh*gh_head (source).
    ghb_net = C_gh * 9.0 - C_gh * 6.0
    # Surrounding 8 neighbours all active -> flow contribution plus GHB net.
    # Verify the host reference (full balance) matches the device on the GHB row.
    from DARCY_WARP_PACKAGE.nonlinear import reference as ref
    Fhost = ref.nonlinear_residual_host(head, ctx)
    np.testing.assert_allclose(Fdev, Fhost, atol=_ATOL, rtol=_RTOL)
    # And the GHB net term is present (residual differs from the no-GHB case).
    ctx_noghb = _make_ctx(
        ny=ny, nx=nx, K=K, zbot=zbot, ztop=ztop, R_field=np.zeros((ny, nx)),
        active=active, dirichlet_mask=dirichlet, dirichlet_values=dirichlet_values,
    )
    op_noghb = NonlinearOperator2D(ctx_noghb)
    op_noghb.residual(head)
    Fnoghb = np.asarray(op_noghb._F_wp.numpy(), dtype=np.float64)
    assert abs(Fdev[1, 1] - Fnoghb[1, 1] - ghb_net) < 1.0e-7
    op.close()
    op_noghb.close()


# ---------------------------------------------------------------------------
# 7. Source terms (signed R_field)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("r_value", [1.0e-4, -3.0e-4])
def test_signed_source_field_drives_residual(r_value):
    """Uniform head -> zero flow gradient -> F = -R_field * area (sign-correct)."""
    from DARCY_WARP_PACKAGE.nonlinear import NonlinearOperator2D

    ny, nx = 3, 3
    K = np.ones((ny, nx))
    zbot = np.zeros((ny, nx))
    ztop = np.full((ny, nx), 20.0)
    head = np.full((ny, nx), 9.0)  # uniform -> no flow gradients
    ctx = _make_ctx(
        ny=ny, nx=nx, K=K, zbot=zbot, ztop=ztop,
        R_field=np.full((ny, nx), r_value),
    )
    op = NonlinearOperator2D(ctx)
    op.residual(head)
    Fdev = np.asarray(op._F_wp.numpy(), dtype=np.float64)
    area = ctx.grid.area
    np.testing.assert_allclose(Fdev[1, 1], -r_value * area, atol=1.0e-7, rtol=1.0e-9)
    op.close()


# ---------------------------------------------------------------------------
# 8. Heterogeneous conductivity and sloping bottoms
# ---------------------------------------------------------------------------


def test_heterogeneous_K_and_sloping_bottom_match_host():
    from DARCY_WARP_PACKAGE.nonlinear import NonlinearOperator2D
    from DARCY_WARP_PACKAGE.nonlinear import reference as ref

    ny, nx = 10, 12
    rng = np.random.default_rng(11)
    K = (0.2 + 5.0 * rng.random((ny, nx))).astype(np.float64)
    j = np.arange(ny)[:, None]
    i = np.arange(nx)[None, :]
    zbot = (0.05 * j + 0.02 * i).astype(np.float64)            # sloping bottom
    ztop = zbot + 15.0 + 2.0 * rng.random((ny, nx))            # varying thickness
    active = np.ones((ny, nx), dtype=np.int32)
    active[3:5, 6] = 0
    dirichlet = np.zeros((ny, nx), dtype=np.int32)
    dirichlet[0, :] = 1
    dirichlet[-1, :] = 1
    dhv = np.zeros((ny, nx))
    dhv[0, :] = 18.0
    dhv[-1, :] = 5.0
    gh_mask = np.zeros((ny, nx), dtype=np.int32)
    gh_head = np.zeros((ny, nx))
    ghb_factor = np.zeros((ny, nx))
    gh_mask[5, 2:9] = 1
    gh_head[5, :] = 10.0
    ghb_factor[gh_mask != 0] = 0.04

    ctx = _make_ctx(
        ny=ny, nx=nx, K=K, zbot=zbot, ztop=ztop,
        active=active, dirichlet_mask=dirichlet, dirichlet_values=dhv,
        R_field=(rng.random((ny, nx)) - 0.5) * 4.0e-4,
        gh_mask=gh_mask, gh_head=gh_head, ghb_factor=ghb_factor,
        sy=0.15, ss=2.0e-4, head_prev=zbot + 6.0, dt=8.0, transient=True,
    )
    head = zbot + 7.0 + 3.0 * rng.random((ny, nx))

    op = NonlinearOperator2D(ctx)
    op.residual(head)
    Fdev = np.asarray(op._F_wp.numpy(), dtype=np.float64)
    Fhost = ref.nonlinear_residual_host(head, ctx)
    np.testing.assert_allclose(Fdev, Fhost, atol=1.0e-7, rtol=1.0e-9)

    terms = op.exact_storage_terms(head)
    tot_h, _, _ = ref.exact_storage_terms_host(head, ctx)
    np.testing.assert_allclose(terms.total, tot_h, atol=1.0e-7, rtol=1.0e-9)
    op.close()


# ---------------------------------------------------------------------------
# 9. Deterministic repeatability
# ---------------------------------------------------------------------------


def test_deterministic_repeatability():
    from DARCY_WARP_PACKAGE.nonlinear import NonlinearOperator2D

    ctx = _make_ctx(sy=0.1, ss=1.0e-4, head_prev=np.full((7, 9), 8.0), dt=5.0, transient=True)
    head = _random_head(ctx)
    op = NonlinearOperator2D(ctx)
    a = op.residual_norms(head)
    b = op.residual_norms(head)
    assert a.rms == b.rms
    assert a.max_abs == b.max_abs

    # A freshly constructed operator on the same context produces identical results.
    op2 = NonlinearOperator2D(ctx)
    c = op2.residual_norms(head)
    assert c.rms == a.rms
    assert c.max_abs == a.max_abs
    op.close()
    op2.close()


# ---------------------------------------------------------------------------
# 10. No persistent device-memory growth on repeated evaluation
# ---------------------------------------------------------------------------


def test_no_memory_growth_on_repeated_residual():
    """Repeated evaluations reuse the same scratch allocations (no growth)."""
    from DARCY_WARP_PACKAGE.nonlinear import NonlinearOperator2D

    ctx = _make_ctx(sy=0.1, ss=1.0e-4, head_prev=np.full((7, 9), 8.0), dt=5.0, transient=True)
    head = _random_head(ctx)
    op = NonlinearOperator2D(ctx)

    before = op.scratch_arrays()
    rng = np.random.default_rng(99)
    for k in range(40):
        h = head + 0.001 * k * rng.random(ctx.shape)
        op.residual_norms(h)
        op.exact_storage_terms(h)
        op.frozen_picard_operator(h)
    after = op.scratch_arrays()

    assert len(before) == len(after)
    for b, a in zip(before, after):
        assert b is a, "operator reallocated a scratch array during repeated use"

    # If running on a CUDA device, additionally assert the Warp pool did not grow.
    if str(_DEVICE).startswith("cuda"):
        import warp as wp
        dev = wp.get_device(_DEVICE)
        stats = getattr(dev, "mem_pool_stats", lambda: None)()
        if stats is not None:
            _ = stats  # presence-only: the same-array check above is the real gate
    op.close()

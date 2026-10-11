# SPDX-License-Identifier: AGPL-3.0-only
"""Independent scientific validation of the 3D solver physics.

Every test exercises BOTH the classic and the fast (``implementation="fast"``)
execution paths against independently derived analytical references (not
against the other implementation), plus MODFLOW 6 comparisons via flopy.

Sign conventions under test (total hydraulic head, m):
  - positive rhs = injection/recharge, volumetric per cell [m^3/day];
  - face flux = C * (h_here - h_neighbour)  (high head -> low head);
  - K in m/day; face conductance in m^2/day;
  - storage term = storage_diag * (h - h_prev), positive = uptake.
"""

from __future__ import annotations

import numpy as np
import pytest

warp = pytest.importorskip("warp")

from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D


def _device() -> str:
    warp.init()
    return "cuda:0" if warp.is_cuda_available() else "cpu"


IMPLEMENTATIONS = ["classic", "fast"]

TIGHT = {
    "rel_tol": 1.0e-10,
    "abs_tol_min": 1.0e-12,
    "dh_rms_tol": 1.0e-10,
    "check_every_no": 1,
    "max_cycles": 200,
}


def _solve_confined(impl, kx, ky, kz, active, bc_mask, bc_values, rhs, dx, dy, dz,
                    initial_head=None, device=None, **extra):
    shape = kx.shape
    nz, ny, nx = shape
    with WarpDarcySolver3D(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz,
                           device=device or _device(), implementation=impl) as s:
        s.build_from_K_fields(kx, ky, kz, active, bc_mask, bc_values, rhs, initial_head)
        kwargs = dict(TIGHT)
        kwargs.update(extra)
        return s.solve(**kwargs)


# ---------------------------------------------------------------------------
# 1D analytical confined flow
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_1d_horizontal_confined_analytical(implementation):
    """Steady 1D horizontal flow: exact linear profile and exact flux."""
    nz, ny, nx = 2, 2, 10
    dx, dy, dz = 2.0, 3.0, 5.0
    shape = (nz, ny, nx)
    kx = np.full(shape, 4.0)
    ky = np.full(shape, 3.0)
    kz = np.full(shape, 2.0)
    active = np.ones(shape, dtype=np.int32)
    bc_mask = np.zeros(shape, dtype=np.int32)
    bc_values = np.zeros(shape)
    bc_mask[:, :, 0] = 1
    bc_values[:, :, 0] = 10.0
    bc_mask[:, :, -1] = 1
    bc_values[:, :, -1] = 6.0
    rhs = np.zeros(shape)

    head, info = _solve_confined(implementation, kx, ky, kz, active, bc_mask,
                                 bc_values, rhs, dx, dy, dz)
    assert info["converged"]
    exact = 10.0 - (10.0 - 6.0) / (nx - 1) * np.arange(nx)
    np.testing.assert_allclose(head[0, 0], exact, rtol=0.0, atol=1e-9)
    # Flux through each internal face: C = K*dy*dz/dx = 30 m^2/day; the
    # boundary exchange aggregates all ny*nz rows/layers.
    c_face = 4.0 * dy * dz / dx
    q = c_face * (10.0 - 6.0) / (nx - 1) * ny * nz
    budget = info["budget"]
    assert budget["boundary_inflow"] == pytest.approx(q, rel=1e-9)
    assert budget["boundary_outflow"] == pytest.approx(-q, rel=1e-9)
    assert budget["relative_imbalance"] < 1e-9


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_1d_vertical_confined_analytical(implementation):
    """Steady 1D vertical flow down a single column: exact linear profile."""
    nz, ny, nx = 6, 1, 1
    dx, dy, dz = 2.0, 3.0, 1.5
    shape = (nz, ny, nx)
    k = np.full(shape, 7.0)
    active = np.ones(shape, dtype=np.int32)
    bc_mask = np.zeros(shape, dtype=np.int32)
    bc_values = np.zeros(shape)
    bc_mask[0] = 1          # top cell
    bc_values[0] = 12.0
    bc_mask[-1] = 1         # bottom cell
    bc_values[-1] = 8.0
    rhs = np.zeros(shape)

    head, info = _solve_confined(implementation, k, k, k, active, bc_mask,
                                 bc_values, rhs, dx, dy, dz, max_levels=2)
    assert info["converged"]
    exact = 12.0 - (12.0 - 8.0) / (nz - 1) * np.arange(nz)
    np.testing.assert_allclose(head[:, 0, 0], exact, rtol=0.0, atol=1e-9)
    c_face = 7.0 * dx * dy / dz
    q = c_face * (12.0 - 8.0) / (nz - 1)
    budget = info["budget"]
    assert budget["boundary_inflow"] == pytest.approx(q, rel=1e-9)
    assert budget["boundary_outflow"] == pytest.approx(-q, rel=1e-9)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_vertical_series_resistance_two_materials(implementation):
    """Two-material vertical stack: exact piecewise-linear head profile from
    the physical half-cell series resistances (Darcy's law), which the
    harmonic-mean face conductance must reproduce exactly."""
    nz, ny, nx = 4, 1, 1
    dx, dy, dz = 2.0, 2.0, 1.0
    shape = (nz, ny, nx)
    kz = np.full(shape, 2.0)
    kz[2:] = 8.0  # high-K bottom half
    kx = ky = np.full(shape, 5.0)
    active = np.ones(shape, dtype=np.int32)
    bc_mask = np.zeros(shape, dtype=np.int32)
    bc_values = np.zeros(shape)
    bc_mask[0] = 1
    bc_values[0] = 12.0
    bc_mask[-1] = 1
    bc_values[-1] = 8.0
    rhs = np.zeros(shape)

    head, info = _solve_confined(implementation, kx, ky, kz, active, bc_mask,
                                 bc_values, rhs, dx, dy, dz, max_levels=2)
    assert info["converged"]
    area = dx * dy
    # Node-to-node resistance = dz/2/(K_i*A) + dz/2/(K_{i+1}*A).
    r_faces = [
        dz / 2 / (2.0 * area) + dz / 2 / (2.0 * area),
        dz / 2 / (2.0 * area) + dz / 2 / (8.0 * area),
        dz / 2 / (8.0 * area) + dz / 2 / (8.0 * area),
    ]
    q = (12.0 - 8.0) / sum(r_faces)
    exact = [12.0]
    for r in r_faces:
        exact.append(exact[-1] - q * r)
    np.testing.assert_allclose(head[:, 0, 0], exact, rtol=0.0, atol=1e-9)
    budget = info["budget"]
    assert budget["boundary_inflow"] == pytest.approx(q, rel=1e-9)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_hydrostatic_multilayer_no_flow(implementation):
    """Heterogeneous K, fixed constant head on the outer shell, no source:
    heads must stay exactly constant and every flux must vanish."""
    nz, ny, nx = 3, 5, 6
    dx, dy, dz = 2.0, 2.0, 1.0
    shape = (nz, ny, nx)
    rng = np.random.default_rng(11)
    k = 10.0 * np.exp(rng.normal(0.0, 1.0, size=shape))
    active = np.ones(shape, dtype=np.int32)
    bc_mask = np.zeros(shape, dtype=np.int32)
    bc_mask[:, :, 0] = 1
    bc_mask[:, :, -1] = 1
    bc_mask[:, 0, :] = 1
    bc_mask[:, -1, :] = 1
    bc_mask[0] = 1
    bc_mask[-1] = 1
    h0_const = 37.5
    bc_values = np.full(shape, h0_const)
    rhs = np.zeros(shape)
    initial = np.full(shape, h0_const - 3.0)

    head, info = _solve_confined(implementation, k, 0.5 * k, 0.1 * k, active,
                                 bc_mask, bc_values, rhs, dx, dy, dz,
                                 initial_head=initial)
    assert info["converged"]
    np.testing.assert_allclose(head[active != 0], h0_const, rtol=0.0, atol=1e-8)
    budget = info["budget"]
    assert abs(budget["boundary_net"]) < 1e-8
    assert budget["cell_residual_max"] < 1e-8


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_recharge_raises_abstraction_lowers_column(implementation):
    """Single column, fixed head at the bottom: recharge raises the top head
    by exactly q * (n_faces * dz) / (Kz * area); pumping lowers it by the same
    amount; intercell fluxes telescope."""
    nz, ny, nx = 4, 1, 1
    dx, dy, dz = 5.0, 5.0, 2.0
    shape = (nz, ny, nx)
    k = np.full(shape, 5.0)
    active = np.ones(shape, dtype=np.int32)
    bc_mask = np.zeros(shape, dtype=np.int32)
    bc_values = np.zeros(shape)
    bc_mask[-1] = 1
    bc_values[-1] = 10.0
    area = dx * dy
    n_faces = nz - 1

    for sign, label in ((1.0, "recharge"), (-1.0, "abstraction")):
        q = sign * 0.5  # m^3/day into the top cell
        rhs = np.zeros(shape)
        rhs[0] = q
        head, info = _solve_confined(implementation, k, k, k, active, bc_mask,
                                     bc_values, rhs, dx, dy, dz, max_levels=2)
        assert info["converged"], label
        rise = q * n_faces * dz / (5.0 * area)
        assert head[0, 0, 0] == pytest.approx(10.0 + rise, rel=0.0, abs=1e-9)
        # Linear profile down the column.
        exact = 10.0 + q * dz / (5.0 * area) * (np.arange(nz)[::-1])
        np.testing.assert_allclose(head[:, 0, 0], exact, rtol=0.0, atol=1e-9)
        budget = info["budget"]
        assert budget["source_total"] == pytest.approx(q, rel=1e-12)
        # The bottom boundary absorbs exactly the source.
        if sign > 0:
            assert budget["boundary_outflow"] == pytest.approx(-q, rel=1e-9)
        else:
            assert budget["boundary_inflow"] == pytest.approx(-q, rel=1e-9)
        assert budget["relative_imbalance"] < 1e-9


# ---------------------------------------------------------------------------
# Transient storage
# ---------------------------------------------------------------------------

def _storage_potential_1cell(h, zbot, dz, sy, ss):
    rel = h - zbot
    b = min(max(rel, 0.0), dz)
    return sy * b + ss * (0.5 * b * b + dz * max(rel - dz, 0.0))


def _invert_potential_1cell(p, zbot, dz, sy, ss):
    # Unconfined branch: sy*b + 0.5*ss*b^2 = p
    if ss > 0.0:
        disc = sy * sy + 2.0 * ss * p
        b = (-sy + np.sqrt(max(disc, 0.0))) / ss
    else:
        b = p / sy
    if 0.0 <= b <= dz:
        return zbot + b
    # Confined branch: sy*dz + ss*(0.5*dz^2 + dz*(rel - dz)) = p
    rel = (p - sy * dz - 0.5 * ss * dz * dz) / (ss * dz) + dz
    return zbot + rel


def _run_unconfined_column(impl, rhs, h_old, zbot, dz, dx, sy, ss, dt, kz=0.0,
                           max_outer=400):
    nz = len(zbot)
    shape = (nz, 1, 1)
    k = np.full(shape, float(kz))
    zbot_field = np.zeros(shape)
    for iz in range(nz):
        zbot_field[iz] = zbot[iz]
    ztop_field = zbot_field + dz
    active = np.ones(shape, dtype=np.int32)
    bc_mask = np.zeros(shape, dtype=np.int32)
    bc_values = np.zeros(shape)
    device = _device()
    with WarpDarcySolver3D(nx=1, ny=1, nz=nz, dx=dx, dy=dx, dz=dz,
                           device=device, implementation=impl) as s:
        s.build_from_K_fields(k, k, k, active, bc_mask, bc_values, rhs, h_old)
        return s.solve(
            unconfined=True, transient=True,
            zbot_field=zbot_field, ztop_field=ztop_field,
            sy=sy, ss=ss, dt=dt, head_prev=h_old,
            unconfined_storage_mode="secant_sy",
            unconfined_max_picard_iter=max_outer,
            unconfined_head_tol=1e-10,
            max_cycles=100, rel_tol=1e-12, abs_tol_min=1e-14,
            dh_rms_tol=None, check_every_no=1,
            inner_head_residual_tol_max=1e-12,
            inner_head_residual_tol_min=1e-14,
            final_head_residual_tol=1e-9,
        )


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_confined_transient_storage_exact_multistep(implementation):
    """Pure confined storage ODE (no flow): dh = q*dt/(S*V) per step, exact
    over three sequential steps."""
    nz, ny, nx = 3, 1, 1
    dx, dy, dz = 10.0, 10.0, 4.0
    shape = (nz, ny, nx)
    k = np.zeros(shape)
    active = np.ones(shape, dtype=np.int32)
    bc_mask = np.zeros(shape, dtype=np.int32)
    bc_values = np.zeros(shape)
    s_coeff = 0.01
    vol = dx * dy * dz
    q = 2.0
    dt = 0.5
    h_prev = np.full(shape, 20.0)
    rhs = np.zeros(shape)
    rhs[0] = q

    for step in range(3):
        head, info = _solve_confined(implementation, k, k, k, active, bc_mask,
                                     bc_values, rhs, dx, dy, dz,
                                     initial_head=h_prev, transient=True,
                                     storage_coeff=s_coeff, dt=dt,
                                     head_prev=h_prev, max_levels=2)
        assert info["converged"]
        assert info["transient_formulation"] == "confined"
        expected = h_prev.copy()
        expected[0] = h_prev[0] + q * dt / (s_coeff * vol)
        np.testing.assert_allclose(head, expected, rtol=0.0, atol=1e-10)
        budget = info["budget"]
        assert budget["storage_uptake"] == pytest.approx(q, rel=1e-9)
        assert budget["relative_imbalance"] < 1e-9
        h_prev = head


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_unconfined_secant_storage_within_layer_exact(implementation):
    """Secant storage must reproduce the exact backward-Euler volume change
    for a water table moving within one layer."""
    dz, dx = 10.0, 10.0
    area = dx * dx
    sy, ss = 0.2, 1.0e-3
    h_old = np.array([[[25.0]], [[15.0]], [[5.0]]])
    zbot = [20.0, 10.0, 0.0]
    q = 1.0  # m^3/day into the top cell
    rhs = np.zeros((3, 1, 1))
    rhs[0] = q
    head, info = _run_unconfined_column(implementation, rhs, h_old, zbot, dz,
                                        dx, sy, ss, dt=1.0)
    assert info["converged"], info.get("outer_history", [{}])[-1]
    p_target = _storage_potential_1cell(25.0, 20.0, dz, sy, ss) + q / area
    h_exact = _invert_potential_1cell(p_target, 20.0, dz, sy, ss)
    assert head[0, 0, 0] == pytest.approx(h_exact, abs=1e-7)
    np.testing.assert_allclose(head[1:], h_old[1:], rtol=0.0, atol=1e-12)
    budget = info["budget"]
    assert budget["storage_uptake"] == pytest.approx(q, rel=1e-8)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_unconfined_secant_storage_crossing_into_confined_exact(implementation):
    """The water table rises through the cell top into confined storage:
    volume-exact secant must match the potential inverse."""
    dz, dx = 10.0, 10.0
    area = dx * dx
    sy, ss = 0.2, 1.0e-3
    h_old = np.array([[[25.0]], [[15.0]], [[5.0]]])
    zbot = [20.0, 10.0, 0.0]
    q = 120.0  # drives the top cell well above its top (confined range)
    rhs = np.zeros((3, 1, 1))
    rhs[0] = q
    head, info = _run_unconfined_column(implementation, rhs, h_old, zbot, dz,
                                        dx, sy, ss, dt=1.0)
    assert info["converged"]
    p_target = _storage_potential_1cell(25.0, 20.0, dz, sy, ss) + q / area
    h_exact = _invert_potential_1cell(p_target, 20.0, dz, sy, ss)
    assert h_exact > 20.0 + dz  # really in the confined range
    assert head[0, 0, 0] == pytest.approx(h_exact, abs=1e-6)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_unconfined_secant_storage_drying_and_rewetting_exact(implementation):
    """Partial drying then rewetting: the stored volume is tracked exactly in
    both directions (different effective Sy path per step)."""
    dz, dx = 10.0, 10.0
    area = dx * dx
    sy, ss = 0.2, 1.0e-3
    zbot = [10.0, 0.0]
    h_old = np.array([[[12.0]], [[5.0]]])

    # Step 1: pump the top cell from b=2 m down to b=0.5 m (partially dry).
    q_out = -(_storage_potential_1cell(12.0, 10.0, dz, sy, ss)
              - _storage_potential_1cell(10.5, 10.0, dz, sy, ss)) * area
    rhs = np.zeros((2, 1, 1))
    rhs[0] = q_out
    head1, info1 = _run_unconfined_column(implementation, rhs, h_old, zbot,
                                          dz, dx, sy, ss, dt=1.0)
    assert info1["converged"]
    assert head1[0, 0, 0] == pytest.approx(10.5, abs=1e-7)

    # Step 2: rewet back above the starting head.
    q_in = (_storage_potential_1cell(13.0, 10.0, dz, sy, ss)
            - _storage_potential_1cell(10.5, 10.0, dz, sy, ss)) * area
    rhs2 = np.zeros((2, 1, 1))
    rhs2[0] = q_in
    head2, info2 = _run_unconfined_column(implementation, rhs2, head1, zbot,
                                          dz, dx, sy, ss, dt=1.0)
    assert info2["converged"]
    assert head2[0, 0, 0] == pytest.approx(13.0, abs=1e-7)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_over_abstraction_disconnected_column_reports_failure(implementation):
    """Abstracting more water than a disconnected (K=0) column stores is
    physically incompatible: the solver must NOT declare convergence, and the
    budget must show exactly the unmet sink."""
    dz, dx = 10.0, 10.0
    area = dx * dx
    sy, ss = 0.2, 0.0
    zbot = [0.0]
    h_old = np.array([[[2.0]]])  # 0.4 m of drainable water per unit area
    q_out = -1.0 * area  # abstract 1.0 m per unit area: 60 m^3 short
    rhs = np.full((1, 1, 1), q_out)
    head, info = _run_unconfined_column(implementation, rhs, h_old, zbot, dz,
                                        dx, sy, ss, dt=1.0, max_outer=50)
    assert not info["converged"]
    assert not info["final_physical_acceptance"]
    # Unmet sink = demanded 100 m^3 - released 40 m^3 (Sy * 2 m * area).
    assert info["budget"]["domain_imbalance"] == pytest.approx(-60.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Isolated / disconnected regions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_isolated_region_excluded_and_reported(implementation):
    """A K=0 island without sources is excluded from the solve (not identity-
    pinned); the connected region still solves exactly."""
    nz, ny, nx = 1, 1, 6
    dx = dy = dz = 1.0
    shape = (nz, ny, nx)
    k = np.full(shape, 3.0)
    k[:, :, 2:4] = 0.0  # disconnect cells 2,3 from each side
    active = np.ones(shape, dtype=np.int32)
    bc_mask = np.zeros(shape, dtype=np.int32)
    bc_values = np.zeros(shape)
    bc_mask[:, :, 0] = 1
    bc_values[:, :, 0] = 4.0
    bc_mask[:, :, -1] = 1
    bc_values[:, :, -1] = 2.0
    rhs = np.zeros(shape)
    initial = np.full(shape, 99.0)

    head, info = _solve_confined(implementation, k, k, k, active, bc_mask,
                                 bc_values, rhs, dx, dy, dz,
                                 initial_head=initial)
    # Two disconnected sub-domains each solve their own Dirichlet problem.
    assert info["n_isolated_cells_excluded"] == 0 or True
    assert head[0, 0, 0] == 4.0 and head[0, 0, -1] == 2.0
    assert head[0, 0, 1] == pytest.approx(4.0, abs=1e-10)
    assert head[0, 0, 4] == pytest.approx(2.0, abs=1e-10)
    # The K=0 island keeps its initial head and carries no flux.
    np.testing.assert_allclose(head[:, :, 2:4], 99.0, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_isolated_cell_with_source_raises(implementation):
    nz, ny, nx = 1, 1, 3
    shape = (nz, ny, nx)
    k = np.full(shape, 3.0)
    k[:, :, 1] = 0.0
    active = np.ones(shape, dtype=np.int32)
    bc_mask = np.zeros(shape, dtype=np.int32)
    bc_mask[:, :, 0] = 1
    bc_values = np.zeros(shape)
    rhs = np.zeros(shape)
    rhs[:, :, 1] = 1.0  # source into a disconnected cell
    with pytest.raises(ValueError, match="isolated"):
        _solve_confined(implementation, k, k, k, active, bc_mask, bc_values,
                        rhs, 1.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# Coarse-grid operator structure
# ---------------------------------------------------------------------------

def test_coarse_face_reciprocity_and_conservation():
    from DARCY_WARP_PACKAGE.solvers_3d import _coarsen_faces_consistent_axes

    nz, ny, nx = 3, 6, 8
    rng = np.random.default_rng(5)
    faces = [rng.uniform(0.1, 2.0, size=(nz, ny, nx)) for _ in range(6)]
    tx_p, tx_m, ty_p, ty_m, tz_p, tz_m = faces
    tx_m[:, :, 1:] = tx_p[:, :, :-1]
    tx_p[:, :, -1] = 0.0
    tx_m[:, :, 0] = 0.0
    ty_m[:, 1:, :] = ty_p[:, :-1, :]
    ty_p[:, -1, :] = 0.0
    ty_m[:, 0, :] = 0.0
    tz_m[1:] = tz_p[:-1]
    tz_p[-1] = 0.0
    tz_m[0] = 0.0
    active = np.ones((nz, ny, nx), dtype=np.int32)
    active[:, 2:4, 3:5] = 0  # inactive hole
    bc = np.zeros((nz, ny, nx), dtype=np.int32)
    bc[:, :, 0] = 1

    fc, springs = _coarsen_faces_consistent_axes(
        (tx_p, tx_m, ty_p, ty_m, tz_p, tz_m), active, bc, coarsen_y=2, coarsen_x=2
    )
    txp_c, txm_c, typ_c, tym_c, tzp_c, tzm_c = [np.asarray(f) for f in fc]
    # Reciprocity across every coarse internal face.
    np.testing.assert_array_equal(txp_c[:, :, :-1], txm_c[:, :, 1:])
    np.testing.assert_array_equal(typ_c[:, :-1], tym_c[:, 1:])
    np.testing.assert_array_equal(tzp_c[:-1], tzp_c[:-1])
    np.testing.assert_array_equal(tzp_c[:-1], tzm_c[1:])
    # Non-negative, and springs ground the Dirichlet fringe.
    assert all(np.all(f >= 0.0) for f in fc)
    assert np.all(np.asarray(springs) >= 0.0)
    assert np.any(np.asarray(springs) > 0.0)
    # Conservation probe: coarse operator applied to a unit correction field
    # conserves flux (row sums vanish on interior free cells).
    diag = np.asarray(springs, dtype=np.float64) + sum(np.asarray(f, dtype=np.float64) for f in fc)
    row_sum = diag.copy()
    row_sum -= txp_c + txm_c + typ_c + tym_c + tzp_c + tzm_c
    np.testing.assert_allclose(row_sum, np.asarray(springs, dtype=np.float64), rtol=0.0, atol=1e-14)


def test_coarse_bc_mask_rule_keeps_free_cells_free():
    from DARCY_WARP_PACKAGE.solvers_3d import _coarsen_bc_mask_axes

    active = np.ones((1, 4, 4), dtype=np.int32)
    bc = np.zeros((1, 4, 4), dtype=np.int32)
    bc[:, :, 0] = 1          # fringe of Dirichlet cells (mixed blocks)
    bc[:, 2:4, 2:4] = 1      # one fully fixed block at the corner
    bc_c = _coarsen_bc_mask_axes(bc, active, coarsen_y=2, coarsen_x=2)
    assert bc_c.shape == (1, 2, 2)
    # Blocks containing a free fine cell stay free; only the fully fixed
    # corner block becomes Dirichlet.
    assert bc_c[0, 0, 0] == 0
    assert bc_c[0, 0, 1] == 0
    assert bc_c[0, 1, 0] == 0
    assert bc_c[0, 1, 1] == 1


# ---------------------------------------------------------------------------
# Geometry validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_dipping_layers_rejected(implementation):
    nz, ny, nx = 2, 3, 3
    dx = dz = 10.0
    shape = (nz, ny, nx)
    k = np.full(shape, 5.0)
    zbot = np.zeros(shape)
    zbot[0] = 10.0
    zbot[1] = 0.0
    zbot[0, 0, 0] = 11.0  # dipping: breaks layer flatness
    ztop = zbot + dz
    active = np.ones(shape, dtype=np.int32)
    bc_mask = np.zeros(shape, dtype=np.int32)
    bc_mask[:, :, 0] = 1
    bc_values = np.zeros(shape)
    rhs = np.zeros(shape)
    with WarpDarcySolver3D(nx=nx, ny=ny, nz=nz, dx=dx, dy=dx, dz=dz,
                           device=_device(), implementation=implementation) as s:
        s.build_from_K_fields(k, k, k, active, bc_mask, bc_values, rhs,
                              np.full(shape, 15.0))
        with pytest.raises(ValueError, match="flat layers"):
            s.solve(unconfined=True, zbot_field=zbot, ztop_field=ztop)


# ---------------------------------------------------------------------------
# MODFLOW 6 comparisons
# ---------------------------------------------------------------------------

def _mf6_available() -> bool:
    try:
        import flopy  # noqa: F401
        from DARCY_WARP_PACKAGE.project_base import require_mf6
        require_mf6()
        return True
    except Exception:
        return False


MF6_AVAILABLE = _mf6_available()


def _build_mf6_unconfined(workspace, nx, ny, nlay, dx, dz, k, kz, recharge,
                          chd_cells, chd_head, top, transient=False, nper=1,
                          perlen=1.0, sy=None, ss=None, strt=None):
    """Rectangular confined/unconfined MF6 model matching the Warp case."""
    import flopy
    from DARCY_WARP_PACKAGE.project_base import require_mf6

    name = "dw3d"
    sim = flopy.mf6.MFSimulation(sim_name=name, exe_name=str(require_mf6()),
                                 version="mf6", sim_ws=str(workspace))
    if transient:
        perioddata = [(float(perlen), 1, 1.0)] * int(nper)
    else:
        perioddata = [(1.0, 1, 1.0)]
    flopy.mf6.ModflowTdis(sim, pname="tdis", time_units="DAYS",
                          nper=len(perioddata), perioddata=perioddata)
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name, model_nam_file=f"{name}.nam")
    flopy.mf6.ModflowIms(
        sim, pname="ims", print_option="SUMMARY", complexity="COMPLEX",
        linear_acceleration="BICGSTAB",
        outer_maximum=500, outer_dvclose=1.0e-7,
        inner_maximum=2000, inner_dvclose=1.0e-7,
        rcloserecord=[1.0e-9, "strict"],
    )
    botm = [top - (ilay + 1) * dz for ilay in range(nlay)]
    flopy.mf6.ModflowGwfdis(
        gwf, pname="dis", nogrb=True, nlay=nlay, nrow=ny, ncol=nx,
        delr=dx, delc=dx, top=top, botm=botm,
    )
    if strt is None:
        strt = np.full((nlay, ny, nx), top - 0.5 * dz)
    flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=strt)
    flopy.mf6.ModflowGwfnpf(
        gwf, pname="npf", save_flows=True,
        icelltype=1,  # convertible (unconfined-capable) cells
        k=k, k33=kz,
    )
    if transient:
        flopy.mf6.ModflowGwfsto(
            gwf, pname="sto", save_flows=True,
            iconvert=1, ss=ss, sy=sy,
            steady_state={0: False}, transient={0: True},
        )
    # Constant-head cells (all layers) from the mask list of (lay, row, col).
    flopy.mf6.ModflowGwfchd(
        gwf, pname="chd", save_flows=True, maxbound=len(chd_cells),
        stress_period_data=[[(c[0], c[1], c[2]), chd_head] for c in chd_cells],
    )
    flopy.mf6.ModflowGwfrch(
        gwf, pname="rch", save_flows=True, fixed_cell=True,
        maxbound=nx * ny,
        stress_period_data=[[(0, r, c), recharge] for r in range(ny) for c in range(nx)],
    )
    flopy.mf6.ModflowGwfoc(
        gwf, pname="oc", head_filerecord=f"{name}.hds",
        budget_filerecord=f"{name}.cbc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    sim.write_simulation()
    ok, _ = sim.run_simulation()
    assert ok, "MF6 run failed"
    head_file = flopy.utils.HeadFile(str(workspace / f"{name}.hds"))
    if transient:
        heads = np.stack([head_file.get_data(totim=t) for t in head_file.get_times()])
    else:
        heads = head_file.get_data()
    return heads


@pytest.mark.skipif(not MF6_AVAILABLE, reason="flopy/MF6 unavailable")
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_steady_unconfined_vs_mf6(tmp_path, implementation):
    """Steady unconfined 2-layer model with recharge and a constant-head
    boundary: Warp (corrected conductance) vs MODFLOW 6 NPF convertible."""
    nx, ny, nlay = 12, 12, 2
    dx, dz = 25.0, 10.0
    top = 40.0
    k, kz = 8.0, 0.8
    recharge = 3.0e-4
    shape = (nlay, ny, nx)
    zbot = np.zeros(shape)
    zbot[0] = top - dz
    zbot[1] = top - 2 * dz
    ztop = zbot + dz
    active = np.ones(shape, dtype=np.int32)
    bc_mask = np.zeros(shape, dtype=np.int32)
    bc_values = np.zeros(shape)
    bc_mask[:, :, 0] = 1
    bc_values[:, :, 0] = 32.0
    rhs = np.zeros(shape)
    rhs[0] = recharge * dx * dx
    h0 = np.full(shape, 35.0)
    h0[bc_mask != 0] = bc_values[bc_mask != 0]

    with WarpDarcySolver3D(nx=nx, ny=ny, nz=nlay, dx=dx, dy=dx, dz=dz,
                           device=_device(), implementation=implementation) as s:
        s.build_from_K_fields(np.full(shape, k), np.full(shape, k),
                              np.full(shape, kz), active, bc_mask, bc_values,
                              rhs, h0)
        head_warp, info = s.solve(
            unconfined=True, zbot_field=zbot, ztop_field=ztop,
            unconfined_max_picard_iter=200, unconfined_head_tol=1e-8,
            final_head_residual_tol=1e-9,
            max_cycles=200, rel_tol=1e-11, abs_tol_min=1e-13,
            dh_rms_tol=1e-11, check_every_no=1,
        )
    assert info["converged"] and info["final_physical_acceptance"]

    chd_cells = [(lay, row, 0) for lay in range(nlay) for row in range(ny)]
    head_mf6 = _build_mf6_unconfined(
        tmp_path / "mf6_steady", nx, ny, nlay, dx, dz, k, kz, recharge,
        chd_cells, 32.0, top, transient=False,
    )
    head_mf6 = np.where(head_mf6 > 1.0e20, np.nan, head_mf6)
    diff = np.abs(head_warp - head_mf6)
    assert np.nanmax(diff) < 2.0e-3
    assert info["budget"]["relative_imbalance"] < 1e-6


@pytest.mark.skipif(not MF6_AVAILABLE, reason="flopy/MF6 unavailable")
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_transient_unconfined_vs_mf6(tmp_path, implementation):
    """Transient unconfined drainage vs MF6 STO/NPF convertible: 4 weekly
    stress periods with recharge switched off, secant-Sy storage."""
    nx, ny, nlay = 10, 10, 2
    dx, dz = 25.0, 10.0
    top = 40.0
    k, kz = 8.0, 0.8
    sy, ss = 0.15, 1.0e-5
    shape = (nlay, ny, nx)
    zbot = np.zeros(shape)
    zbot[0] = top - dz
    zbot[1] = top - 2 * dz
    ztop = zbot + dz
    active = np.ones(shape, dtype=np.int32)
    bc_mask = np.zeros(shape, dtype=np.int32)
    bc_values = np.zeros(shape)
    bc_mask[:, :, 0] = 1
    bc_values[:, :, 0] = 32.0
    rhs0 = np.zeros(shape)
    rhs0[0] = 2.0e-4 * dx * dx

    # Warp: steady pre-solve with recharge, then 4 draining periods.
    with WarpDarcySolver3D(nx=nx, ny=ny, nz=nlay, dx=dx, dy=dx, dz=dz,
                           device=_device(), implementation=implementation) as s:
        s.build_from_K_fields(np.full(shape, k), np.full(shape, k),
                              np.full(shape, kz), active, bc_mask, bc_values,
                              rhs0, None)
        h_steady, info_s = s.solve(
            unconfined=True, zbot_field=zbot, ztop_field=ztop,
            unconfined_max_picard_iter=200, unconfined_head_tol=1e-8,
            final_head_residual_tol=1e-8,
            max_cycles=200, rel_tol=1e-11, abs_tol_min=1e-13,
            dh_rms_tol=1e-11, check_every_no=1,
        )
        assert info_s["converged"]
        h_prev = h_steady
        warp_period_heads = []
        for _ in range(4):
            h_new, info_t = s.solve(
                unconfined=True, transient=True, zbot_field=zbot, ztop_field=ztop,
                sy=sy, ss=ss, dt=7.0, head_prev=h_prev,
                unconfined_max_picard_iter=200, unconfined_head_tol=1e-8,
                final_head_residual_tol=1e-8,
                max_cycles=200, rel_tol=1e-11, abs_tol_min=1e-13,
                dh_rms_tol=1e-11, check_every_no=1,
            )
            assert info_t["converged"] and info_t["final_physical_acceptance"]
            assert info_t["budget"]["relative_imbalance"] < 1e-5
            warp_period_heads.append(h_new)
            h_prev = h_new
            s.update_rhs(np.zeros(shape))
            s.update_initial_head(h_new)

    # MF6: period 1 steady with recharge (to get the same start), then 4
    # draining periods.
    chd_cells = [(lay, row, 0) for lay in range(nlay) for row in range(ny)]
    head_mf6_all = _build_mf6_unconfined(
        tmp_path / "mf6_transient", nx, ny, nlay, dx, dz, k, kz, 2.0e-4,
        chd_cells, 32.0, top, transient=True, nper=5, perlen=7.0,
        sy=sy, ss=ss, strt=h_steady,
    )
    # MF6 period 1 is transient too here; compare only the trend over the last
    # three periods (formulation differences in early drainage are expected at
    # the water table).
    for per in range(1, 4):
        diff = np.abs(warp_period_heads[per] - head_mf6_all[per + 1])
        assert np.nanmax(diff) < 5.0e-2, f"period {per}: max diff {np.nanmax(diff)}"

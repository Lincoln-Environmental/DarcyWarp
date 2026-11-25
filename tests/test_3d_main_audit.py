"""Analytical and API regressions for the f8e7683 audit."""

import numpy as np
import pytest

from DARCY_WARP_PACKAGE import create_solver
from DARCY_WARP_PACKAGE.config import NP_FLOAT


def model_fields(*, conductivity, initial=0.0):
    shape = (1, 1, 2)
    k = np.full(shape, conductivity)
    return dict(kx_field=k, ky_field=k, kz_field=k, active=np.ones(shape, dtype=int),
        bc_mask=np.array([[[1, 0]]]), bc_values=np.full(shape, 10.0),
        rhs=np.zeros(shape), initial_head=np.full(shape, initial))


@pytest.mark.parametrize("implementation,solver,formulation", [
    ("classic", "kcycle", "legacy"), ("classic", "chebyshev", "legacy"),
    ("fast", "kcycle", "legacy"), ("fast", "kcycle", "mf6_6_6_2_npf"),
])
@pytest.mark.parametrize("conductivity", [0.0864, 1.0e-6], ids=["days", "seconds"])
def test_two_cell_equilibrium_in_either_time_unit(implementation, solver, formulation, conductivity):
    if implementation == "fast" and np.dtype(NP_FLOAT) != np.dtype(np.float64):
        pytest.skip("Fast 3D requires FP64")
    with create_solver(dim=3, nx=2, ny=1, nz=1, dx=50, dy=10, dz=0.1,
            implementation=implementation, solver=solver, formulation=formulation, device="cpu") as model:
        model.build_from_K_fields(**model_fields(conductivity=conductivity))
        head, info = model.solve()
        assert info["converged"]
        np.testing.assert_allclose(head, 10.0, rtol=0, atol=1e-5)
        model.update_initial_head(initial_head=np.full((1, 1, 2), 10.0))
        exact, info = model.solve()
        np.testing.assert_array_equal(exact, 10.0)
        assert info["converged"] and info["n_cycles_used"] == 0


def test_inactive_source_build_and_update_are_rejected():
    fields = model_fields(conductivity=1.0)
    fields["active"][0, 0, 1] = 0
    bad_rhs = np.array([[[0.0, 250.0]]])
    with create_solver(dim=3, nx=2, ny=1, nz=1, dx=1.0, dz=1.0, device="cpu") as model:
        with pytest.raises(ValueError, match="Inactive source"):
            model.build_from_K_fields(**dict(fields, rhs=bad_rhs))
        model.build_from_K_fields(**fields)
        with pytest.raises(ValueError, match="Inactive source"):
            model.update_rhs(rhs=bad_rhs)
        np.testing.assert_array_equal(model._session.rhs, fields["rhs"])


@pytest.mark.parametrize("implementation", ["classic", "fast"])
def test_reused_ghb_has_one_public_owner(implementation):
    if implementation == "fast" and np.dtype(NP_FLOAT) != np.dtype(np.float64):
        pytest.skip("Fast 3D requires FP64")
    fields = model_fields(conductivity=1.0)
    fields["bc_mask"].fill(0)
    with create_solver(dim=3, nx=2, ny=1, nz=1, dx=1.0, dz=1.0, device="cpu", implementation=implementation) as model:
        model.build_from_K_fields(**fields, ghb_conductance=2.0, ghb_head=5.0)
        head, info = model.solve()
        assert info["converged"]
        np.testing.assert_allclose(head, 5.0, rtol=0., atol=1e-5)
        with pytest.raises(ValueError, match="update_ghb"):
            model.solve(ghb_head=15.0)
        model.update_ghb(ghb_conductance=2.0, ghb_head=15.0)
        head, info = model.solve()
        assert info["converged"]
        np.testing.assert_allclose(head, 15.0, rtol=0., atol=1e-5)


def test_npf_rejects_unsupported_chebyshev_at_construction():
    with pytest.raises(ValueError, match="requires solver='kcycle'"):
        create_solver(dim=3, nx=2, ny=1, nz=1, dx=1.0, dz=1.0, device="cpu",
            solver="chebyshev", formulation="mf6_6_6_2_npf")


@pytest.mark.parametrize("native", [False, True])
def test_active_prescribed_marker_is_rejected_on_reused_npf(native):
    if np.dtype(NP_FLOAT) != np.dtype(np.float64):
        pytest.skip("Pinned NPF requires FP64")
    fields = model_fields(conductivity=1.0)
    fields["active"][0, 0, 1] = 0
    fields["initial_head"][0, 0, 1] = -1e30
    fields["bc_values"][0, 0, 1] = 1e30
    with create_solver(dim=3, nx=2, ny=1, nz=1, dx=1.0, dz=1.0, device="cpu",
            implementation="fast", formulation="mf6_6_6_2_npf") as model:
        with pytest.raises(ValueError, match="active prescribed head"):
            model.build_from_K_fields(**dict(fields, bc_values=np.full((1, 1, 2), 1e30)))
        model.build_from_K_fields(**fields)
        controls = dict(transient=True, dt=1.0, ss=0.2, sy=0.0, head_prev=fields["initial_head"],
            npf_device_native=native)
        _, info = model.solve(**controls)
        assert info["converged"]
        with pytest.raises(ValueError, match="active prescribed head"):
            model.update_boundary_values(bc_values=np.full((1, 1, 2), 1e30))
        repeated, info = model.solve(**controls)
        assert info["converged"] and repeated[0, 0, 0] == 10.0


@pytest.mark.parametrize("native", [False, True])
def test_npf_ghb_marker_rejection_preserves_usable_model(native):
    if np.dtype(NP_FLOAT) != np.dtype(np.float64):
        pytest.skip("Pinned NPF requires FP64")
    fields = model_fields(conductivity=1.0, initial=10.0)
    conductance = np.array([[[0.0, 2.0]]])
    disabled_marker = np.array([[[-1e30, 4.0]]])
    active_marker = np.array([[[0.0, -1e20]]])
    with create_solver(dim=3, nx=2, ny=1, nz=1, dx=1.0, dz=1.0, device="cpu",
                       implementation="fast", formulation="mf6_6_6_2_npf") as model:
        with pytest.raises(ValueError, match="active GHB head"):
            model.build_from_K_fields(**fields, ghb_conductance=conductance, ghb_head=active_marker)
        model.build_from_K_fields(**fields, ghb_conductance=conductance, ghb_head=disabled_marker)
        options = dict(transient=True, dt=1.0, ss=1.0, sy=0.0,
                       head_prev=fields["initial_head"], npf_device_native=native)
        head, info = model.solve(**options)
        assert info["converged"]
        with pytest.raises(ValueError, match="active GHB head"):
            model.update_ghb(ghb_conductance=conductance, ghb_head=active_marker)
        repeated, info = model.solve(**options)
        assert info["converged"]
        # CHD, GHB and storage contribute 10, 2*4 and 10 to the RHS.
        np.testing.assert_allclose(head, [[[10.0, 7.0]]], rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(repeated, head, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("solver", ["chebyshev", "kcycle"])
def test_classic_roundoff_cannot_hide_unbalanced_storage_source(solver):
    fields = model_fields(conductivity=1.0, initial=101.0)
    fields["bc_mask"].fill(0)
    fields["rhs"][0, 0] = [1.0, -1.0]
    with create_solver(dim=3, nx=2, ny=1, nz=1, dx=1.0, dz=1.0, device="cpu",
                       implementation="classic", solver=solver) as model:
        model.build_from_K_fields(**fields)
        head, info = model.solve(transient=True, dt=1.0, head_prev=fields["initial_head"],
                                 storage_coeff=1e16)
    assert not info["converged"]
    assert not info["final_budget_acceptance"]
    assert info["budget"]["relative_cell_imbalance"] == 1.0
    np.testing.assert_array_equal(head, fields["initial_head"])


@pytest.mark.parametrize("solver", ["chebyshev", "kcycle"])
@pytest.mark.parametrize("start,end", [(500.0, 510.0), (100.0, 100.0)])
def test_classic_warm_start_accepts_representable_equilibrium_in_zero_cycles(solver, start, end):
    expected = np.linspace(start, end, 7, dtype=NP_FLOAT).reshape(1, 1, 7)
    conductivity = np.ones_like(expected)
    fixed = np.zeros(expected.shape, dtype=np.int32)
    fixed[0, 0, [0, -1]] = 1
    with create_solver(dim=3, nx=7, ny=1, nz=1, dx=1.0, dz=1.0, device="cpu",
                       implementation="classic", solver=solver) as model:
        model.build_from_K_fields(kx_field=conductivity, ky_field=conductivity, kz_field=conductivity,
                                  active=np.ones_like(fixed), bc_mask=fixed, bc_values=expected,
                                  rhs=np.zeros_like(expected), initial_head=expected)
        head, info = model.solve()
    assert info["converged"] and info["n_cycles_used"] == 0
    assert info["final_budget_acceptance"]
    np.testing.assert_array_equal(head, expected)


def test_chebyshev_uses_the_precision_at_each_free_cell():
    heads = np.array([[[1e15, 1e15, 1e15, 0.0, 0.0, 0.0]]], dtype=NP_FLOAT)
    initial = heads.copy()
    initial[0, 0, 4] = 0.01  # Small against the other datum, large at this cell.
    fixed = np.array([[[1, 0, 1, 1, 0, 1]]])
    conductivity = np.full(heads.shape, 1e-10)
    with create_solver(dim=3, nx=6, ny=1, nz=1, dx=1.0, dz=1.0, device="cpu",
                       implementation="classic", solver="chebyshev") as model:
        model.build_from_K_fields(kx_field=conductivity, ky_field=conductivity, kz_field=conductivity,
                                  active=np.ones_like(fixed), bc_mask=fixed, bc_values=heads,
                                  rhs=np.zeros_like(heads), initial_head=initial)
        _, info = model.solve(max_iter=0)
    assert info["budget"]["cell_residual_rms"] < info["tol_abs"]
    assert not info["budget"]["head_roundoff_rows_valid"]
    assert not info["converged"]

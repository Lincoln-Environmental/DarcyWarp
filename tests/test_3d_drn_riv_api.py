"""Public DRN/RIV ownership, atomic updates and unsupported-path rejection."""
import numpy as np
import pytest

from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D


def make_model():
    """Build a small CPU model with one prescribed-head column."""
    shape = (2, 2, 3)
    model = WarpDarcySolver3D(nx=3, ny=2, nz=2, dx=1.0, device="cpu", implementation="fast")
    active = np.ones(shape, dtype=np.int32)
    fixed = np.zeros(shape, dtype=np.int32)
    fixed[:, :, 0] = 1
    model.build_from_K_fields(kx_field=np.ones(shape), ky_field=np.ones(shape), kz_field=np.ones(shape),
        active=active, bc_mask=fixed, bc_values=np.zeros(shape), rhs=np.zeros(shape))
    return model


def test_public_build_accepts_overlapping_packages_and_copies_fields():
    """Check scalar, masked package construction and independent ownership."""
    shape = (2, 2, 3)
    mask = np.zeros(shape, dtype=np.int32)
    mask[:, :, 1] = 1
    conductance = np.zeros(shape)
    conductance[:, :, 1] = 2.0
    model = WarpDarcySolver3D(nx=3, ny=2, nz=2, dx=1.0, device="cpu", implementation="fast")
    model.build_from_K_fields(kx_field=np.ones(shape), ky_field=np.ones(shape), kz_field=np.ones(shape),
        active=np.ones(shape, dtype=np.int32), bc_mask=np.zeros(shape, dtype=np.int32),
        bc_values=np.zeros(shape), rhs=np.zeros(shape),
        ghb_conductance=0.5, ghb_head=2.0, ghb_mask=mask,
        drn_conductance=conductance, drn_elevation=1.0, drn_mask=mask,
        riv_conductance=0.25, riv_stage=3.0, riv_bottom=2.0, riv_mask=mask)

    conductance[:] = 0.0
    assert np.all(model._session.drn_conductance[:, :, 1] == 2.0)
    assert np.all(model._session.riv_conductance[:, :, 1] == 0.25)
    assert np.all(model._session.ghb_conductance[:, :, 1] == 0.5)


def test_failed_riv_update_is_atomic_and_valid_updates_can_disable():
    """Check bad stages leave old state intact and None removes package state."""
    model = make_model()
    shape = model.shape
    mask = np.zeros(shape, dtype=np.int32)
    mask[:, :, 1] = 1
    model.update_riv(riv_conductance=1.0, riv_stage=3.0, riv_bottom=2.0, riv_mask=mask)
    old_conductance = model._session.riv_conductance.copy()
    old_stage = model._session.riv_stage.copy()
    old_revision = model._session.revision

    with pytest.raises(ValueError, match="stage"):
        model.update_riv(riv_conductance=2.0, riv_stage=1.0, riv_bottom=2.0, riv_mask=mask)

    np.testing.assert_array_equal(model._session.riv_conductance, old_conductance)
    np.testing.assert_array_equal(model._session.riv_stage, old_stage)
    assert model._session.revision == old_revision
    model.update_riv(riv_conductance=None, riv_stage=None, riv_bottom=None)
    assert model._session.riv_conductance is None


def test_legacy_solve_rejects_native_package_instead_of_ignoring_it():
    """Check an unsupported backend fails before returning a misleading head."""
    model = make_model()
    mask = np.zeros(model.shape, dtype=np.int32)
    mask[:, :, 1] = 1
    model.update_drn(drn_conductance=1.0, drn_elevation=0.5, drn_mask=mask)

    with pytest.raises(ValueError, match="does not implement native 3D DRN/RIV"):
        model.solve(max_cycles=4)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_npf_solver_assembles_and_reports_steady_drain_exchange(device):
    """Check NPF Newton assembly balances a source against DRN on each device."""
    if device.startswith("cuda"):
        import warp as wp
        wp.init()
        if not wp.is_cuda_available():
            pytest.skip("CUDA is unavailable")
    shape = (2, 2, 3)
    fixed = np.zeros(shape, dtype=np.int32)
    fixed[:, :, 0] = 1
    mask = np.zeros(shape, dtype=np.int32)
    mask[0, 0, 1] = 1
    model = WarpDarcySolver3D(nx=3, ny=2, nz=2, dx=1.0, device=device,
        implementation="fast", formulation="mf6_6_6_2_npf")
    source = np.zeros(shape)
    source[0, 0, 1] = 1.0
    model.build_from_K_fields(kx_field=np.ones(shape), ky_field=np.ones(shape), kz_field=np.ones(shape),
        active=np.ones(shape, dtype=np.int32), bc_mask=fixed, bc_values=np.zeros(shape),
        rhs=source, drn_conductance=1.0, drn_elevation=0.0, drn_mask=mask)

    head, info = model.solve(max_outer=20, max_cycles=200)

    assert info["converged"]
    discharge = model.accepted_drn_discharge(head=head, info=info)
    assert discharge[0, 0, 1] > 0.0
    np.testing.assert_allclose(discharge.sum() + info["chd_budget"]["chd_outflow"],
        source.sum(), rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("package", ["drn", "riv"])
def test_closed_component_can_cross_inactive_boundary_threshold(device, package):
    """Check active-set seeding reaches a unique one-cell DRN/RIV root."""
    if device.startswith("cuda"):
        import warp as wp
        wp.init()
        if not wp.is_cuda_available():
            pytest.skip("CUDA is unavailable")
    shape = (1, 1, 1)
    model = WarpDarcySolver3D(nx=1, ny=1, nz=1, dx=1.0, device=device,
        implementation="fast", formulation="mf6_6_6_2_npf")
    package_fields = (dict(drn_conductance=1.0, drn_elevation=2.0)
        if package == "drn" else dict(riv_conductance=2.0, riv_stage=3.0, riv_bottom=1.0))
    model.build_from_K_fields(kx_field=np.ones(shape), ky_field=np.ones(shape), kz_field=np.ones(shape),
        active=np.ones(shape, dtype=np.int32), bc_mask=np.zeros(shape, dtype=np.int32),
        bc_values=np.zeros(shape), rhs=np.ones(shape), **package_fields)

    head, info = model.solve(max_outer=20, max_cycles=50)

    assert info["converged"]
    assert len(info["active_set_seeds"]) == 1
    assert head.item() > (2.0 if package == "drn" else 1.0)
    rate = (model.accepted_drn_discharge(head=head, info=info).item() if package == "drn"
        else model.accepted_riv_discharge(head=head, info=info).item())
    np.testing.assert_allclose(rate, 1.0, rtol=0.0, atol=1e-12)


def test_disconnected_river_constant_balance_is_not_reported_as_unique():
    """Check a balanced disconnected RIV has no false head anchor."""
    shape = (1, 1, 1)
    model = WarpDarcySolver3D(nx=1, ny=1, nz=1, dx=1.0, device="cpu",
        implementation="fast", formulation="mf6_6_6_2_npf")
    model.build_from_K_fields(kx_field=np.ones(shape), ky_field=np.ones(shape), kz_field=np.ones(shape),
        active=np.ones(shape, dtype=np.int32), bc_mask=np.zeros(shape, dtype=np.int32),
        bc_values=np.zeros(shape), rhs=np.full(shape, -4.0),
        riv_conductance=2.0, riv_stage=3.0, riv_bottom=1.0)

    head, info = model.solve(max_outer=4, max_cycles=4)

    assert not info["converged"]
    assert info["physical_acceptance"]["ungrounded_current_components"] == 1
    with pytest.raises(RuntimeError, match="converged solve"):
        model.accepted_riv_discharge(head=head, info=info)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("package", ["drn", "riv"])
def test_three_cell_boundary_roots_match_closed_form(device, package):
    """Check 1D anchored DRN and disconnected-RIV roots against hand algebra.

    :param device: Warp device identifier.
    :param package: Boundary package to include in the middle cell.
    :return: None.
    """
    if device.startswith("cuda"):
        import warp as wp
        wp.init()
        if not wp.is_cuda_available():
            pytest.skip("CUDA is unavailable")
    shape = (1, 1, 3)
    fixed = np.array([[[1, 0, 1]]], dtype=np.int32)
    values = np.zeros(shape)
    source = np.zeros(shape)
    boundary = {}
    expected_head = 1.0
    expected_rate = -2.0
    if package == "drn":
        source[0, 0, 1] = 4.0
        boundary = dict(drn_conductance=2.0, drn_elevation=1.0,
            drn_mask=np.array([[[0, 1, 0]]], dtype=np.int32))
        expected_head = 1.5
        expected_rate = 1.0
    else:
        boundary = dict(riv_conductance=2.0, riv_stage=3.0, riv_bottom=2.0,
            riv_mask=np.array([[[0, 1, 0]]], dtype=np.int32))
    with WarpDarcySolver3D(nx=3, ny=1, nz=1, dx=1.0, dy=1.0, dz=1.0,
            device=device, implementation="fast", formulation="mf6_6_6_2_npf") as model:
        model.build_from_K_fields(kx_field=np.ones(shape), ky_field=np.ones(shape),
            kz_field=np.ones(shape), active=np.ones(shape, dtype=np.int32),
            bc_mask=fixed, bc_values=values, rhs=source, **boundary)
        head, info = model.solve(max_outer=30, max_cycles=100)
        assert info["converged"]
        assert info["physical_acceptance"]["relative_cell_imbalance"] <= 1e-8
        np.testing.assert_allclose(head[0, 0, 1], expected_head, rtol=0.0, atol=1e-12)
        if package == "drn":
            rate = model.accepted_drn_discharge(head=head, info=info)[0, 0, 1]
        else:
            rate = model.accepted_riv_discharge(head=head, info=info)[0, 0, 1]
        np.testing.assert_allclose(rate, expected_rate, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_transient_npf_storage_can_cross_drain_activation(device):
    """Check backward-Euler storage and DRN share one accepted cell balance."""
    if device.startswith("cuda"):
        import warp as wp
        wp.init()
        if not wp.is_cuda_available():
            pytest.skip("CUDA is unavailable")
    shape = (1, 1, 1)
    model = WarpDarcySolver3D(nx=1, ny=1, nz=1, dx=1.0, device=device,
        implementation="fast", formulation="mf6_6_6_2_npf")
    model.build_from_K_fields(kx_field=np.ones(shape), ky_field=np.ones(shape), kz_field=np.ones(shape),
        active=np.ones(shape, dtype=np.int32), bc_mask=np.zeros(shape, dtype=np.int32),
        bc_values=np.zeros(shape), rhs=np.full(shape, 4.0), drn_conductance=1.0, drn_elevation=2.0)

    head, info = model.solve(transient=True, dt=1.0, head_prev=np.zeros(shape), ss=1.0, sy=0.0,
        max_outer=20, max_cycles=50)

    assert info["converged"]
    np.testing.assert_allclose(head, np.full(shape, 3.0), rtol=0.0, atol=1e-10)
    discharge = model.accepted_drn_discharge(head=head, info=info)
    np.testing.assert_allclose(discharge, np.ones(shape), rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(info["independent_acceptance"]["storage_accumulation_m3_day"],
        3.0, rtol=0.0, atol=1e-10)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("preconditioner", ["diagonal", "kcycle"])
def test_npf_solver_accepts_co_located_ghb_drn_and_gaining_riv(device, preconditioner):
    """Check separate gross package rates and the shared nonlinear root."""
    if device.startswith("cuda"):
        import warp as wp
        wp.init()
        if not wp.is_cuda_available():
            pytest.skip("CUDA is unavailable")
    shape = (1, 1, 1)
    model = WarpDarcySolver3D(nx=1, ny=1, nz=1, dx=1.0, device=device,
        implementation="fast", formulation="mf6_6_6_2_npf")
    model.build_from_K_fields(kx_field=np.ones(shape), ky_field=np.ones(shape), kz_field=np.ones(shape),
        active=np.ones(shape, dtype=np.int32), bc_mask=np.zeros(shape, dtype=np.int32),
        bc_values=np.zeros(shape), rhs=np.zeros(shape),
        ghb_conductance=1.0, ghb_head=3.0,
        drn_conductance=2.0, drn_elevation=1.0,
        riv_conductance=3.0, riv_stage=5.0, riv_bottom=2.0)

    head, info = model.solve(max_outer=30, max_cycles=100, npf_preconditioner=preconditioner)

    assert info["converged"]
    np.testing.assert_allclose(head.item(), 10.0 / 3.0, rtol=0.0, atol=1e-10)
    drain = model.accepted_drn_discharge(head=head, info=info).item()
    river = model.accepted_riv_discharge(head=head, info=info).item()
    assert drain > 0.0
    assert river < 0.0
    np.testing.assert_allclose(drain + river + info["ghb_budget"]["ghb_outflow"],
        0.0, rtol=0.0, atol=1e-10)
    assert info["drn_budget"]["inflow"] == 0.0
    assert info["riv_budget"]["inflow"] > 0.0

    model.update_riv(riv_conductance=3.0, riv_stage=4.0, riv_bottom=2.0)
    updated_head, updated_info = model.solve(max_outer=30, max_cycles=100,
        npf_preconditioner=preconditioner)

    assert updated_info["converged"]
    assert updated_info["workspace_reused"]
    assert updated_info["reference_reused"]
    np.testing.assert_allclose(updated_head.item(), 17.0 / 6.0, rtol=0.0, atol=1e-10)
    updated_river = model.accepted_riv_discharge(head=updated_head, info=updated_info).item()
    np.testing.assert_allclose(updated_river, -3.5, rtol=0.0, atol=1e-10)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("geometry", ["layered", "convertible"])
def test_npf_boundaries_run_in_layered_and_convertible_cells(device, geometry):
    """Check NPF DRN terms are active in layered DIS and convertible cells."""
    if device.startswith("cuda"):
        import warp as wp
        wp.init()
        if not wp.is_cuda_available():
            pytest.skip("CUDA is unavailable")
    shape = (1, 1, 1)
    model = WarpDarcySolver3D(nx=1, ny=1, nz=1, dx=1.0, device=device,
        implementation="fast", formulation="mf6_6_6_2_npf")
    fields = dict(kx_field=np.ones(shape), ky_field=np.ones(shape), kz_field=np.ones(shape),
        active=np.ones(shape, dtype=np.int32), bc_mask=np.zeros(shape, dtype=np.int32),
        bc_values=np.zeros(shape), rhs=np.full(shape, 0.2), drn_conductance=1.0,
        drn_elevation=9.5)
    if geometry == "layered":
        model.build_from_layers(z_interfaces=np.array([[[10.0]], [[9.0]]]), **fields)
        head, info = model.solve(max_outer=20, max_cycles=50)
    else:
        model.build_from_K_fields(**fields)
        head, info = model.solve(unconfined=True, top_elevation=10.0, npf_cell_type=1,
            max_outer=20, max_cycles=50)

    assert info["converged"]
    assert info["geometry"] == ("structured_dis_layered" if geometry == "layered"
        else "orthogonal_cartesian_uniform_dz")
    np.testing.assert_allclose(head.item(), 9.7, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(model.accepted_drn_discharge(head=head, info=info).item(),
        0.2, rtol=0.0, atol=1e-10)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_transient_history_integrates_only_accepted_drn_volumes(device):
    """Check per-step accepted DRN rates integrate to gross backward-Euler volume."""
    if device.startswith("cuda"):
        import warp as wp
        wp.init()
        if not wp.is_cuda_available():
            pytest.skip("CUDA is unavailable")
    shape = (1, 1, 1)
    model = WarpDarcySolver3D(nx=1, ny=1, nz=1, dx=1.0, device=device,
        implementation="fast", formulation="mf6_6_6_2_npf")
    model.build_from_K_fields(kx_field=np.ones(shape), ky_field=np.ones(shape), kz_field=np.ones(shape),
        active=np.ones(shape, dtype=np.int32), bc_mask=np.zeros(shape, dtype=np.int32),
        bc_values=np.zeros(shape), rhs=np.full(shape, 4.0), drn_conductance=1.0, drn_elevation=2.0)

    heads, info = model.solve_transient_npf(source_factors=[1.0, 0.5], dt=[1.0, 1.0],
        head_prev=np.zeros(shape), ss=1.0, sy=0.0, max_outer=20, max_cycles=50)

    assert info["converged"]
    np.testing.assert_allclose(heads[:, 0, 0, 0], [3.0, 3.5], rtol=0.0, atol=1e-10)
    period_rates = [row["drn_budget"]["outflow"] for row in info["periods"]]
    np.testing.assert_allclose(period_rates, [1.0, 1.5], rtol=0.0, atol=1e-10)
    assert info["periods"][0]["drn_volume_budget"]["outflow"] == 1.0
    assert info["periods"][1]["drn_volume_budget"]["outflow"] == 1.5
    assert info["accepted_exchange_volume_totals"]["drn"]["outflow"] == 2.5

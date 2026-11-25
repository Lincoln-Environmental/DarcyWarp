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


def test_npf_solve_rejects_unintegrated_boundary_kernel_explicitly():
    """Check NPF package fields cannot silently enter a solver that ignores them."""
    shape = (2, 2, 3)
    fixed = np.zeros(shape, dtype=np.int32)
    fixed[:, :, 0] = 1
    mask = np.zeros(shape, dtype=np.int32)
    mask[0, 0, 1] = 1
    model = WarpDarcySolver3D(nx=3, ny=2, nz=2, dx=1.0, device="cpu",
        implementation="fast", formulation="mf6_6_6_2_npf")
    model.build_from_K_fields(kx_field=np.ones(shape), ky_field=np.ones(shape), kz_field=np.ones(shape),
        active=np.ones(shape, dtype=np.int32), bc_mask=fixed, bc_values=np.zeros(shape),
        rhs=np.zeros(shape), drn_conductance=1.0, drn_elevation=0.0, drn_mask=mask)

    with pytest.raises(ValueError, match="does not yet assemble DRN/RIV"):
        model.solve(max_outer=1)

# SPDX-License-Identifier: AGPL-3.0-or-later
"""Analytical checks for sparse multi-entry DRN/RIV in the native 3D NPF path.

Multiple entries in one cell must evaluate each physical flux independently
and sum the cell contributions without averaging stages or elevations.
"""
import numpy as np
import pytest
import warp as wp

from DARCY_WARP_PACKAGE.physics.drn_riv_3d import (
    csr_entry_terms,
    csr_from_boundary_field,
    drn_terms,
    riv_terms,
    validate_boundary_entries,
)
from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D

DEVICES = ["cpu", "cuda:0"]


def _cuda_available(device):
    if not device.startswith("cuda"):
        return True
    wp.init()
    return wp.is_cuda_available()


def _one_cell_model(device, **boundary):
    shape = (1, 1, 1)
    model = WarpDarcySolver3D(nx=1, ny=1, nz=1, dx=1., dy=1., dz=1.,
        formulation="mf6_6_6_2_npf", solver="kcycle", implementation="fast", device=device)
    model.build_from_K_fields(kx_field=np.ones(shape), ky_field=np.ones(shape),
        kz_field=np.ones(shape), active=np.ones(shape, dtype=np.int32),
        bc_mask=np.zeros(shape, dtype=np.int32), bc_values=np.zeros(shape),
        rhs=np.full(shape, 1.0), initial_head=np.zeros(shape), **boundary)
    return model


@pytest.mark.parametrize("device", DEVICES)
def test_two_drains_in_one_cell_match_closed_form(device):
    """One cell with drains (c=1, e=2) and (c=1, e=1) under unit source.

    Outflow max(h-2,0)+max(h-1,0)=1 has the exact root h=2 where only the
    lower drain carries the source; no single averaged drain reproduces it.
    """
    if not _cuda_available(device):
        pytest.skip("CUDA is unavailable")
    model = _one_cell_model(device,
        drn_entries=[((0, 0, 0), 1.0, 2.0), ((0, 0, 0), 1.0, 1.0)])
    try:
        head, info = model.solve(max_outer=200)
        assert info["converged"]
        np.testing.assert_allclose(head[0, 0, 0], 2.0, rtol=0., atol=1e-9)
        entries = np.sort(info["drn_entry_rates"])
        np.testing.assert_allclose(entries, [0., 1.], rtol=0., atol=1e-9)
        np.testing.assert_allclose(info["drn_budget"]["outflow"], 1.0, rtol=0., atol=1e-9)
    finally:
        model.close()


@pytest.mark.parametrize("device", DEVICES)
def test_two_river_entries_in_one_cell_stay_separate(device):
    """Entries with different stages/bottoms keep independent flux laws.

    Outflow max(h,2)-3 + max(h,1)-2 = 1 has the exact root h=3, where the
    lower-bottom entry carries the source and the deep entry sits at its
    kink. Averaging the two stages into one entry would change the root.
    """
    if not _cuda_available(device):
        pytest.skip("CUDA is unavailable")
    model = _one_cell_model(device,
        riv_entries=[((0, 0, 0), 1.0, 3.0, 2.0), ((0, 0, 0), 1.0, 2.0, 1.0)])
    try:
        head, info = model.solve(max_outer=200)
        assert info["converged"]
        np.testing.assert_allclose(head[0, 0, 0], 3.0, rtol=0., atol=1e-9)
        entries = np.sort(info["riv_entry_rates"])
        np.testing.assert_allclose(entries, [0., 1.], rtol=0., atol=1e-9)
    finally:
        model.close()


def test_entry_terms_match_dense_reference_terms():
    """Dense fields reduced to CSR reproduce the dense flux laws exactly."""
    shape = (2, 3, 4)
    rng = np.random.default_rng(7)
    head = rng.normal(size=shape)
    drn_c = np.zeros(shape)
    drn_c[0, 1, 2] = 3.5
    drn_c[1, 0, 1] = 0.25
    elevation = np.full(shape, 0.3)
    dense = drn_terms(head=head, conductance=drn_c, elevation=elevation)
    entries = csr_from_boundary_field(conductance=drn_c, levels=[elevation], shape=shape)
    unused_entry_rates, terms = csr_entry_terms(head_flat=head.ravel(), entries=entries,
        package="DRN", shape=shape)
    np.testing.assert_allclose(terms.discharge, dense.discharge, rtol=1e-15, atol=1e-300)
    np.testing.assert_allclose(terms.diagonal, dense.diagonal, rtol=1e-15, atol=0.)
    np.testing.assert_allclose(terms.rhs, dense.rhs, rtol=1e-15, atol=1e-300)

    riv_c = np.zeros(shape)
    riv_c[0, 2, 3] = 2.0
    riv_c[1, 1, 0] = 1.0
    stage = np.full(shape, 0.9)
    bottom = np.full(shape, 0.2)
    dense_riv = riv_terms(head=head, conductance=riv_c, stage=stage, bottom=bottom)
    riv_entries = csr_from_boundary_field(conductance=riv_c, levels=[stage, bottom], shape=shape)
    unused_rates, river = csr_entry_terms(head_flat=head.ravel(), entries=riv_entries,
        package="RIV", shape=shape)
    np.testing.assert_allclose(river.discharge, dense_riv.discharge, rtol=1e-15, atol=1e-300)
    np.testing.assert_allclose(river.diagonal, dense_riv.diagonal, rtol=1e-15, atol=0.)


def test_entry_validation_rejects_bad_records():
    """Invalid entries fail before any model mutation."""
    shape = (1, 2, 2)
    active = np.ones(shape, dtype=np.int32)
    fixed = np.zeros(shape, dtype=np.int32)
    with pytest.raises(ValueError):
        validate_boundary_entries(entries=[((0, 0, 0), -1.0, 0.0)], package="DRN",
            shape=shape, active=active, fixed=fixed)
    with pytest.raises(ValueError):
        validate_boundary_entries(entries=[((0, 5, 0), 1.0, 0.0)], package="DRN",
            shape=shape, active=active, fixed=fixed)
    inactive = active.copy()
    inactive[0, 1, 1] = 0
    with pytest.raises(ValueError):
        validate_boundary_entries(entries=[((0, 1, 1), 1.0, 0.0)], package="DRN",
            shape=shape, active=inactive, fixed=fixed)
    with pytest.raises(ValueError):
        validate_boundary_entries(entries=[((0, 0, 0), 1.0, 1.0, 2.0)], package="RIV",
            shape=shape, active=active, fixed=fixed)  # stage below bottom


@pytest.mark.parametrize("device", DEVICES)
def test_entry_updates_replace_between_solves(device):
    """Sparse entries swap with dense fields across accepted solves."""
    if not _cuda_available(device):
        pytest.skip("CUDA is unavailable")
    model = _one_cell_model(device, drn_entries=[((0, 0, 0), 2.0, 1.0)])
    try:
        head, info = model.solve(max_outer=200)
        assert info["converged"]
        # (c=2, e=1) alone: outflow 2*(h-1)=1 -> h=1.5
        np.testing.assert_allclose(head[0, 0, 0], 1.5, rtol=0., atol=1e-9)
        model.update_drn_entries(None)
        # This accepted historical equation owns its exchange even after
        # the current package is removed. Current solves still reject it.
        np.testing.assert_allclose(model.accepted_drn_discharge(head=head, info=info), 1.)
        with pytest.raises(ValueError):
            # The isolated source is ungrounded with the drain removed.
            model.solve(max_outer=200)
        model.update_drn(drn_conductance=1.0, drn_elevation=1.0)
        head, info = model.solve(max_outer=200)
        assert info["converged"]
        np.testing.assert_allclose(head[0, 0, 0], 2.0, rtol=0., atol=1e-9)
    finally:
        model.close()


def test_entry_list_digest_changes_with_values():
    """Case digests distinguish entry values so caches cannot silently alias."""
    from DARCY_WARP_PACKAGE.validation.npf662_reference import NPF662Case
    shape = (1, 1, 1)
    base = dict(kx=np.ones(shape), ky=np.ones(shape), kz=np.ones(shape),
        active=np.ones(shape, dtype=np.int32), fixed=np.zeros(shape, dtype=np.int32),
        values=np.zeros(shape), source=np.zeros(shape), initial=np.zeros(shape),
        dx=1., dy=1., dz=1.)
    first = NPF662Case(**base, drn_entries=[((0, 0, 0), 1.0, 1.0)])
    second = NPF662Case(**base, drn_entries=[((0, 0, 0), 2.0, 1.0)])
    assert first.digest() != second.digest()
    assert first.digest() == NPF662Case(**base, drn_entries=[((0, 0, 0), 1.0, 1.0)]).digest()

# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression tests for 3D input ownership, transactional rebuilds and reuse."""
from __future__ import annotations

import numpy as np
import pytest

from DARCY_WARP_PACKAGE.physics.terrain_grid_3d import TerrainGrid3D
from DARCY_WARP_PACKAGE.solvers.session_3d import ThreeDOperatorSession
from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D


SHAPE = (2, 2, 5)


def case_fields() -> dict[str, np.ndarray]:
    fixed = np.zeros(SHAPE, dtype=np.int32)
    fixed[:, :, 0] = 1
    fixed[:, :, -1] = 1
    values = np.broadcast_to(np.linspace(10.0, 9.0, SHAPE[2]), SHAPE).copy()
    return {
        "kx_field": np.ones(SHAPE),
        "ky_field": np.ones(SHAPE),
        "kz_field": np.ones(SHAPE),
        "active": np.ones(SHAPE, dtype=np.int32),
        "bc_mask": fixed,
        "bc_values": values,
        "rhs": np.zeros(SHAPE),
        "initial_head": np.zeros(SHAPE),
    }


def interfaces() -> np.ndarray:
    return np.broadcast_to(np.arange(3.0, 0.0, -1.0)[:, None, None], (3, 2, 5)).copy()


def new_model(*, formulation: str = "legacy") -> WarpDarcySolver3D:
    return WarpDarcySolver3D(
        nx=5, ny=2, nz=2, dx=2.0, dy=3.0, dz=1.0,
        device="cpu", implementation="fast", formulation=formulation,
    )


def terrain_fields() -> dict[str, np.ndarray]:
    fields = case_fields()
    fields.pop("active")
    fields["z_interfaces"] = interfaces()
    return fields


@pytest.mark.parametrize("dimension", ["nx", "ny", "nz"])
@pytest.mark.parametrize("value", [0.5, 2.5, True])
def test_constructor_rejects_noninteger_dimensions(dimension, value):
    arguments = {"nx": 5, "ny": 2, "nz": 2, "dx": 1.0, "device": "cpu"}
    arguments[dimension] = value
    with pytest.raises(ValueError, match="integer"):
        WarpDarcySolver3D(**arguments)


@pytest.mark.parametrize("name", ["active", "bc_mask"])
@pytest.mark.parametrize("value", [0.5, 1.5, 2**32])
def test_cartesian_builder_rejects_masks_before_integer_cast(name, value):
    fields = case_fields()
    fields[name] = fields[name].astype(np.float64)
    fields[name][0, 0, 2] = value
    with new_model() as model:
        with pytest.raises(ValueError, match="binary|0/1"):
            model.build_from_K_fields(**fields)


@pytest.mark.parametrize("value", [0.5, 1.5, 2**32])
def test_terrain_builder_rejects_fractional_and_overflowing_boundary_masks(value):
    fields = terrain_fields()
    fields["bc_mask"] = fields["bc_mask"].astype(np.float64)
    fields["bc_mask"][0, 0, 2] = value
    with new_model() as model:
        with pytest.raises(ValueError, match="binary|0/1"):
            model.build_from_terrain(**fields)


@pytest.mark.parametrize("field", ["rhs", "initial_head", "bc_values"])
def test_failed_cartesian_rebuild_retains_live_model(field):
    fields = case_fields()
    with new_model() as model:
        model.build_from_K_fields(**fields)
        session = model._session
        conductivity = model._kx_field.copy()
        replacement = case_fields()
        replacement[field][0, 0, 2] = np.nan
        with pytest.raises(ValueError):
            model.build_from_K_fields(**replacement)
        assert model._session is session
        assert not session._closed
        assert model.geometry == "cartesian"
        np.testing.assert_array_equal(model._kx_field, conductivity)
        head, info = model.solve(max_cycles=100, rel_tol=1e-11, abs_tol_min=1e-12)
        assert info["converged"]
        np.testing.assert_allclose(head, fields["bc_values"], atol=1e-7, rtol=0.0)


def test_failed_terrain_rebuild_does_not_close_cartesian_session():
    with new_model() as model:
        model.build_from_K_fields(**case_fields())
        session = model._session
        replacement = terrain_fields()
        replacement["rhs"][0, 0, 2] = np.nan
        with pytest.raises(ValueError):
            model.build_from_terrain(**replacement)
        assert model._session is session
        assert not session._closed
        assert model.geometry == "cartesian"


def test_failed_cartesian_rebuild_retains_terrain_geometry_and_workspace():
    with new_model() as model:
        fields = terrain_fields()
        model.build_from_terrain(**fields)
        first, info = model.solve(max_cycles=200)
        assert info["converged"]
        session = model._session
        workspace = session.workspace
        invalid = case_fields()
        invalid["rhs"][0, 0, 2] = np.nan
        with pytest.raises(ValueError):
            model.build_from_K_fields(**invalid)
        assert model.geometry == "terrain"
        assert model._session is session
        assert session.workspace is workspace
        second, info = model.solve(max_cycles=200)
        assert info["converged"]
        np.testing.assert_allclose(second, first, atol=1e-9, rtol=0.0)


def test_layered_initial_guess_can_be_cleared_and_resolved():
    fields = case_fields()
    with new_model(formulation="mf6_6_6_2_npf") as model:
        model.build_from_layers(z_interfaces=interfaces(), **fields)
        model.update_initial_head(None)
        head, info = model.solve(max_outer=20, max_cycles=300)
        assert info["converged"]
        np.testing.assert_allclose(head, fields["bc_values"], atol=1e-7, rtol=0.0)


def test_raw_face_update_invalidates_original_conductivity():
    with new_model() as model:
        model.build_from_K_fields(**case_fields())
        changed = tuple(2.0 * face for face in model._session.faces)
        model.update_face_conductance(*changed)
        assert model._kx_field is None
        assert model._ky_field is None
        assert model._kz_field is None
        with pytest.raises(ValueError, match="kx_field|conductivity|K fields"):
            model.solve(unconfined=True)


def integer_session() -> ThreeDOperatorSession:
    with new_model() as model:
        model.build_from_K_fields(**case_fields())
        session = model._session
        return ThreeDOperatorSession(
            faces=tuple(face.astype(np.int64) for face in session.faces),
            active=session.active, bc_mask=session.bc_mask,
            bc_values=session.bc_values.astype(np.int64),
            rhs=np.zeros(SHAPE, dtype=np.int64),
            initial_head=np.zeros(SHAPE, dtype=np.int64),
            dx=2.0, dy=3.0, dz=1.0,
        )


def test_session_does_not_truncate_fractional_updates_into_integer_arrays():
    session = integer_session()
    try:
        session.update_rhs(np.full(SHAPE, 0.25))
        session.update_boundary_values(np.full(SHAPE, 9.75))
        session.update_face_conductance(tuple(0.5 * face for face in session.faces))
        np.testing.assert_array_equal(session.rhs, np.full(SHAPE, 0.25))
        np.testing.assert_array_equal(session.bc_values, np.full(SHAPE, 9.75))
        assert session.faces[0][0, 0, 0] == 0.5
    finally:
        session.close()


@pytest.mark.parametrize("extra", ["surface_elevation", "zeta_faces"])
def test_terrain_geometry_rejects_ambiguous_input_modes(extra):
    arguments = {"z_interfaces": interfaces(), "dx": 2.0, "dy": 3.0}
    if extra == "surface_elevation":
        arguments[extra] = np.full((2, 5), 100.0)
    else:
        arguments[extra] = np.array([0.0, 20.0, 30.0])
    with pytest.raises(ValueError, match="either|exclusive|combine"):
        TerrainGrid3D(**arguments)


@pytest.mark.parametrize("shape", [(3, 0, 5), (3, 2, 0)])
def test_terrain_geometry_requires_nonempty_horizontal_dimensions(shape):
    with pytest.raises(ValueError, match="nonempty|positive"):
        TerrainGrid3D(z_interfaces=np.zeros(shape), dx=1.0, dy=1.0)


@pytest.mark.parametrize("all_fixed", [False, True])
def test_returned_cpu_heads_are_snapshots_not_reusable_workspace_views(all_fixed):
    fields = case_fields()
    if all_fixed:
        fields["bc_mask"][:] = 1
    with new_model() as model:
        model.build_from_K_fields(**fields)
        first, info = model.solve(max_cycles=100)
        assert info["converged"]
        snapshot = first.copy()
        model.update_boundary_values(fields["bc_values"] + 1.0)
        second, info = model.solve(max_cycles=100)
        assert info["converged"]
        assert info["workspace_reused"]
        np.testing.assert_array_equal(first, snapshot)
        assert not np.shares_memory(first, second)
        np.testing.assert_allclose(second, snapshot + 1.0, atol=1e-6, rtol=0.0)

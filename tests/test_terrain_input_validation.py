"""Reject invalid terrain boundary masks before integer conversion or rebuild."""

import numpy as np
import pytest

from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D


def terrain_fields(*, shape):
    nz, ny, nx = shape
    fixed = np.zeros(shape=shape, dtype=np.int32)
    fixed[:, :, [0, -1]] = 1
    head = np.broadcast_to(array=np.linspace(start=1.0, stop=2.0, num=nx), shape=shape).copy()
    return dict(surface_elevation=np.full(shape=(ny, nx), fill_value=3.0),
                zeta_faces=np.arange(stop=nz + 1, dtype=float),
                kx_field=1.0, ky_field=1.0, kz_field=1.0,
                bc_mask=fixed, bc_values=head, rhs=np.zeros(shape=shape),
                initial_head=head.copy())


@pytest.mark.parametrize("value", [0.9, 1.9, -0.1, np.nan, np.inf, -np.inf, 2.0])
def test_terrain_rejects_original_nonbinary_mask_and_preserves_model(value):
    shape = (2, 3, 4)
    fields = terrain_fields(shape=shape)
    with WarpDarcySolver3D(nx=shape[2], ny=shape[1], nz=shape[0], dx=1.0,
                           implementation="fast", device="cpu") as model:
        model.build_from_terrain(**fields)
        session = model._session
        expected, info = model.solve()
        assert info["converged"]
        invalid = dict(fields)
        invalid["bc_mask"] = fields["bc_mask"].astype(dtype=float)
        invalid["bc_mask"][0, 1, 1] = value
        with pytest.raises(ValueError, match="bc_mask"):
            model.build_from_terrain(**invalid)
        assert model._session is session
        actual, info = model.solve()
        assert info["converged"]
        np.testing.assert_allclose(actual=actual, desired=expected, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("dtype", [np.bool_, np.int32, np.float64])
def test_terrain_accepts_binary_masks_and_owns_the_input(dtype):
    shape = (2, 3, 4)
    fields = terrain_fields(shape=shape)
    fields["bc_mask"] = fields["bc_mask"].astype(dtype=dtype)
    expected = fields["bc_mask"].copy()
    with WarpDarcySolver3D(nx=shape[2], ny=shape[1], nz=shape[0], dx=1.0,
                           implementation="fast", device="cpu") as model:
        model.build_from_terrain(**fields)
        fields["bc_mask"][...] = 0
        np.testing.assert_array_equal(actual=model._session.bc_mask, desired=expected)
        assert model._session.bc_mask.dtype == np.int32

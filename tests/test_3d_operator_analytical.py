"""CPU analytical checks for the structured 3D FV conductance builder."""

import numpy as np

from DARCY_WARP_PACKAGE.physics.budget_3d import boundary_interface_flux
from DARCY_WARP_PACKAGE.solvers_3d import build_7point_face_conductance_from_k


def _faces(kx, ky, kz, *, dx=1.0, dy=1.0, dz=1.0):
    active = np.ones_like(kx, dtype=np.int32)
    return build_7point_face_conductance_from_k(
        kx_field=kx, ky_field=ky, kz_field=kz, active=active,
        dx=dx, dy=dy, dz=dz,
    )


def _flux(head, boundary, faces):
    return boundary_interface_flux(
        head=head, boundary_mask=boundary, active=np.ones_like(head, dtype=np.int32),
        tx_p=faces[0], tx_m=faces[1], ty_p=faces[2], ty_m=faces[3],
        tz_p=faces[4], tz_m=faces[5],
    )


def test_homogeneous_horizontal_linear_flux():
    shape = (1, 1, 5)
    k = np.full(shape, 3.0)
    faces = _faces(k, k, k, dx=2.0, dy=4.0, dz=5.0)
    head = np.array([[[10.0, 9.0, 8.0, 7.0, 6.0]]])
    boundary = np.zeros(shape, dtype=np.int32); boundary[:, :, 0] = 1
    expected = 3.0 * 4.0 * 5.0 / 2.0
    assert np.isclose(_flux(head, boundary, faces), expected)


def test_homogeneous_vertical_linear_flux():
    shape = (4, 1, 1)
    k = np.full(shape, 2.0)
    faces = _faces(k, k, k, dx=3.0, dy=4.0, dz=2.0)
    head = np.array([[[8.0]], [[7.0]], [[6.0]], [[5.0]]])
    boundary = np.zeros(shape, dtype=np.int32); boundary[0, :, :] = 1
    expected = 2.0 * 3.0 * 4.0 / 2.0
    assert np.isclose(_flux(head, boundary, faces), expected)


def test_two_material_vertical_series_resistance():
    k = np.array([[[2.0]], [[8.0]]])
    faces = _faces(k, k, k, dx=1.0, dy=1.0, dz=1.0)
    head = np.array([[[10.0]], [[0.0]]])
    boundary = np.zeros_like(k, dtype=np.int32); boundary[0, 0, 0] = 1
    # The Dirichlet heads are cell-centre heads, so each material contributes
    # a half-cell resistance at the interface-to-centre distance.
    expected = 10.0 / (0.5 / 2.0 + 0.5 / 8.0)
    assert np.isclose(_flux(head, boundary, faces), expected)


def test_parallel_horizontal_layers_sum_fluxes():
    k = np.array([[[2.0, 2.0]], [[5.0, 5.0]]])
    faces = _faces(k, k, k, dx=2.0, dy=1.0, dz=3.0)
    head = np.array([[[10.0, 8.0]], [[10.0, 8.0]]])
    boundary = np.zeros_like(k, dtype=np.int32); boundary[:, :, 0] = 1
    expected = (2.0 + 5.0) * 1.0 * 3.0 / 2.0 * 2.0
    assert np.isclose(_flux(head, boundary, faces), expected)


def test_isotropic_rectangular_grid_geometric_factors():
    shape = (2, 1, 2)
    k = np.full(shape, 4.0)
    faces = _faces(k, k, k, dx=2.0, dy=3.0, dz=5.0)
    assert np.isclose(faces[0][0, 0, 0], 4.0 * 3.0 * 5.0 / 2.0)
    assert np.isclose(faces[4][0, 0, 0], 4.0 * 2.0 * 3.0 / 5.0)
    assert np.allclose(faces[2], 0.0) and np.allclose(faces[3], 0.0)
